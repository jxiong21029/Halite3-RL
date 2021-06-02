import copy
import random
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from bot_fast import FastBot
from engine import Game
from entity import Position, Shipyard, MoveCommand, SpawnShipCommand


@dataclass
class Transition:
    features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_features: np.ndarray


def extract_features(game, my_id) -> np.ndarray:
    cell_features = np.zeros(shape=(11, game.size, game.size), dtype=np.float32)
    # Feature 0: cell halite
    np.copyto(dst=cell_features[0, :, :], src=game.cells[:, :, 0] / 1000)
    for ship in game.ships.values():
        if ship.owner_id == my_id:
            # Feature 1 - has ally ship - bool
            cell_features[1, ship.y, ship.x] = 1
            # Feature 2 - ally ship cargo - normalized (scaled to be between 0 and 1) number
            cell_features[2, ship.y, ship.x] = ship.halite / 1000
        else:
            # Feature 3 - has enemy ship - bool
            cell_features[3, ship.y, ship.x] = 1
            # Feature 4 - enemy ship cargo - normalized number
            cell_features[4, ship.y, ship.x] = ship.halite / 1000
    for constr in game.constructs.values():
        if isinstance(constr, Shipyard):
            if constr.owner_id == my_id:
                # Feature 5 - has ally shipyard - bool
                cell_features[5, constr.y, constr.x] = 1
            else:
                # Feature 6 - has enemy shipyard - bool
                cell_features[6, constr.y, constr.x] = 1
        else:
            if constr.owner_id == my_id:
                # Feature 7 - has ally dropoff - bool
                cell_features[7, constr.y, constr.x] = 1
            else:
                # Feature 8 - has enemy dropoff - bool
                cell_features[8, constr.y, constr.x] = 1
    # Feature 9 - distance to nearest dropoff - normalized
    for y in range(game.size):
        for x in range(game.size):
            nearest_dist = min(game.size - abs(game.size // 2 - abs(constr.y - y)) - abs(
                game.size // 2 - abs(constr.x - x))
                               for constr in game.constructs.values()
                               if constr.owner_id == my_id)
            cell_features[9, y, x] = nearest_dist / 32
    # Feature 10 - turns left - normalized
    cell_features[10] = (game.max_turns - game.turn) / 500
    return cell_features


class ReplayMemory(Dataset):
    def __init__(self, size, device):
        self.size = size
        self.features = torch.zeros(size=(size, 11, 64, 64), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size=(size, 64, 64), dtype=torch.int64, device=device)
        self.rewards = torch.zeros(size=(size, 64, 64), dtype=torch.float32, device=device)
        self.next_features = torch.zeros(size=(size, 11, 64, 64), dtype=torch.float32, device=device)
        self.terminal = torch.zeros(size=(size,), dtype=torch.bool, device=device)
        self.num_entries = 0
        self._idx = 0

    def add_sample(self, new_feat, new_actions, new_rewards, new_next_feat, terminal=False):
        self.features[self._idx] = torch.tile(torch.as_tensor(new_feat, dtype=torch.float32), (2, 2))[:, :64, :64]
        self.actions[self._idx] = torch.tile(torch.as_tensor(new_actions, dtype=torch.int64), (2, 2))[:64, :64]
        self.rewards[self._idx] = torch.tile(torch.as_tensor(new_rewards, dtype=torch.float32), (2, 2))[:64, :64]
        self.next_features[self._idx] = torch.tile(torch.as_tensor(
            new_next_feat, dtype=torch.float32), (2, 2))[:, :64, :64]
        self.terminal[self._idx] = terminal

        self._idx = (self._idx + 1) % self.size
        if self.num_entries < self.size:
            self.num_entries += 1

    def __getitem__(self, item):
        return (self.features[item], self.actions[item], self.rewards[item], self.next_features[item],
                self.terminal[item])

    def __len__(self):
        return self.num_entries


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
        ).float()

    def forward(self, x):
        return self.model.forward(x)


class QNetBot:
    def __init__(self, my_id, model, device_):
        self.id = my_id
        self.model = model
        self.shipyard = None
        self.map_starting_halite = None
        self.halite = None
        self.device = device_

    def generate_commands(self, game, eps=0):
        if self.shipyard is None:
            for shipyard in game.constructs.values():
                if shipyard.owner_id == self.id:
                    self.shipyard = shipyard
                    break
            self.map_starting_halite = np.sum(game.cells[:, :, 0])
        if self.device is not None:
            features = torch.from_numpy(extract_features(game, self.id)).float().to(self.device)
        else:
            features = torch.from_numpy(extract_features(game, self.id)).float()
        # print(features.unsqueeze(0).shape)
        self.model.eval()
        q_values = self.model.forward(features.unsqueeze(0))
        self.model.train()
        commands = {}
        queue = []  # ships that need to be checked for collisions
        next_pos = {}  # ship id: next position
        next_ships = []  # next_ships[y][x] = list of ships that will end want to move to y x
        turns_left = game.max_turns - game.turn
        self.halite = game.banks[self.id]
        for y in range(game.size):
            next_ships.append([])
            for x in range(game.size):
                next_ships[y].append([])
        for ship in game.ships.values():
            if ship.owner_id == self.id:
                if ship.halite < game.cells[ship.y][ship.x][0] // 10:
                    eps_greedy_action = 0
                elif random.random() < eps:
                    eps_greedy_action = random.randint(0, 4)
                else:
                    eps_greedy_action = torch.max(q_values[0, :, ship.y, ship.x], dim=0)[1].item()
                if eps_greedy_action != 0:
                    queue.append(ship)
                direction = 'ONESW'[eps_greedy_action]
                new_command = MoveCommand(ship.id, direction)
                commands[ship.id] = new_command
                next_pos[ship.id] = (ship.pos + new_command.direction_vector)
                next_pos[ship.id].wrap(64)
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].append(ship)
        while queue:
            ship = queue.pop()
            nx = next_pos[ship.id].x
            ny = next_pos[ship.id].y
            if len(next_ships[ny][nx]) > 1 and not (
                    turns_left <= 50 and nx == self.shipyard.x and ny == self.shipyard.y
            ):
                cur = Position(ship.x, ship.y)
                done = False
                visited = set()
                while not done:
                    cur = next_pos[game.cells[cur.y][cur.x][2]]
                    # if hits empty or enemy, then not a cycle
                    if game.cells[cur.y][cur.x][2] == -1 or game.ships[game.cells[cur.y][cur.x][2]].owner_id != self.id:
                        break
                    # if ship stops, then not a cycle
                    if cur == next_pos[game.cells[cur.y][cur.x][2]]:
                        break
                    if cur == Position(ship.x, ship.y):
                        done = True
                        continue
                    elif game.cells[cur.y][cur.x][2] in visited:
                        break
                    visited.add(game.cells[cur.y][cur.x][2])
                else:
                    continue
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].remove(ship)
                next_pos[ship.id].x = ship.x
                next_pos[ship.id].y = ship.y
                commands[ship.id] = MoveCommand(ship.id, 'O')
                queue.extend(next_ships[ship.y][ship.x])
                next_ships[ship.y][ship.x].append(ship)
        ret = list(commands.values())
        if (len(next_ships[self.shipyard.y][self.shipyard.x]) == 0
                and self.halite >= 1000
                and turns_left > min(100, game.max_turns // 2)
                and np.sum(game.cells[:, :, 0]) * 3 > self.map_starting_halite):
            ret.append(SpawnShipCommand(None))
        return ret


def evaluate_performance(model, device_, seeds=None):
    import time
    if seeds is None:
        seeds = [491777, 221533, 427817, 849970, 230048, 208860, 253589, 794821, 106587, 659861]
    wins_count = 0
    ship_counts = []
    banks = []
    bank_ratios = []
    for seed in tqdm(seeds, desc='Evaluating Performance'):
        current_game = Game(2, size=64, seed=seed)
        rl_bot, opp_fastbot = QNetBot(0, model, device_), FastBot(1)
        ships_set = set()
        for current_t in range(current_game.max_turns):
            current_game.step([rl_bot.generate_commands(current_game),
                               opp_fastbot.generate_commands(current_game)])
            for s in current_game.ships.values():
                if s.owner_id == 0:
                    ships_set.add(s.id)
        ship_counts.append(len(ships_set))
        if current_game.winner_id == 0:
            wins_count += 1
        banks.append(current_game.banks[0])
        bank_ratios.append(current_game.banks[0] / current_game.banks[1])
    time.sleep(1)
    print(f'Winrate: {wins_count * 100 // len(seeds)}% -- Mean Ships: {sum(ship_counts) / len(seeds)} -- Max Ships: '
          f'{max(ship_counts)} -- Mean Bank: {sum(banks) / len(seeds)} -- Max Bank: {max(banks)}')
    time.sleep(1)


TRAIN_START_THRESH = 1000
TURNS_PER_EP = 100
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-4
EPS_HI = 1
EPS_LO = .05
EPS_DECAY = 0.995

if __name__ == '__main__':
    my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    replay_memory = ReplayMemory(2000, my_device)
    # q_network = lraspp_mobilenet_v3_large(pretrained=False, num_classes=5, pretrained_backbone=False)
    # TODO: q_network = QNet()
    q_network = torch.nn.Conv2d(
        in_channels=11, out_channels=5, kernel_size=(7, 7), padding=(3, 3), padding_mode='circular'
    )
    frozen_q = copy.deepcopy(q_network)
    q_network.to(my_device)
    frozen_q.to(my_device)

    dataloader = None
    # loss_fn = torch.nn.SmoothL1Loss()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=LEARNING_RATE)
    for ep in range(1000):
        g = Game(num_players=2, size=64, max_turns=TURNS_PER_EP, seed=221533)
        cur_rl_bot = QNetBot(0, q_network, my_device)
        opponent_bot = FastBot(player_id=1)

        tot_mined = 0
        if ep >= TRAIN_START_THRESH / TURNS_PER_EP:
            cur_eps = (EPS_HI - EPS_LO) * (EPS_DECAY ** (ep - TRAIN_START_THRESH / TURNS_PER_EP)) + EPS_LO
        else:
            cur_eps = 1
        # for t in tqdm(range(g.max_turns), desc=f'Episode {ep:03}'):
        for t in range(g.max_turns):
            current_features = extract_features(g, 0)

            rl_bot_commands = cur_rl_bot.generate_commands(g, eps=cur_eps)

            actions_arr = np.full(shape=(g.size, g.size), fill_value=-1)  # 0O 1N 2E 3S 4W
            rewards_arr = np.zeros(shape=(g.size, g.size))
            unprocessed_ships = set(ship.id for ship in g.ships.values() if ship.owner_id == 0)
            for action in rl_bot_commands:
                if not isinstance(action, MoveCommand):
                    continue
                ship = g.ships[action.target_id]
                unprocessed_ships.remove(ship.id)
                if action.direction == 'N':
                    actions_arr[ship.y, ship.x] = 1
                elif action.direction == 'E':
                    actions_arr[ship.y, ship.x] = 2
                elif action.direction == 'S':
                    actions_arr[ship.y, ship.x] = 3
                elif action.direction == 'W':
                    actions_arr[ship.y, ship.x] = 4
                resulting_pos = ship.pos + action.direction_vector
                for construct in g.constructs.values():
                    if construct.owner_id == 0 and resulting_pos == construct.pos:
                        tot_mined += ship.halite
                        rewards_arr[ship.y, ship.x] = ship.halite / 100
            for ship_id in unprocessed_ships:
                actions_arr[g.ships[ship_id].y, g.ships[ship_id].x] = 0

            g.step([rl_bot_commands, opponent_bot.generate_commands(g)])

            is_terminal = False
            if not g.done:
                next_features = extract_features(g, 0)
            else:
                is_terminal = True
                next_features = np.full(shape=(11, 64, 64), fill_value=-1)

            if not is_terminal:
                replay_memory.add_sample(current_features, actions_arr, rewards_arr, next_features)
            else:
                replay_memory.add_sample(current_features, actions_arr, rewards_arr, next_features, terminal=True)
            if len(replay_memory) == TRAIN_START_THRESH:
                dataloader = DataLoader(replay_memory, batch_size=32, shuffle=True, pin_memory=False)

            if replay_memory.num_entries >= TRAIN_START_THRESH:
                b_feat, b_actions, b_rewards, b_nextfeat, b_term = next(iter(dataloader))
                b_feat = b_feat
                b_actions = b_actions
                b_rewards = b_rewards
                b_nextfeat = b_nextfeat
                b_actions[b_actions == -1] = 0

                q_prediction = q_network.forward(b_feat).gather(1, b_actions.unsqueeze(1)).squeeze()
                expected_return = frozen_q.forward(b_nextfeat)
                q_target = torch.clone(b_rewards)
                for action, disp in enumerate(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))):
                    mask = b_actions[~b_term] == action
                    temp = q_target[~b_term]
                    temp[mask] += DISCOUNT_FACTOR * (
                        # gamma * maxQ' term
                        torch.max(expected_return.roll((-disp[0], -disp[1]), (2, 3)), 1).values[~b_term][mask]
                        # zeros out return for dead ships
                        * (b_nextfeat.roll((-disp[0], -disp[1]), (2, 3))[~b_term][:, 1, :, :][mask] == 1)
                    )
                    q_target[~b_term] = temp
                loss = loss_fn(q_prediction[b_feat[:, 1] == 1], q_target[b_feat[:, 1] == 1].cuda())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if next(q_network.parameters()).isnan().any().item():
            raise OverflowError('Model parameters contain nan')

        print(f'Episode {ep: 3d} -- Total halite mined: {tot_mined:4d} -- Epsilon: {cur_eps:.3f}')

        if ep % 10 == 9:
            frozen_q = copy.deepcopy(q_network)
        if ep % 100 == 99:
            torch.save(q_network.state_dict(), f'checkpoints/f50-dqn-linear-{ep + 1}.pt')
