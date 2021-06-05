import copy
import random
from dataclasses import dataclass

# from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from simple_bot import SimpleBot
from engine import Game
from entity import Position, Shipyard, MoveCommand, SpawnShipCommand


@dataclass
class Transition:
    features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    nextfeatures: np.ndarray


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
            cell_features[9, y, x] = nearest_dist / 64
    # Feature 10 - turns remaining - normalized
    cell_features[10] = (game.max_turns - game.turn) / 500
    return cell_features


class ReplayMemory(IterableDataset):
    def __init__(self, size, device):
        self.size = size
        self.features = torch.zeros(size=(size, 11, 64, 64), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size=(size, 64, 64), dtype=torch.int16, device=device)
        self.rewards = torch.zeros(size=(size, 64, 64), dtype=torch.float32, device=device)
        self.nextfeatures = torch.zeros(size=(size, 11, 64, 64), dtype=torch.float32, device=device)
        self.terminal = torch.zeros(size=(size,), dtype=torch.bool, device=device)
        self.num_entries = 0
        self._idx = 0

    def add_sample(self, new_feat, new_actions, new_rewards, new_nextfeat, terminal=False):
        self.features[self._idx] = torch.tile(torch.as_tensor(new_feat, dtype=torch.float32), (2, 2))[:, :64, :64]
        self.actions[self._idx] = torch.tile(torch.as_tensor(new_actions, dtype=torch.int16), (2, 2))[:64, :64]
        self.rewards[self._idx] = torch.tile(torch.as_tensor(new_rewards, dtype=torch.float32), (2, 2))[:64, :64]
        self.nextfeatures[self._idx] = torch.tile(torch.as_tensor(
            new_nextfeat, dtype=torch.float32), (2, 2))[:, :64, :64]
        self.terminal[self._idx] = terminal

        self._idx = (self._idx + 1) % self.size
        if self.num_entries < self.size:
            self.num_entries += 1

    def __iter__(self):
        return self

    def __next__(self):
        idx = random.randint(0, self.num_entries - 1)
        return self.features[idx], self.actions[idx], self.rewards[idx], self.nextfeatures[idx], self.terminal[idx]


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=16, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=5, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
        ).float()

    def forward(self, x):
        return self.model.forward(x)


class DeepRLBot:
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


def augment(batch):
    feat, actions, rewards, nextfeat, term = batch
    num_rot = random.randint(0, 3)
    if random.random() < 0.5:
        feat = torch.rot90(torch.flip(feat, (2,)), num_rot, (2, 3))
        actions = torch.rot90(torch.flip(actions, (1,)), num_rot, (1, 2))
        rewards = torch.rot90(torch.flip(rewards, (1,)), num_rot, (1, 2))
        nextfeat = torch.rot90(torch.flip(nextfeat, (2,)), num_rot, (2, 3))

        if num_rot == 0:
            rep = ((1, 3), (3, 1))
        elif num_rot == 1:
            rep = ((1, 4), (4, 1), (2, 3), (3, 2))
        elif num_rot == 2:
            rep = ((2, 4), (4, 2))
        else:
            rep = ((1, 2), (2, 1), (3, 4), (4, 3))
    else:
        feat = torch.rot90(feat, num_rot, (2, 3))
        actions = torch.rot90(actions, num_rot, (1, 2))
        rewards = torch.rot90(rewards, num_rot, (1, 2))
        nextfeat = torch.rot90(nextfeat, num_rot, (2, 3))

        if num_rot == 0:
            rep = ()
        elif num_rot == 1:
            rep = ((1, 2), (2, 3), (3, 4), (4, 1))
        elif num_rot == 2:
            rep = ((1, 3), (2, 4), (3, 1), (4, 2))
        else:
            rep = ((1, 4), (2, 1), (3, 2), (4, 3))
    if len(rep) > 0:
        new_actions = torch.clone(actions)
        for a, b in rep:
            new_actions[actions == a] = b
        return feat, new_actions, rewards, nextfeat, term
    return feat, actions, rewards, nextfeat, term


def train_step(batch, model, frozen_model, loss_fn, optimizer):
    feat, actions, rewards, nextfeat, term = batch
    # feat, actions, rewards, nextfeat, term = batch
    actions[actions == -1] = 0

    q_prediction = model.forward(feat).gather(1, actions.unsqueeze(1).long()).squeeze()
    expected_return = frozen_model.forward(nextfeat)
    q_target = torch.clone(rewards)
    for action, disp in enumerate(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))):
        mask = actions[~term] == action
        temp = q_target[~term]
        temp[mask] += DISCOUNT_FACTOR * (
            # gamma * maxQ' term
            torch.max(expected_return.roll((-disp[0], -disp[1]), (2, 3)), 1).values[~term][mask]
            # zeros out return for dead ships
            * (nextfeat.roll((-disp[0], -disp[1]), (2, 3))[~term][:, 1, :, :][mask] == 1)
        )
        q_target[~term] = temp
    loss = loss_fn(q_prediction[feat[:, 1] == 1], q_target[feat[:, 1] == 1].cuda())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


# def evaluate_performance(model, device_, seeds=None):
#     import time
#     if seeds is None:
#         seeds = [491777, 221533, 427817, 849970, 230048, 208860, 253589, 794821, 106587, 659861]
#     wins_count = 0
#     ship_counts = []
#     banks = []
#     bank_ratios = []
#     for seed in tqdm(seeds, desc='Evaluating Performance'):
#         current_game = Game(2, size=64, seed=seed)
#         rl_bot, opp_bot = DeepRLBot(0, model, device_), SimpleBot(1)
#         ships_set = set()
#         for current_t in range(current_game.max_turns):
#             current_game.step([rl_bot.generate_commands(current_game),
#                                opp_bot.generate_commands(current_game)])
#             for s in current_game.ships.values():
#                 if s.owner_id == 0:
#                     ships_set.add(s.id)
#         ship_counts.append(len(ships_set))
#         if current_game.winner_id == 0:
#             wins_count += 1
#         banks.append(current_game.banks[0])
#         bank_ratios.append(current_game.banks[0] / current_game.banks[1])
#     time.sleep(1)
#     print(f'Winrate: {wins_count * 100 // len(seeds)}% -- Mean Ships: {sum(ship_counts) / len(seeds)} -- Max Ships: '
#           f'{max(ship_counts)} -- Mean Bank: {sum(banks) / len(seeds)} -- Max Bank: {max(banks)}')
#     time.sleep(1)


NUM_EPISODES = 1500
TRAIN_START_THRESH = 1000
TURNS_PER_EP = 100
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-5
EPS_HI = 1
EPS_LO = 0
EPS_DECAY = 0.999

if __name__ == '__main__':
    my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    replay_memory = ReplayMemory(2500, my_device)
    # q_network = lraspp_mobilenet_v3_large(pretrained=False, num_classes=5, pretrained_backbone=False)
    q_network = QNet()
    frozen_q = copy.deepcopy(q_network)
    q_network.to(my_device)
    frozen_q.to(my_device)

    data_iter = None
    my_loss_fn = torch.nn.MSELoss()
    my_optimizer = torch.optim.RMSprop(q_network.parameters(), lr=LEARNING_RATE)
    for ep in range(NUM_EPISODES):
        g = Game(num_players=2, size=64, max_turns=TURNS_PER_EP, seed=221533)
        cur_rl_bot = DeepRLBot(0, q_network, my_device)
        opponent_bot = SimpleBot(player_id=1)

        tot_mined = 0
        if ep >= TRAIN_START_THRESH / TURNS_PER_EP:
            cur_eps = (EPS_HI - EPS_LO) * (EPS_DECAY ** (ep - TRAIN_START_THRESH / TURNS_PER_EP)) + EPS_LO
        else:
            cur_eps = 1

        for t in range(g.max_turns):
            current_features = extract_features(g, 0)

            rl_bot_commands = cur_rl_bot.generate_commands(g, eps=cur_eps)

            actions_arr = np.full(shape=(g.size, g.size), fill_value=-1)  # 0O 1N 2E 3S 4W
            rewards_arr = np.zeros(shape=(g.size, g.size))
            unprocessed_ships = set(ship.id for ship in g.ships.values() if ship.owner_id == 0)
            for command in rl_bot_commands:
                if not isinstance(command, MoveCommand):
                    continue
                ship = g.ships[command.target_id]
                unprocessed_ships.remove(ship.id)
                if command.direction == 'O':
                    actions_arr[ship.y, ship.x] = 0
                elif command.direction == 'N':
                    actions_arr[ship.y, ship.x] = 1
                elif command.direction == 'E':
                    actions_arr[ship.y, ship.x] = 2
                elif command.direction == 'S':
                    actions_arr[ship.y, ship.x] = 3
                else:
                    actions_arr[ship.y, ship.x] = 4
                resulting_pos = ship.pos + command.direction_vector
                for construct in g.constructs.values():
                    if construct.owner_id == 0 and resulting_pos == construct.pos:
                        tot_mined += ship.halite
                        rewards_arr[ship.y, ship.x] = ship.halite / 100
            for ship_id in unprocessed_ships:
                actions_arr[g.ships[ship_id].y, g.ships[ship_id].x] = 0

            g.step([rl_bot_commands, opponent_bot.generate_commands(g)])

            if not g.done:
                nextfeatures = extract_features(g, 0)
            else:
                nextfeatures = np.full(shape=(11, 64, 64), fill_value=-1)

            replay_memory.add_sample(current_features, actions_arr, rewards_arr, nextfeatures, g.done)

            if replay_memory.num_entries == TRAIN_START_THRESH:
                data_iter = iter(DataLoader(replay_memory, batch_size=BATCH_SIZE))

            if replay_memory.num_entries > TRAIN_START_THRESH:
                cur_batch = next(data_iter)
                train_step(augment(cur_batch), q_network, frozen_q, my_loss_fn, my_optimizer)

        if next(q_network.parameters()).isnan().any().item():
            raise OverflowError('Model parameters contain nan')

        print(f'Episode {ep: 3d} -- Total halite mined: {tot_mined:4d} -- Epsilon: {cur_eps:.3f}')

        if ep % 10 == 9:
            with torch.no_grad():
                num_parameters = sum(1 for _ in q_network.parameters())
                total = 0
                for _ in range(num_parameters):
                    p0, p1 = next(q_network.parameters()), next(frozen_q.parameters())
                    total += ((p0 - p1).abs() / p1.abs()).mean().item()
                print(f'Mean parameter change: {total / num_parameters:.2%}')

            frozen_q = copy.deepcopy(q_network)
        if ep % 100 == 99:
            torch.save(q_network.state_dict(), f'checkpoints/f100-dqn-conv2-{ep + 1}.pt')
