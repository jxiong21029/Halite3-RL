import copy
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from bot_fast import FastBot
from engine import Game
from entity import Position, Shipyard, MoveCommand, SpawnShipCommand


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular'),
        ).float()

    def forward(self, x):
        return self.model.forward(x)


@dataclass
class Transition:
    features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_features: np.ndarray


def extract_features(game, my_id) -> np.ndarray:
    cell_features = np.zeros(shape=(10, game.size, game.size), dtype=np.float32)
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
    # Feature 9 - turns left after returning to nearest dropoff - normalized
    for y in range(game.size):
        for x in range(game.size):
            nearest_dist = min(game.size - abs(game.size // 2 - abs(constr.y - y)) - abs(
                                    game.size // 2 - abs(constr.x - x))
                               for constr in game.constructs.values()
                               if constr.owner_id == my_id)
            cell_features[9, y, x] = (game.max_turns - game.turn - nearest_dist + 64) / 564
    return cell_features


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.features = torch.zeros(size=(size, 10, 64, 64), dtype=torch.float32)
        self.actions = torch.zeros(size=(size, 64, 64), dtype=torch.int64)
        self.rewards = torch.zeros(size=(size, 64, 64), dtype=torch.float32)
        self.next_features = torch.zeros(size=(size, 10, 64, 64), dtype=torch.float32)
        self.terminal = torch.zeros(size=(size,), dtype=torch.bool)
        self.num_entries = 0
        self._idx = 0

    def add_sample(self, new_features, new_actions, new_rewards, new_next_features, terminal=False):
        self.features[self._idx] = torch.tile(torch.as_tensor(new_features, dtype=torch.float32), (2, 2))[:, :64, :64]
        self.actions[self._idx] = torch.tile(torch.as_tensor(new_actions, dtype=torch.int64), (2, 2))[:64, :64]
        self.rewards[self._idx] = torch.tile(torch.as_tensor(new_rewards, dtype=torch.float32), (2, 2))[:64, :64]
        self.next_features[self._idx] = torch.tile(torch.as_tensor(
            new_next_features, dtype=torch.float32), (2, 2))[:, :64, :64]
        self.terminal[self._idx] = terminal

        self._idx = (self._idx + 1) % self.size
        if self.num_entries < self.size:
            self.num_entries += 1

    def sample(self, batch_size):
        if self.num_entries < batch_size:
            raise ValueError(f'Cannot sample a batch of {batch_size} from a dataset of {self.num_entries} entries')
        idxs = np.random.choice(np.arange(self.num_entries), size=batch_size)
        return (self.features[idxs], self.actions[idxs], self.rewards[idxs], self.next_features[idxs],
                self.terminal[idxs])


class QNetBot:
    def __init__(self, my_id, model: QNet):
        self.id = my_id
        self.model = model
        self.shipyard = None
        self.map_starting_halite = None
        self.halite = None

    def generate_commands(self, game):
        if self.shipyard is None:
            for shipyard in game.constructs.values():
                if shipyard.owner_id == self.id:
                    self.shipyard = shipyard
                    break
            self.map_starting_halite = np.sum(game.cells[:, :, 0])
        features = torch.from_numpy(extract_features(game, self.id)).float().to(device)
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
                greedy_action = torch.max(q_values[0, :, ship.y, ship.x], dim=0)[1].item()
                if greedy_action != 0:
                    queue.append(ship)
                direction = 'ONESW'[greedy_action]
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


def evaluate_performance():
    import time
    seeds = [491777, 221533, 427817, 849970, 230048, 208860, 253589, 794821, 106587, 659861]
    wins_count = 0
    bank_ratios = []
    for seed in tqdm(seeds, desc='Evaluating Performance'):
        current_game = Game(2, size=64, seed=seed)
        rl_bot, opp_fastbot = QNetBot(0, q_network), FastBot(1)
        for current_t in range(current_game.max_turns):
            current_game.step([rl_bot.generate_commands(current_game), opp_fastbot.generate_commands(current_game)])
        if current_game.winner_id == 0:
            wins_count += 1
        bank_ratios.append(current_game.banks[0] / current_game.banks[1])
    time.sleep(1)
    print(f'Winrate: {wins_count * 100 // len(seeds)}%')
    time.sleep(0.4)
    print(f'On average, the RL bot\'s bank is {100 * sum(bank_ratios) / len(seeds):.1f}% the size of FastBot\'s')
    time.sleep(1)


TRAIN_START_THRESH = 1000
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    replay_memory = ReplayMemory(10000)
    q_network = QNet()
    frozen_q = copy.deepcopy(q_network)
    q_network.to(device)
    frozen_q.to(device)
    for ep in range(1000):
        g = Game(num_players=2, size=64)
        teacher_bot = FastBot(player_id=0, eps=0.1)
        opponent_bot = FastBot(player_id=1)

        for t in tqdm(range(g.max_turns), desc=f'Episode {ep:03}'):
            current_features = extract_features(g, 0)

            teacher_commands = teacher_bot.generate_commands(g)

            actions_arr = np.full(shape=(g.size, g.size), fill_value=-1)  # 0O 1N 2E 3S 4W
            rewards_arr = np.zeros(shape=(g.size, g.size))
            unprocessed_ships = set(ship.id for ship in g.ships.values() if ship.owner_id == 0)
            for action in teacher_commands:
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
                        rewards_arr[ship.y, ship.x] = ship.halite / 100
            for ship_id in unprocessed_ships:
                actions_arr[g.ships[ship_id].y, g.ships[ship_id].x] = 0

            g.step([teacher_commands, opponent_bot.generate_commands(g)])

            is_terminal = False
            if not g.done:
                next_features = extract_features(g, 0)
            else:
                is_terminal = True
                next_features = np.full(shape=(10, 64, 64), fill_value=-1)

            if not is_terminal:
                replay_memory.add_sample(current_features, actions_arr, rewards_arr, next_features)
            else:
                replay_memory.add_sample(current_features, actions_arr, rewards_arr, next_features, terminal=True)

            if replay_memory.num_entries >= TRAIN_START_THRESH:
                batch_feat, batch_actions, batch_rewards, batch_nextfeat, batch_term = replay_memory.sample(BATCH_SIZE)
                batch_feat = batch_feat.to(device)
                batch_actions = batch_actions.to(device)
                batch_rewards = batch_rewards.to(device)
                batch_nextfeat = batch_nextfeat.to(device)
                batch_actions[batch_actions == -1] = 0
                q_prediction = q_network.forward(batch_feat).gather(1, batch_actions.unsqueeze(1)).squeeze()
                expected_return = frozen_q.forward(batch_nextfeat)
                q_target = torch.clone(batch_rewards)
                for action, disp in enumerate(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))):
                    mask = batch_actions[~batch_term] == action
                    q_target[~batch_term][mask] += DISCOUNT_FACTOR * (
                        # gamma * maxQ' term
                        torch.max(expected_return.roll((-disp[0], -disp[1]), (2, 3)), 1).values[~batch_term][mask]
                        # zeros out return for dead ships
                        * (batch_nextfeat.roll((-disp[0], -disp[1]), (2, 3))[:, 1, :, :][~batch_term][mask] == 1)
                    )
                criterion = torch.nn.MSELoss()
                loss = criterion(q_prediction[batch_feat[:, 1] == 1], q_target[batch_feat[:, 1] == 1])
                # loss = (q_target - q_prediction)[batch_feat[:, 1] == 1].pow(2).sum()
                # print('Loss: ', loss)
                q_network.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in q_network.parameters():
                        param -= LEARNING_RATE * param.grad
                        # print(param.grad)

            # TODO
            #   Make mapgen deterministic (seeded) in order to create an effective validation dataset

        if ep % 10 == 9:
            frozen_q = copy.deepcopy(q_network)
        if ep % 20 == 0:
            evaluate_performance()
        if ep % 200 == 199:
            torch.save(q_network.state_dict(), f'conv_dqn_obsL_{ep + 1}.pt')
