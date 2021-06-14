import copy
import logging
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simple_bot import SimpleBot
from engine import Game
from entity import Position, MoveCommand, SpawnShipCommand
from data_utils import PrioritizedReplayMemory, extract_features, augment


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
                next_pos[ship.id].wrap(32)
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


def train_step(batch, model, frozen_model, optimizer):
    idxs, feat, actions, rewards, nextfeat, term = batch

    actions[actions == -1] = 0

    q_network_output = model.forward(feat)
    pred_q_vals = q_network_output.gather(1, actions.unsqueeze(1).long()).squeeze()
    with torch.no_grad():
        next_predict = frozen_model.forward(nextfeat)
        max_return = torch.zeros((BATCH_SIZE, 32, 32), dtype=torch.float32, device=pred_q_vals.get_device())
        for action, disp in enumerate(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))):
            mask = (actions == action)
            max_return[mask] += (
                # gamma * maxQ' term
                torch.max(next_predict.roll((-disp[0], -disp[1]), (2, 3)), 1).values[mask]
                # zeros out return for dead ships
                * (nextfeat.roll((-disp[0], -disp[1]), (2, 3))[:, 1, :, :][mask] == 1)
            )
        max_return[term] = 0
        td_target = rewards + GAMMA * max_return
    td_error = (td_target - pred_q_vals) * (feat[:, 1] == 1)
    loss = td_error.pow(2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    residual_variance = (torch.var(td_error.flatten()) / torch.var((td_target * (feat[:, 1] == 1)).flatten())).item()
    logging.debug(residual_variance)

    return td_error.mean((1, 2))


NUM_EPISODES = 10000
REPLAY_MEM_SIZE = 2500
TRAIN_START_THRESH = 100
TURNS_PER_EP = 100
BATCH_SIZE = 100
GAMMA = 0.99
LEARNING_RATE = 1e-5
EPS_HI = 0.01
EPS_LO = 0.01
EPS_DECAY = 0
PRIORITY_FACTOR = 0.6

if __name__ == '__main__':
    logging.basicConfig(
        filename='dqn_training.log',
        filemode='w',
        level=logging.DEBUG,
        datefmt='%H:%M'
    )
    my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    replay_memory = PrioritizedReplayMemory(REPLAY_MEM_SIZE, my_device)
    q_network = QNet()
    frozen_q = copy.deepcopy(q_network)
    q_network.to(my_device)
    frozen_q.to(my_device)

    def dist(a, b):
        return min(abs(a.x - b.x), 32 - abs(a.x - b.x)) + min(abs(a.y - b.y), 32 - abs(a.y - b.y))

    data_iter = None
    my_optimizer = torch.optim.RMSprop(q_network.parameters(), lr=LEARNING_RATE)
    for ep in range(NUM_EPISODES):
        g = Game(num_players=2, size=32, max_turns=TURNS_PER_EP, seed=221533)
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
                    if current_features[0, ship.y, ship.x] == 0:
                        rewards_arr[ship.y, ship.x] = -1
                    else:
                        amt_mined = (math.ceil(current_features[0, ship.y, ship.x] / 4)
                                     if ship.halite + math.ceil(current_features[0, ship.y, ship.x] / 4) <= 1000
                                     else 1000 - ship.halite)
                        rewards_arr[ship.y, ship.x] += amt_mined / 1000
                elif command.direction == 'N':
                    actions_arr[ship.y, ship.x] = 1
                elif command.direction == 'E':
                    actions_arr[ship.y, ship.x] = 2
                elif command.direction == 'S':
                    actions_arr[ship.y, ship.x] = 3
                else:
                    actions_arr[ship.y, ship.x] = 4
                resulting_pos = ship.pos + command.direction_vector
                min_cur_dist = min(
                    dist(ship, constr)
                    for constr in g.constructs.values()
                    if constr.owner_id == 0
                )
                min_nxt_dist = min(
                    dist(ship.pos + command.direction_vector, constr)
                    for constr in g.constructs.values()
                    if constr.owner_id == 0
                )
                if ship.halite >= 950 and min_nxt_dist < min_cur_dist:
                    rewards_arr[ship.y, ship.x] += 1
                for construct in g.constructs.values():
                    if (
                        construct.owner_id == 0
                        and resulting_pos == construct.pos
                        and actions_arr[ship.y, ship.x] != 0
                    ):
                        tot_mined += ship.halite
                        rewards_arr[ship.y, ship.x] = ship.halite / 100
            assert len(unprocessed_ships) == 0
            # for ship_id in unprocessed_ships:
            #     actions_arr[g.ships[ship_id].y, g.ships[ship_id].x] = 0

            g.step([rl_bot_commands, opponent_bot.generate_commands(g)])

            if not g.done:
                nextfeatures = extract_features(g, 0)
            else:
                nextfeatures = np.full(shape=(11, 32, 32), fill_value=-1)

            replay_memory.add_sample(current_features, actions_arr, rewards_arr, nextfeatures, g.done)

            if replay_memory.num_entries == TRAIN_START_THRESH:
                data_iter = iter(DataLoader(replay_memory, batch_size=BATCH_SIZE))

            if replay_memory.num_entries > TRAIN_START_THRESH:
                cur_batch = next(data_iter)
                cur_td_error = train_step(augment(cur_batch), q_network, frozen_q, my_optimizer)
                replay_memory.update_prio(cur_batch[0], cur_td_error.abs().pow(PRIORITY_FACTOR) + 0.01)
                replay_memory.recalculate_max()

        if next(q_network.parameters()).isnan().any().item():
            raise OverflowError('Model parameters contain nan')

        logging.info(f'Episode {ep: 3d} -- Total halite mined: {tot_mined:4d} -- Epsilon: {cur_eps:.3f}')
        rst = replay_memory.sumtree
        rs = replay_memory.size
        logging.debug(f'PER min: {rst[rs - 1:rs - 1 + replay_memory.num_entries].min()}')
        logging.debug(f'PER max: {rst[rs - 1:rs - 1 + replay_memory.num_entries].max()}')
        logging.debug(f'PER avg: {rst[0] / replay_memory.num_entries}')
        logging.debug(f'PER std: {rst[rs - 1:rs - 1 + replay_memory.num_entries].std()}')

        if ep % 10 == 9:
            with torch.no_grad():
                num_parameters = sum(1 for _ in q_network.parameters())

                total = 0
                for _ in range(num_parameters):
                    p0, p1 = next(q_network.parameters()), next(frozen_q.parameters())
                    total += ((p0 - p1).abs() / p1.abs()).mean().item()
                logging.debug(f'Mean parameter change: {total / num_parameters:.2%}')

            frozen_q = copy.deepcopy(q_network)
        if ep % 100 == 99:
            torch.save(q_network.state_dict(), f'checkpoints/f100-dqn-conv2-per-shaping-{ep + 1:04}.pt')
