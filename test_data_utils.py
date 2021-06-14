import math
import pickle
import random

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from entity import MoveCommand
from engine import Game
from simple_bot import SimpleBot
from data_utils import ReplayMemory, PrioritizedReplayMemory
from dqn import BATCH_SIZE, extract_features, augment


@pytest.fixture(scope='session')
def example_game():
    g = Game(num_players=2, size=32)
    bot0, bot1 = SimpleBot(0), SimpleBot(1)
    for t in range(300):
        g.step([bot0.generate_commands(g, eps=0.2), bot1.generate_commands(g, eps=0.2)])
    return g


@pytest.fixture(scope='session')
def example_dataloader():
    def dist(a, b):
        return min(abs(a.x - b.x), 32 - abs(a.x - b.x)) + min(abs(a.y - b.y), 32 - abs(a.y - b.y))

    ret = ReplayMemory(2000, device=torch.device('cuda:0'))
    for ep in range(5):
        g = Game(num_players=2, size=32)
        bot0, bot1 = SimpleBot(0), SimpleBot(1)
        for t in range(g.max_turns):
            current_features = extract_features(g, 0)

            commands = bot0.generate_commands(g, eps=0.2)

            actions_arr = np.full(shape=(g.size, g.size), fill_value=-1)  # 0O 1N 2E 3S 4W
            rewards_arr = np.zeros(shape=(g.size, g.size))
            unprocessed_ships = set(ship.id for ship in g.ships.values() if ship.owner_id == 0)
            for command in commands:
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
                        rewards_arr[ship.y, ship.x] = ship.halite / 100
            # assert len(unprocessed_ships) == 0
            for ship_id in unprocessed_ships:
                actions_arr[g.ships[ship_id].y, g.ships[ship_id].x] = 0
                if current_features[0, g.ships[ship_id].y, g.ships[ship_id].x] == 0:
                    rewards_arr[g.ships[ship_id].y, g.ships[ship_id].x] = -1

            g.step([commands, bot1.generate_commands(g, eps=0.2)])

            if not g.done:
                nextfeatures = extract_features(g, 0)
            else:
                nextfeatures = np.full(shape=(11, 32, 32), fill_value=-1)

            ret.add_sample(current_features, actions_arr, rewards_arr, nextfeatures, g.done)

    idxs, feat, actions, rewards, nextfeat, term = next(iter(DataLoader(ret, batch_size=BATCH_SIZE)))

    count = 0
    for b in range(32):
        for y in range(32):
            for x in range(32):
                if feat[b, 1, y, x] == 0:
                    continue
                if feat[b, 0, y, x] == 0 and actions[b, y, x] == 0:
                    assert rewards[b, y, x] == -1
                    count += 1
                else:
                    assert rewards[b, y, x] >= 0

    return DataLoader(ret, batch_size=BATCH_SIZE)


@pytest.fixture
def example_data_iter(example_dataloader):
    return iter(example_dataloader)


def test_extract_features_bounds(example_game):
    assert np.min(extract_features(example_game, 0)) >= 0
    assert np.min(extract_features(example_game, 1)) >= 0
    assert np.max(extract_features(example_game, 0)[1:]) <= 1
    assert np.max(extract_features(example_game, 1)[1:]) <= 1


def test_data_loader_yields_unique_entries(example_data_iter):
    assert not all((next(example_data_iter)[i] == next(example_data_iter)[i]).all() for i in range(5))


def test_replay_memory_dtype(example_data_iter):
    for _ in range(10):
        idxs, feat, actions, rewards, nextfeat, term = next(example_data_iter)
        assert idxs.dtype == torch.int64
        assert feat.dtype == torch.float32
        assert actions.dtype == torch.int16
        assert rewards.dtype == torch.float32
        assert nextfeat.dtype == torch.float32
        assert term.dtype == torch.bool


def test_replay_memory_shape(example_data_iter):
    for _ in range(10):
        idxs, feat, actions, rewards, nextfeat, term = next(example_data_iter)
        assert idxs.shape == (BATCH_SIZE,)
        assert feat.shape == (BATCH_SIZE, 11, 32, 32)
        assert actions.shape == (BATCH_SIZE, 32, 32)
        assert rewards.shape == (BATCH_SIZE, 32, 32)
        assert nextfeat.shape == (BATCH_SIZE, 11, 32, 32)
        assert term.shape == (BATCH_SIZE,)


def test_replay_memory_bounds(example_data_iter):
    for _ in range(100):
        idxs, feat, actions, rewards, nextfeat, term = next(example_data_iter)
        assert torch.min(idxs) >= 0
        assert torch.max(idxs) < 2000
        assert torch.min(feat) >= 0
        assert torch.max(feat[:, 1:]) <= 1
        assert torch.min(actions[actions != -1]) >= 0
        assert (feat[:, 1][actions == -1] == 0).all()
        assert (feat[:, 1][actions != -1] == 1).all()
        assert torch.max(actions) <= 4
        assert torch.min(rewards) >= -1
        assert torch.max(rewards) <= 10
        assert torch.min(nextfeat[~term]) >= 0
        assert torch.min(nextfeat[~term]) >= -1
        assert torch.max(nextfeat[:, 1:]) <= 1


def test_augment_dtype(example_data_iter):
    idxs, feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
    assert feat.dtype == torch.float32
    assert actions.dtype == torch.int16
    assert rewards.dtype == torch.float32
    assert nextfeat.dtype == torch.float32
    assert term.dtype == torch.bool


def test_augment_shape(example_data_iter):
    for _ in range(10):
        idxs, feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        assert idxs.shape == (BATCH_SIZE,)
        assert feat.shape == (BATCH_SIZE, 11, 32, 32)
        assert actions.shape == (BATCH_SIZE, 32, 32)
        assert rewards.shape == (BATCH_SIZE, 32, 32)
        assert nextfeat.shape == (BATCH_SIZE, 11, 32, 32)
        assert term.shape == (BATCH_SIZE,)


def test_augment_bounds(example_data_iter):
    for i in range(100):
        idxs, feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        assert torch.min(idxs) >= 0
        assert torch.max(idxs) < 2000
        assert torch.min(feat) >= 0
        assert torch.max(feat[:, 1:]) <= 1
        assert torch.min(actions[actions != -1]) >= 0
        assert (feat[:, 1][actions == -1] == 0).all()
        assert (feat[:, 1][actions != -1] == 1).all()
        assert torch.max(actions) <= 4
        assert torch.min(rewards) >= -1
        assert torch.max(rewards) <= 10
        assert torch.min(nextfeat[~term]) >= 0
        assert torch.min(nextfeat[~term]) >= -1
        assert torch.max(nextfeat[:, 1:]) <= 1


def test_augment_correctly_modifies_action_directions(example_data_iter):
    # Assumes that ally ships will never crash into other ally ships,
    # except over a shipyard/dropoff
    for _ in range(1):
        idxs, feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        for b in range(min(10, BATCH_SIZE)):
            if term[b]:
                continue
            ships_lost = 0
            for y in range(32):
                for x in range(32):
                    if actions[b, y, x] == -1:
                        assert feat[b, 1, y, x] == 0
                        continue
                    assert feat[b, 1, y, x] == 1
                    if actions[b, y, x] == 0:
                        if nextfeat[b, 1, y, x] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + dy) % 32, (x + dx) % 32] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, y, x] == 1
                                or nextfeat[b, 7, y, x] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 1:
                        if nextfeat[b, 1, (y + 1) % 32, x] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + 1 + dy) % 32, (x + dx) % 32] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, (y + 1) % 32, x] == 1
                                or nextfeat[b, 7, (y + 1) % 32, x] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 2:
                        if nextfeat[b, 1, y, (x + 1) % 32] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + dy) % 32, (x + 1 + dx) % 32] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, y, (x + 1) % 32] == 1
                                or nextfeat[b, 7, y, (x + 1) % 32] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 3:
                        if nextfeat[b, 1, (y - 1) % 32, x] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y - 1 + dy) % 32, (x + dx) % 32] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, (y - 1) % 32, x] == 1
                                or nextfeat[b, 7, (y - 1) % 32, x] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 4:
                        if nextfeat[b, 1, y, (x - 1) % 32] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + dy) % 32, (x - 1 + dx) % 32] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, y, (x - 1) % 32] == 1
                                or nextfeat[b, 7, y, (x - 1) % 32] == 1
                            )
                            ships_lost += 1
            cur_ships = torch.count_nonzero(feat[b, 1])
            next_ships = torch.count_nonzero(nextfeat[b, 1])
            assert (cur_ships - ships_lost) == next_ships or (cur_ships - ships_lost) == next_ships - 1


def test_prioritized_replay_memory_correct_total_priority(example_data_iter):
    for _ in range(10):
        prio_mem = PrioritizedReplayMemory(10, device=torch.device('cuda:0'))
        tot_prio = 0
        for i in range(10):
            idxs, feat, actions, rewards, nextfeat, term = next(example_data_iter)
            prio_mem.add_sample(feat[0], actions[0], rewards[0], nextfeat[0], term[0])
            new_prio = random.random() * 3
            tot_prio += new_prio
            prio_mem.update_prio(i, new_prio)
            diff = tot_prio - prio_mem.sumtree[0]
            assert abs(diff / tot_prio) < 0.001

        rng = np.random.default_rng()
        for i in range(10):
            prio_mem.update_prio(rng.integers(0, 10, size=10), rng.random(10) * 10)
            diff = prio_mem.sumtree[0] - prio_mem.sumtree[prio_mem.size - 1:].sum()
            assert abs(diff / prio_mem.sumtree[0]) < 0.001


def test_prioritized_replay_memory_correct_max_priority(example_data_iter):
    prio_mem = PrioritizedReplayMemory(30, device=torch.device('cuda:0'))
    for i in range(10):
        idxs, feat, actions, rewards, nextfeat, term = next(example_data_iter)
        prio_mem.add_sample(feat[0], actions[0], rewards[0], nextfeat[0], term[0])
        prio_mem.update_prio(i, 1 + random.random() * 3)
        assert prio_mem.maxprio == prio_mem.sumtree[prio_mem.size - 1:].max()

    rng = np.random.default_rng()
    for i in range(10):
        prio_mem.update_prio(rng.integers(0, 10, size=10), rng.random(10) * 10)
        prio_mem.recalculate_max()
        assert prio_mem.maxprio == prio_mem.sumtree[prio_mem.size - 1:].max()


def test_prioritized_replay_memory_correct_samples(example_data_iter):
    prio_mem = PrioritizedReplayMemory(10, device=torch.device('cuda:0'))
    for _ in range(3):
        idxs, feat, actions, rewards, nextfeat, term = next(example_data_iter)
        prio_mem.add_sample(feat[0], actions[0], rewards[0], nextfeat[0], term[0])
    prio_mem.update_prio([0, 1, 2], [1, 5, 25])

    counts = [0, 0, 0]
    data_iter = iter(DataLoader(prio_mem, batch_size=2))
    for _ in range(1550):
        batch = next(data_iter)
        idxs = batch[0]
        assert isinstance(idxs, torch.Tensor)
        assert idxs.shape[0] == 2
        assert idxs.dtype == torch.int64
        assert 0 <= idxs.min()
        assert idxs.max() <= 2
        counts[idxs[0]] += 1
        counts[idxs[1]] += 1

    import scipy.stats
    # This is not a 100% reliable test, and may give both false positives
    # and false negatives
    chisq, p = scipy.stats.chisquare(counts, (100, 500, 2500))
    assert p > 0.01
