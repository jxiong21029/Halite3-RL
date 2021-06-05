import copy
import pickle

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from entity import MoveCommand
from engine import Game
from simple_bot import SimpleBot
from dqn import BATCH_SIZE, LEARNING_RATE, ReplayMemory, QNet, extract_features, augment, train_step


@pytest.fixture
def example_game():
    with open('data/example_game.pickle', 'rb') as f:
        return pickle.load(f)


def create_simplebot_replay_memory():
    ret = ReplayMemory(2000, device=torch.device('cuda:0'))
    for ep in range(2):
        g = Game(num_players=2, size=64, seed=221533)
        bot0, bot1 = SimpleBot(0), SimpleBot(1)
        for t in range(g.max_turns):
            commands = bot0.generate_commands(g)

            features = extract_features(g, 0)
            actions = np.full(shape=(g.size, g.size), fill_value=-1)
            rewards = np.zeros(shape=(g.size, g.size))
            unprocessed_ships = set(ship.id for ship in g.ships.values() if ship.owner_id == 0)
            for command in commands:
                if not isinstance(command, MoveCommand):
                    continue
                ship = g.ships[command.target_id]
                unprocessed_ships.remove(ship.id)
                if command.direction == 'O':
                    actions[ship.y, ship.x] = 0
                elif command.direction == 'N':
                    actions[ship.y, ship.x] = 1
                elif command.direction == 'E':
                    actions[ship.y, ship.x] = 2
                elif command.direction == 'S':
                    actions[ship.y, ship.x] = 3
                else:
                    actions[ship.y, ship.x] = 4
                resulting_pos = ship.pos + command.direction_vector
                for construct in g.constructs.values():
                    if construct.owner_id == 0 and resulting_pos == construct.pos:
                        rewards[ship.y, ship.x] = ship.halite / 100
            for ship_id in unprocessed_ships:
                actions[g.ships[ship_id].y, g.ships[ship_id].x] = 0

            g.step([commands, bot1.generate_commands(g)])

            if not g.done:
                nextfeatures = extract_features(g, 0)
            else:
                nextfeatures = np.full(shape=(11, 64, 64), fill_value=-1)

            ret.add_sample(features, actions, rewards, nextfeatures, g.done)
    return ret


@pytest.fixture
def example_data_iter():
    with open('data/example_replay_memory.pickle', 'rb') as f:
        return iter(DataLoader(pickle.load(f), batch_size=BATCH_SIZE))


def test_extract_features_bounds(example_game):
    assert np.min(extract_features(example_game, 0)) >= 0
    assert np.min(extract_features(example_game, 1)) >= 0
    assert np.max(extract_features(example_game, 0)[1:]) <= 1
    assert np.max(extract_features(example_game, 1)[1:]) <= 1


def test_data_loader_yields_unique_entries(example_data_iter):
    assert not all((next(example_data_iter)[i] == next(example_data_iter)[i]).all() for i in range(5))


def test_replay_memory_dtype(example_data_iter):
    for _ in range(10):
        feat, actions, rewards, nextfeat, term = next(example_data_iter)
        assert feat.dtype == torch.float32
        assert actions.dtype == torch.int16
        assert rewards.dtype == torch.float32
        assert nextfeat.dtype == torch.float32
        assert term.dtype == torch.bool


def test_replay_memory_shape(example_data_iter):
    for _ in range(10):
        feat, actions, rewards, nextfeat, term = next(example_data_iter)
        assert feat.shape == (BATCH_SIZE, 11, 64, 64)
        assert actions.shape == (BATCH_SIZE, 64, 64)
        assert rewards.shape == (BATCH_SIZE, 64, 64)
        assert nextfeat.shape == (BATCH_SIZE, 11, 64, 64)
        assert term.shape == (BATCH_SIZE,)


def test_replay_memory_bounds(example_data_iter):
    for _ in range(100):
        feat, actions, rewards, nextfeat, term = next(example_data_iter)
        assert torch.min(feat) >= 0
        assert torch.max(feat[:, 1:]) <= 1
        assert torch.min(actions[actions != -1]) >= 0
        assert (feat[:, 1][actions == -1] == 0).all()
        assert (feat[:, 1][actions != -1] == 1).all()
        assert torch.max(actions) <= 4
        assert torch.min(rewards) >= 0
        assert torch.max(rewards) <= 10
        assert torch.min(nextfeat[~term]) >= 0
        assert torch.min(nextfeat[~term]) >= -1
        assert torch.max(nextfeat[:, 1:]) <= 1


def test_augment_dtype(example_data_iter):
    feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
    assert feat.dtype == torch.float32
    assert actions.dtype == torch.int16
    assert rewards.dtype == torch.float32
    assert nextfeat.dtype == torch.float32
    assert term.dtype == torch.bool


def test_augment_shape(example_data_iter):
    for _ in range(10):
        feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        assert feat.shape == (BATCH_SIZE, 11, 64, 64)
        assert actions.shape == (BATCH_SIZE, 64, 64)
        assert rewards.shape == (BATCH_SIZE, 64, 64)
        assert nextfeat.shape == (BATCH_SIZE, 11, 64, 64)
        assert term.shape == (BATCH_SIZE,)


def test_augment_bounds(example_data_iter):
    for i in range(100):
        feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        assert torch.min(feat) >= 0
        assert torch.max(feat[:, 1:]) <= 1
        assert torch.min(actions[actions != -1]) >= 0
        assert (feat[:, 1][actions == -1] == 0).all()
        assert (feat[:, 1][actions != -1] == 1).all()
        assert torch.max(actions) <= 4
        assert torch.min(rewards) >= 0
        assert torch.max(rewards) <= 10
        assert torch.min(nextfeat[~term]) >= 0
        assert torch.min(nextfeat[~term]) >= -1
        assert torch.max(nextfeat[:, 1:]) <= 1


def test_augment_correctly_modifies_action_directions(example_data_iter):
    # Assumes that ally ships will never crash into other ally ships,
    # except over a shipyard/dropoff
    for _ in range(10):
        feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        for b in range(BATCH_SIZE):
            if term[b]:
                continue
            ships_lost = 0
            for y in range(64):
                for x in range(64):
                    if actions[b, y, x] == -1:
                        assert feat[b, 1, y, x] == 0
                        continue
                    assert feat[b, 1, y, x] == 1
                    if actions[b, y, x] == 0:
                        if nextfeat[b, 1, y, x] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + dy) % 64, (x + dx) % 64] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, y, x] == 1
                                or nextfeat[b, 7, y, x] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 1:
                        if nextfeat[b, 1, (y + 1) % 64, x] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + 1 + dy) % 64, (x + dx) % 64] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, (y + 1) % 64, x] == 1
                                or nextfeat[b, 7, (y + 1) % 64, x] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 2:
                        if nextfeat[b, 1, y, (x + 1) % 64] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + dy) % 64, (x + 1 + dx) % 64] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, y, (x + 1) % 64] == 1
                                or nextfeat[b, 7, y, (x + 1) % 64] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 3:
                        if nextfeat[b, 1, (y - 1) % 64, x] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y - 1 + dy) % 64, (x + dx) % 64] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, (y - 1) % 64, x] == 1
                                or nextfeat[b, 7, (y - 1) % 64, x] == 1
                            )
                            ships_lost += 1
                    elif actions[b, y, x] == 4:
                        if nextfeat[b, 1, y, (x - 1) % 64] == 1:
                            continue
                        else:
                            assert (
                                any(feat[b, 3, (y + dy) % 64, (x - 1 + dx) % 64] == 1
                                    for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
                                or nextfeat[b, 5, y, (x - 1) % 64] == 1
                                or nextfeat[b, 7, y, (x - 1) % 64] == 1
                            )
                            ships_lost += 1
            cur_ships = torch.count_nonzero(feat[b, 1])
            next_ships = torch.count_nonzero(nextfeat[b, 1])
            assert (cur_ships - ships_lost) == next_ships or (cur_ships - ships_lost) == next_ships - 1


def test_training_step_improves_model(example_data_iter):
    my_device = torch.device('cuda:0')
    for i in range(100):
        model = QNet()
        model.to(my_device)
        frozen_model = copy.deepcopy(model)
        frozen_model.to(my_device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
        batch = augment(next(example_data_iter))

        loss = train_step(batch, model, frozen_model, loss_fn, optimizer)

        # Checks if any parameters are nan
        assert not any(next(model.parameters()).isnan().any() for _ in range(len(list(model.parameters()))))
        # Checks that model and frozen_model are now different
        assert not all(
            (next(model.parameters()) == next(frozen_model.parameters())).all()
            for _ in range(len(list(model.parameters())))
        )
        # Checks that loss decreased, at least for the current batch
        assert loss > train_step(batch, model, frozen_model, loss_fn, optimizer)
