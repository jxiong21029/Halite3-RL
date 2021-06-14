import random

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

from entity import Shipyard


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
    # Feature 10 - turns remaining - normalized
    cell_features[10] = (game.max_turns - game.turn) / 500
    return cell_features


class ReplayMemory(IterableDataset):
    def __init__(self, size, device):
        self.size = size
        self.features = torch.zeros(size=(size, 11, 32, 32), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size=(size, 32, 32), dtype=torch.int16, device=device)
        self.rewards = torch.zeros(size=(size, 32, 32), dtype=torch.float32, device=device)
        self.nextfeatures = torch.zeros(size=(size, 11, 32, 32), dtype=torch.float32, device=device)
        self.terminal = torch.zeros(size=(size,), dtype=torch.bool, device=device)
        self.num_entries = 0
        self._idx = 0

    def add_sample(self, new_feat, new_actions, new_rewards, new_nextfeat, terminal=False):
        self.features[self._idx] = torch.as_tensor(new_feat, dtype=torch.float32)
        self.actions[self._idx] = torch.as_tensor(new_actions, dtype=torch.int16)
        self.rewards[self._idx] = torch.as_tensor(new_rewards, dtype=torch.float32)
        self.nextfeatures[self._idx] = torch.as_tensor(new_nextfeat, dtype=torch.float32)
        self.terminal[self._idx] = terminal

        self._idx = (self._idx + 1) % self.size
        if self.num_entries < self.size:
            self.num_entries += 1

    def __iter__(self):
        return self

    def __next__(self):
        idx = random.randint(0, self.num_entries - 1)
        return idx, self.features[idx], self.actions[idx], self.rewards[idx], self.nextfeatures[idx], self.terminal[idx]

    def __getitem__(self, index) -> T_co:
        return NotImplementedError


class PrioritizedReplayMemory(IterableDataset):
    def __init__(self, size, device):
        self.size = size
        self.features = torch.zeros(size=(size, 11, 32, 32), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size=(size, 32, 32), dtype=torch.int16, device=device)
        self.rewards = torch.zeros(size=(size, 32, 32), dtype=torch.float32, device=device)
        self.nextfeatures = torch.zeros(size=(size, 11, 32, 32), dtype=torch.float32, device=device)
        self.terminal = torch.zeros(size=(size,), dtype=torch.bool, device=device)
        self.sumtree = np.zeros(2 * self.size - 1)
        self.maxprio = 1
        self.num_entries = 0
        self._idx = 0

    def _propagate_prio(self, idxs, change):
        # TODO: investigate why average prio is turning negative, and then add a test
        idxs, change = np.asarray(idxs), np.asarray(change)
        # new_change = change

        parents = np.asarray((idxs - 1) // 2)
        np.add.at(self.sumtree, parents, change)
        if np.count_nonzero(parents) > 0:
            self._propagate_prio(parents[parents > 0], change[parents > 0])

    def update_prio(self, idxs, new_prios):
        if isinstance(idxs, torch.Tensor):
            idxs, new_prios = idxs.detach().cpu().numpy(), new_prios.detach().cpu().numpy()
        else:
            idxs, new_prios = np.asarray(idxs), np.asarray(new_prios)
        self.maxprio = max(self.maxprio, np.amax(new_prios))
        change = new_prios - self.sumtree[idxs + self.size - 1]
        self.sumtree[idxs + self.size - 1] = new_prios

        idxs, invind = np.unique(idxs, return_inverse=True)
        new_change = np.zeros(idxs.shape)
        new_change[invind] = change
        self._propagate_prio(idxs + self.size - 1, new_change)

    def recalculate_max(self):
        self.maxprio = self.sumtree[self.size - 1:].max()

    def add_sample(self, new_feat, new_actions, new_rewards, new_nextfeat, terminal=False):
        self.features[self._idx] = torch.as_tensor(new_feat, dtype=torch.float32)
        self.actions[self._idx] = torch.as_tensor(new_actions, dtype=torch.int16)
        self.rewards[self._idx] = torch.as_tensor(new_rewards, dtype=torch.float32)
        self.nextfeatures[self._idx] = torch.as_tensor(new_nextfeat, dtype=torch.float32)
        self.terminal[self._idx] = terminal

        self.update_prio(self._idx, self.maxprio)

        self._idx = (self._idx + 1) % self.size
        if self.num_entries < self.size:
            self.num_entries += 1

    def __iter__(self):
        return self

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.sumtree):
            return idx
        if s <= self.sumtree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sumtree[left])

    def __next__(self):
        idx = self._retrieve(0, random.random() * self.sumtree[0])
        data_idx = idx - self.size + 1
        return (data_idx, self.features[data_idx], self.actions[data_idx], self.rewards[data_idx],
                self.nextfeatures[data_idx], self.terminal[data_idx])

    def __getitem__(self, index) -> T_co:
        return NotImplementedError


def augment(batch, flip=None, num_rot=None):
    if flip is None:
        flip = (random.random() < 0.5)
    if num_rot is None:
        num_rot = random.randint(0, 3)
    idxs, feat, actions, rewards, nextfeat, term = batch
    if flip:
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
        return idxs, feat, new_actions, rewards, nextfeat, term
    return idxs, feat, actions, rewards, nextfeat, term
