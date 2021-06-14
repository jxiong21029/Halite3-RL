import copy

import torch
from dqn import LEARNING_RATE, QNet, train_step
from data_utils import augment
from test_data_utils import example_dataloader, example_data_iter


def test_reward_shaping(example_data_iter):
    count = 0
    for i in range(1):
        idxs, feat, actions, rewards, nextfeat, term = augment(next(example_data_iter))
        for b in range(32):
            for y in range(32):
                for x in range(32):
                    if feat[b, 1, y, x] == 0:
                        continue
                    if feat[b, 0, y, x] == 0 and actions[b, y, x] == 0:
                        assert rewards[b, y, x] == -1
                        count += 1
                    # TODO: testing other rewards more thorougly
                    else:
                        assert rewards[b, y, x] >= 0


def test_training_step_improves_model(example_data_iter):
    my_device = torch.device('cuda:0')
    small_loss_count = 0
    for i in range(100):
        model = QNet()
        model.to(my_device)
        frozen_model = copy.deepcopy(model)
        frozen_model.to(my_device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
        batch = augment(next(example_data_iter))

        td_error = train_step(batch, model, frozen_model, optimizer)

        # Checks if any parameters are nan
        assert not any(next(model.parameters()).isnan().any() for _ in range(len(list(model.parameters()))))
        # Checks that model and frozen_model are now different
        assert not all(
            (next(model.parameters()) == next(frozen_model.parameters())).all()
            for _ in range(len(list(model.parameters())))
        )
        if td_error.pow(2).mean() < 2e-5:
            small_loss_count += 1
        else:
            assert td_error.pow(2).mean() > train_step(batch, model, frozen_model, optimizer).pow(2).mean()
    assert small_loss_count < 10
