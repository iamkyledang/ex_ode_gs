import random

import numpy as np
import torch

from utils.checkpoint_utils import checkpoint_path, load_checkpoint, remove_checkpoint, save_checkpoint, set_rng_state


def test_checkpoint_round_trip_restores_state_and_rng(tmp_path):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    save_checkpoint(str(tmp_path), 1000, 12000, {'value': torch.arange(3), 'history': [1, 2]})
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.random(), np.random.rand(), torch.rand(1)  # move every generator on
    state = load_checkpoint(str(tmp_path), 12000)
    assert state['iteration'] == 1000 and state['history'] == [1, 2] and state['value'].tolist() == [0, 1, 2]
    set_rng_state(state['rng'])
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_checkpoint_of_another_schedule_is_ignored(tmp_path):
    save_checkpoint(str(tmp_path), 1000, 30000, {})
    assert load_checkpoint(str(tmp_path), 12000) is None


def test_missing_or_damaged_checkpoint_means_a_fresh_start(tmp_path):
    assert load_checkpoint(str(tmp_path), 12000) is None
    with open(checkpoint_path(str(tmp_path)), 'wb') as f:
        f.write(b'not a checkpoint')
    assert load_checkpoint(str(tmp_path), 12000) is None
    remove_checkpoint(str(tmp_path))
    assert not (tmp_path / 'checkpoint.pth').exists()
