"""Tests exercise the pinned reference checkout without installing its legacy deps."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "pytorch_rl2"
sys.path.insert(0, str(UPSTREAM))

from train import create_env, create_net  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def cpu_determinism():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def make_system():
    def build(environment="tabular_mdp", seed=731):
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = create_env(environment, num_states=4, num_actions=3, max_episode_len=2)
        nets = [
            create_net(
                net_type=kind,
                environment=environment,
                architecture="gru",
                num_states=4,
                num_actions=3,
                num_features=16,
                context_size=6,
            )
            for kind in ("policy", "value")
        ]
        return env, *nets

    return build
