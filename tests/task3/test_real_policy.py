"""Full-size visual recurrent replay on actual ViZDoom RGB, without optimization."""
import numpy as np
import pytest
import torch

from rl2_reproduction.task3.env import MazeEpisodeEnv
from rl2_reproduction.task3.scene_builder import debug_tasks, build_scene
from rl2_reproduction.task3.visual_policy import VisualPolicy
from rl2_reproduction.task3.trial_collector import TrialCollector
from rl2_reproduction.task3.sequence_batch import collate_trials, compute_gae
from rl2_reproduction.task3.ppo_learner import replay_check, compute_losses


def test_real_rgb_full_model_trial_replay_and_backward(tmp_path):
    pytest.importorskip("vizdoom")
    torch.manual_seed(714)
    policy = VisualPolicy()
    task = debug_tasks()["l_corridor"]
    task["scene_path"] = str(build_scene(task, tmp_path / "visual.wad"))
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    with MazeEpisodeEnv() as env:
        trial = TrialCollector(env, policy, torch.Generator().manual_seed(715)).collect(task)
    assert len(trial.episode_summaries) == 2
    assert all(1 <= e["length"] <= 250 for e in trial.episode_summaries)
    assert trial.arrays["images_uint8"].dtype == np.uint8
    batch = compute_gae(collate_trials([trial]))
    assert int(batch["mask"].sum()) == trial.length
    replay_check(policy, batch)
    loss = compute_losses(policy, batch)["loss"]
    loss.backward()
    for module in (policy.cnn, policy.gru, policy.policy_head, policy.value_head):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    # This integration test performs no optimizer step and is not part of S4 training.
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
