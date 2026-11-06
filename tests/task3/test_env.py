"""Wrapper reward/termination contracts, with explicit simulated engine fixtures."""
import numpy as np
import pytest
from rl2_reproduction.task3.env import MazeEpisodeEnv, preprocess_rgb


class StubEngine:
    def __init__(self):
        self.x, self.y = 64., 64.
        self.reach, self.collision = False, False
        self.rgb = np.full((120, 160, 3), 100, dtype=np.uint8)
    def load_task(self, *args): pass
    def reset(self):
        self.x = 64.
        return self.rgb
    def pose(self): return dict(x=self.x, y=self.y, yaw=0.)
    def action(self, index, stop_when):
        if self.reach: self.x = 192.
        return dict(image=self.rgb, collided=self.collision, ticks=4)
    def close(self): pass


def task(tmp_path):
    return dict(scene_path=str(tmp_path / "stub.wad"), occupancy=[[0, 0], [0, 0]], goal_cell=[1, 1])


def test_exact_area_rgb_and_no_privileged_channel():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[0:4, 0:4] = [0, 127, 255]
    result = preprocess_rgb(image)
    assert result.shape == (3, 30, 40) and result.dtype == np.uint8
    np.testing.assert_array_equal(result[:, 0, 0], [0, 127, 255])
    with pytest.raises(ValueError): preprocess_rgb(image[:, :, 0])


def test_timeout_exact_reward_no_implicit_reset(tmp_path):
    engine = StubEngine()
    env = MazeEpisodeEnv(engine=engine)
    env.reset_episode(task(tmp_path))
    rewards = []
    for t in range(250):
        obs, reward, done, info = env.step(0)
        rewards.append(reward)
        assert done == (t == 249)
        assert not info["collided"]
    assert sum(rewards) == pytest.approx(-10.)
    assert obs is None and info["timeout"]
    with pytest.raises(RuntimeError): env.step(0)


def test_last_step_success_priority_and_event_addition(tmp_path):
    engine = StubEngine()
    env = MazeEpisodeEnv(max_steps=1, engine=engine)
    env.reset_episode(task(tmp_path))
    engine.reach = engine.collision = True
    obs, reward, done, info = env.step(2)
    assert done and info["success"] and not info["timeout"] and obs is None
    assert reward == pytest.approx(.959)


def test_collision_counted_once_for_forward_only(tmp_path):
    engine = StubEngine()
    env = MazeEpisodeEnv(engine=engine)
    env.reset_episode(task(tmp_path))
    engine.collision = True
    assert env.step(2)[1] == pytest.approx(-.041)
    assert env.step(0)[1] == pytest.approx(-.04)


def test_reject_config_changes_that_would_be_ignored():
    import copy
    import yaml
    from rl2_reproduction.task3.config import validate_config
    from rl2_reproduction.task3.utils import PROJECT
    config = yaml.safe_load((PROJECT / "configs/task3/smoke.yaml").read_text())
    validate_config(config)
    for section, key, value in [("env", "max_steps_per_episode", 20), ("model", "pretrained_weights", "old.pt"),
                                ("observation", "privileged_inputs", True), ("run", "trials_per_update", 9)]:
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError): validate_config(changed)


def test_real_wrapper_reward_and_terminal_goal(tmp_path):
    pytest.importorskip("vizdoom")
    from rl2_reproduction.task3.scene_builder import build_scene, debug_tasks
    real_task = debug_tasks()["corridor"]
    real_task["goal_cell"] = [0, 2]
    real_task["scene_path"] = str(build_scene(real_task, tmp_path / "goal.wad"))
    with MazeEpisodeEnv() as env:
        obs, info = env.reset_episode(real_task)
        assert obs.shape == (3, 30, 40)
        rewards = []
        for _ in range(5):
            obs, reward, done, info = env.step(2)
            rewards.append(reward)
            if done: break
        assert done and info["success"] and not info["timeout"]
        assert obs is None and len(rewards) == 3
        assert rewards[-1] == pytest.approx(.96)
        assert sum(rewards) == pytest.approx(.88)
        assert 1 <= info["ticks"] < 4
        with pytest.raises(RuntimeError): env.step(2)
        env.reset_episode(real_task)
        assert env._pose() == pytest.approx((192., 64., 0.), abs=.01)


def test_real_two_timeout_returns(tmp_path):
    pytest.importorskip("vizdoom")
    from rl2_reproduction.task3.scene_builder import build_scene, debug_tasks
    real_task = debug_tasks()["corridor"]
    real_task["scene_path"] = str(build_scene(real_task, tmp_path / "timeout.wad"))
    returns = []
    with MazeEpisodeEnv() as env:
        for _ in range(2):
            env.reset_episode(real_task)
            total = 0
            for step in range(250):
                obs, reward, done, info = env.step(0)
                total += reward
                assert done == (step == 249)
            assert info["timeout"] and not info["success"]
            returns.append(total)
    assert returns == pytest.approx([-10., -10.])
    assert sum(returns) == pytest.approx(-20.)
