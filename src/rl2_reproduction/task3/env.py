"""Two-episode-compatible wrapper; only RGB and causal feedback reach the model."""
from pathlib import Path
import math

import numpy as np


def preprocess_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape != (120, 160, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Expected RGB uint8 HWC120x160, got {rgb.shape}/{rgb.dtype}")
    # Exact 4x4 box-area filter. Fixed round-half-up, no external image library.
    sums = rgb.astype(np.uint16).reshape(30, 4, 40, 4, 3).sum(axis=(1, 3))
    return np.ascontiguousarray(((sums + 8) // 16).astype(np.uint8).transpose(2, 0, 1))


class MazeEpisodeEnv:
    def __init__(self, max_steps=250, engine=None, calibration=None):
        if engine is None:
            from .engine import Engine
            engine = Engine(calibration)
        self.engine = engine
        self.max_steps = max_steps
        self.max_steps_per_episode = max_steps
        self.task = None
        self.steps = 0
        self.done = True

    def _pose(self):
        raw = self.engine.pose()
        if isinstance(raw, dict):
            return tuple(float(raw[k]) for k in ("x", "y", "yaw"))
        return tuple(float(x) for x in raw[:3])

    def _goal(self, pose):
        if isinstance(pose, dict):
            pose = (pose["x"], pose["y"], pose["yaw"])
        n = len(self.task["occupancy"])
        col = math.floor(pose[0] / 128)
        row = n - 1 - math.floor(pose[1] / 128)
        return (row, col) == tuple(self.task["goal_cell"])

    def reset_episode(self, task_spec):
        scene = str(Path(task_spec["scene_path"]).resolve())
        key = (scene, task_spec.get("scene_hash"))
        if self.task is None or key != self._key:
            self.engine.load_task(task_spec, scene)
            self._key = key
        self.task = task_spec
        rgb = self.engine.reset()
        self.steps = 0
        self.done = False
        pose = self._pose()
        if self._goal(pose):
            raise RuntimeError("Spawn is inside goal")
        self.last_rgb = rgb
        return preprocess_rgb(rgb), {"pose": pose, "success": False, "episode_steps": 0}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended; collector must explicitly reset")
        if action not in (0, 1, 2):
            raise ValueError("Only left/right/forward are allowed")
        before = self._pose()
        result = self.engine.action(int(action), stop_when=self._goal)
        self.steps += 1
        pose = self._pose()
        success = self._goal(pose)
        collided = bool(result["collided"]) if action == 2 else False
        reward = -0.04 + float(success) - 0.001 * float(collided)
        self.done = success or self.steps >= self.max_steps
        self.last_rgb = result["image"]
        info = {"pose": pose, "success": success, "collided": collided,
                "episode_steps": self.steps, "timeout": self.done and not success,
                "distance": float(result.get("distance", math.dist(before[:2], pose[:2]))),
                "ticks": int(result.get("ticks", 4)), "tick_poses": result.get("poses", [])}
        return (None if self.done else preprocess_rgb(self.last_rgb)), reward, self.done, info

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
