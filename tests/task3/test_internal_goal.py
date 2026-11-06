"""The visual target must never shrink the success region or introduce solids."""
import copy
import re

import pytest

from rl2_reproduction.task3.internal_goal import SPEC, blocks, transform_internal_goal, validate_structure, write_internal_goal
from rl2_reproduction.task3.scene_builder import build_scene, debug_tasks
from rl2_reproduction.task3.target_preview import read_lumps


def test_original_geometry_lumps_and_floor_preserved(tmp_path):
    task = debug_tasks()["corridor"]
    old = read_lumps(build_scene(task, tmp_path / "source.wad"))
    new, report = transform_internal_goal(old, task)
    assert validate_structure(old, new, task) == report
    assert report["added"] == {"vertex": 4, "linedef": 4, "sidedef": 8, "sector": 0, "thing": 0}
    assert report["bounds_xy"] == [936, 40, 984, 88]
    assert SPEC["alpha"] == 1 and SPEC["inset_from_cell_boundary_units"] > 6.4
    text = dict(new)["TEXTMAP"].decode()
    lines = blocks(text, "linedef")[-4:]
    sides = blocks(text, "sidedef")[-8:]
    for line in lines:
        assert "blocking = false" in line and "midtex3d = false" in line
        front = int(re.search(r"sidefront = (\d+)", line)[1])
        back = int(re.search(r"sideback = (\d+)", line)[1])
        assert front >= 0 and back >= 0
    assert len({re.search(r"sector = (\d+)", side)[1] for side in sides}) == 1
    bad = [(n, data.replace(b"blockplayers = false", b"blockplayers = true") if n == "TEXTMAP" else data) for n, data in new]
    with pytest.raises(AssertionError):
        validate_structure(old, bad, task)


def test_no_overwrite_wrong_goal_or_duplicate_marker(tmp_path):
    task = debug_tasks()["corridor"]
    source = build_scene(task, tmp_path / "source.wad")
    target = tmp_path / "target.wad"
    write_internal_goal(source, target, task)
    with pytest.raises(FileExistsError):
        write_internal_goal(source, source, task)
    with pytest.raises(FileExistsError):
        write_internal_goal(source, target, task)
    with pytest.raises(ValueError, match="already present"):
        transform_internal_goal(read_lumps(target), task)
    wrong = copy.deepcopy(task)
    wrong["goal_cell"] = [0, 6]
    with pytest.raises(ValueError, match="goal does not match"):
        transform_internal_goal(read_lumps(source), wrong)


@pytest.mark.parametrize("yaw,start", [(0, [1, 0]), (90, [2, 1]), (180, [1, 2]), (270, [0, 1])])
def test_real_four_direction_goal_before_marker_and_walk_through(tmp_path, yaw, start):
    pytest.importorskip("vizdoom")
    from rl2_reproduction.task3.env import MazeEpisodeEnv, preprocess_rgb
    from rl2_reproduction.task3.engine import Engine
    import numpy as np
    task = dict(occupancy=[[0] * 3 for _ in range(3)], start_cell=start, goal_cell=[1, 1], start_yaw_deg=yaw)
    old = build_scene(task, tmp_path / "old.wad")
    new = tmp_path / "new.wad"
    write_internal_goal(old, new, task)
    wrapper, raw = [], []
    for path in (old, new):
        records = []
        with MazeEpisodeEnv() as env:
            obs, _ = env.reset_episode({**task, "scene_path": str(path)})
            assert np.array_equal(obs, preprocess_rgb(env.last_rgb))
            for _ in range(4):
                _, reward, done, info = env.step(2)
                records.append((reward, done, info))
                if done:
                    break
            assert done and info["success"] and not info["collided"]
            x, y, _ = info["pose"]
            assert not (168 <= x <= 216 and 168 <= y <= 216), "Must succeed before the central marker"
        wrapper.append(records)
        with Engine() as engine:
            engine.load_task(task, path)
            engine.reset()
            rows = []
            for _ in range(10):
                result = engine.action(2)
                rows.append((engine.pose(), result["collided"], result["poses"]))
            raw.append(rows)
    assert wrapper[0] == wrapper[1]
    assert raw[0] == raw[1]
