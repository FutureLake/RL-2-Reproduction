"""Scene structure tests and real ViZDoom integration tests."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from rl2_reproduction.task3.engine import Engine
from rl2_reproduction.task3.scene_builder import build_scene, debug_tasks


def read_lumps(path):
    data = path.read_bytes()
    magic, count, offset = struct.unpack_from('<4sii', data)
    assert magic == b'PWAD'
    result = {}
    for i in range(count):
        pos, size, name = struct.unpack_from('<ii8s', data, offset + i * 16)
        result[name.rstrip(b'\0').decode()] = data[pos:pos + size]
    return result


def test_wad_rebuild_and_goal_sector(tmp_path):
    task = debug_tasks()['l_corridor']
    a = build_scene(task, tmp_path / 'a.wad')
    b = build_scene(task, tmp_path / 'b.wad')
    assert a.read_bytes() == b.read_bytes()
    lumps = read_lumps(a)
    text = lumps['TEXTMAP'].decode()
    assert text.count('sector {') == 5
    assert text.count('texturefloor = "RLGOAL"') == 1
    assert 'twosided = true' in text
    assert 'blocking = true' in text
    assert b'Player.ViewHeight 64' in lumps['DECORATE']
    assert lumps['RLGOAL'].startswith(b'\x89PNG')
    assert lumps['BEHAVIOR'].startswith(b'ACS')


def test_invalid_spawn_is_rejected(tmp_path):
    task = debug_tasks()['l_corridor']
    task['start_cell'] = [0, 1]
    with pytest.raises(ValueError, match='open cells'):
        build_scene(task, tmp_path / 'bad.wad')


def test_contact_detects_sliding_and_ignores_open_corridor():
    engine = Engine()
    engine.task = debug_tasks()['corridor']
    assert not engine._would_contact(dict(x=200., y=64.), 6.4, 0.)
    assert engine._would_contact(dict(x=200., y=16.1), 6.4, -1.)
    assert engine._would_contact(dict(x=200., y=16.), 6.4, 0.)


@pytest.fixture
def corridor_engine(tmp_path):
    pytest.importorskip('vizdoom')
    task = debug_tasks()['corridor']
    path = build_scene(task, tmp_path / 'corridor.wad')
    with Engine() as engine:
        engine.load_task(task, path)
        engine.reset()
        yield engine


def test_real_rgb_camera_and_reset(corridor_engine):
    engine = corridor_engine
    original = engine.image()
    pose = engine.pose()
    assert original.shape == (120, 160, 3)
    assert original.dtype == np.uint8
    assert np.ptp(original) > 50
    assert pose['x'] == pytest.approx(192., abs=1e-3)
    assert pose['y'] == pytest.approx(64., abs=1e-3)
    assert pose['viewheight'] == pytest.approx(64., abs=.01)
    assert pose['camera_z'] == pytest.approx(64., abs=.01)
    assert pose['fov'] == pytest.approx(90., abs=.01)
    engine.action(2)
    engine.action(0)
    reset_image = engine.reset()
    assert np.array_equal(original, reset_image)
    reset = engine.pose()
    assert reset['yaw'] == pytest.approx(0., abs=.01)
    assert reset['vx'] == pytest.approx(0., abs=1e-5)
    assert reset['vy'] == pytest.approx(0., abs=1e-5)


def test_real_action_calibration_and_collision(corridor_engine):
    engine = corridor_engine
    for action, sign in [(0, 1), (1, -1)]:
        engine.reset()
        start = engine.pose()
        result = engine.action(action)
        end = engine.pose()
        angle = (end['yaw'] - start['yaw'] + 180) % 360 - 180
        assert angle == pytest.approx(sign * 5., abs=.1)
        assert result['distance'] <= 1.28
        assert not result['collided']
    engine.reset()
    for _ in range(5):
        result = engine.action(2)
        assert result['distance'] == pytest.approx(25.6, rel=.05)
        assert not result['collided']
    start = engine.pose()
    engine.action(0)
    assert np.hypot(engine.pose()['x'] - start['x'], engine.pose()['y'] - start['y']) <= 1.28
    engine.reset()
    for _ in range(18):
        engine.action(0)
    results = [engine.action(2) for _ in range(6)]
    assert any(r['collided'] for r in results)
    assert engine.pose()['y'] <= 112.01


def test_goal_callback_interrupts_engine_ticks(corridor_engine):
    engine = corridor_engine
    calls = []
    def stop(pose):
        calls.append(pose)
        return True
    result = engine.action(2, stop_when=stop)
    assert result['ticks'] == 1
    assert len(calls) == len(result['poses']) == 1


def test_real_wall_sliding_still_reports_contact(corridor_engine):
    engine = corridor_engine
    for _ in range(9):
        engine.action(0)
    for _ in range(3):
        engine.action(2)
    start = engine.pose()
    slide = engine.action(2)
    end = engine.pose()
    assert slide['collided']
    assert end['x'] - start['x'] > 10.
    assert abs(end['y'] - start['y']) < 1.
    assert end['y'] <= 112.01
    turn = engine.action(1)
    assert not turn['collided']
    assert turn['distance'] <= 1.28


@pytest.mark.parametrize('name,actions', [
    ('l_corridor', [2] * 10 + [1] * 18 + [2] * 10),
    ('t_junction', [2] * 10 + [0] * 18 + [2] * 5),
])
def test_real_l_and_t_goal_region(tmp_path, name, actions):
    pytest.importorskip('vizdoom')
    task = debug_tasks()[name]
    path = build_scene(task, tmp_path / f'{name}.wad')
    with Engine() as engine:
        engine.load_task(task, path)
        engine.reset()
        maximum_red_fraction = 0.
        for action in actions:
            result = engine.action(action)
            image = result['image'].astype(np.int16)
            red = (image[:, :, 0] > image[:, :, 1] + 50) & (image[:, :, 0] > image[:, :, 2] + 50)
            maximum_red_fraction = max(maximum_red_fraction, red.mean())
            for pose in result['poses']:
                col = int(pose['x'] // 128)
                row = len(task['occupancy']) - 1 - int(pose['y'] // 128)
                assert task['occupancy'][row][col] == 0
        pose = engine.pose()
        reached = [len(task['occupancy']) - 1 - int(pose['y'] // 128), int(pose['x'] // 128)]
        assert reached == task['goal_cell']
        # At the center of a dead end the tall camera can face only its close
        # wall. The goal floor must be visibly red during the approach.
        assert maximum_red_fraction > .05
