"""Single-process ViZDoom transport, physical actions, and calibration.

This module never computes an RL reward. All privileged poses remain diagnostic
data; the policy receives only RGB frames from the higher-level wrapper.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .scene_builder import CELL, build_scene, debug_tasks


class Engine:
    def __init__(self, calibration: dict | str | Path | None = None, *, visible: bool = False):
        if isinstance(calibration, (str, Path)):
            calibration = json.loads(Path(calibration).read_text(encoding='utf-8'))
        self.calibration = calibration or dict(mode='velocity', turn_delta=1.25, velocity_per_tick=6.4, ticks_per_action=4, spawn_settle_ticks=10)
        if 'parameters' in self.calibration:
            self.calibration = self.calibration['parameters']
        self.visible = visible
        self.game = None
        self.task = None
        self.scene_path = None

    def load_task(self, task: dict, scene_path: str | Path) -> None:
        import vizdoom as vzd
        path = Path(scene_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.game is not None and path == self.scene_path:
            self.task = task
            return
        self.close()
        game = vzd.DoomGame()
        game.set_doom_scenario_path(str(path))
        game.set_doom_map('MAP01')
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_window_visible(self.visible)
        game.set_sound_enabled(False)
        game.set_console_enabled(False)
        game.set_render_hud(False)
        game.set_render_minimal_hud(False)
        game.set_render_weapon(False)
        game.set_render_crosshair(False)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)
        game.set_render_messages(False)
        game.set_render_corpses(False)
        game.set_render_screen_flashes(False)
        # The Doom player has a short spawn reaction delay. Waiting with zero
        # controls before exposing an episode prevents the first RL actions
        # being silently ignored; these reset ticks are not policy steps.
        game.set_episode_start_time(10)
        game.set_episode_timeout(1000000)
        game.set_living_reward(0.)
        game.set_death_penalty(0.)
        game.set_seed(0)
        game.add_game_args('+fov 90 +movebob 0 +stillbob 0 +cl_run 0 +freelook 0 +sv_cheats 1')
        game.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.MOVE_FORWARD_BACKWARD_DELTA])
        game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 10.)
        game.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, 100.)
        names = ['POSITION_X', 'POSITION_Y', 'POSITION_Z', 'ANGLE', 'PITCH', 'ROLL', 'VELOCITY_X', 'VELOCITY_Y', 'VELOCITY_Z', 'VIEW_HEIGHT', 'CAMERA_POSITION_Z', 'CAMERA_FOV']
        game.set_available_game_variables([getattr(vzd.GameVariable, name) for name in names])
        try:
            game.init()
        except BaseException:
            game.close()
            raise
        self.game = game
        self.task = task
        self.scene_path = path
        self._variables = {name: getattr(vzd.GameVariable, name) for name in names}

    def _check(self):
        if self.game is None:
            raise RuntimeError('No task loaded')
        if self.game.is_episode_finished() or self.game.is_player_dead():
            raise RuntimeError('Unexpected ViZDoom terminal state/death; wrapper owns episode limits')

    def image(self) -> np.ndarray:
        self._check()
        image = self.game.get_state().screen_buffer
        if image.shape != (120, 160, 3) or image.dtype != np.uint8:
            raise RuntimeError(f'Unexpected ViZDoom RGB24 frame: {image.shape}, {image.dtype}')
        return image.copy()

    def pose(self) -> dict[str, float]:
        self._check()
        values = {name: float(self.game.get_game_variable(var)) for name, var in self._variables.items()}
        return dict(x=values['POSITION_X'], y=values['POSITION_Y'], z=values['POSITION_Z'], yaw=values['ANGLE'], pitch=values['PITCH'], roll=values['ROLL'], vx=values['VELOCITY_X'], vy=values['VELOCITY_Y'], vz=values['VELOCITY_Z'], viewheight=values['VIEW_HEIGHT'], camera_z=values['CAMERA_POSITION_Z'], fov=values['CAMERA_FOV'])

    def reset(self) -> np.ndarray:
        if self.game is None:
            raise RuntimeError('No task loaded')
        self.game.set_action([0., 0.])
        self.game.new_episode()
        self.game.set_action([0., 0.])
        self._check()
        pose = self.pose()
        if max(abs(pose[key]) for key in ('vx', 'vy', 'vz')) > 1e-4:
            raise RuntimeError(f'Reset did not clear velocity: {pose}')
        return self.image()

    def _set_velocity(self, vx: float, vy: float) -> None:
        # ACS_ExecuteWithResult executes the hook immediately. Plain `puke`
        # schedules it one thinker tick later, causing a 6.4-unit drift after
        # a forward action; the calibration regression explicitly checks this.
        self.game.send_game_command(f'special 84 1 {round(vx * 65536)} {round(vy * 65536)} 0')

    @staticmethod
    def _segment_hits_box(x, y, dx, dy, box):
        lower, upper = 0., 1.
        for origin, delta, lo, hi in ((x, dx, box[0], box[2]), (y, dy, box[1], box[3])):
            if abs(delta) < 1e-12:
                if origin < lo or origin > hi:
                    return False
            else:
                a, b = (lo - origin) / delta, (hi - origin) / delta
                lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
                if lower > upper:
                    return False
        # A contact at t=0 while moving away is not a new blocked movement.
        return upper > 1e-8 and lower <= 1.

    def _would_contact(self, pose, dx, dy) -> bool:
        # Doom's horizontal body test uses an axis-aligned radius bounding box.
        grid = self.task['occupancy']
        rows, cols = len(grid), len(grid[0])
        radius = 16.
        endpoint_x, endpoint_y = pose['x'] + dx, pose['y'] + dy
        if (endpoint_x <= radius and dx <= 0.) or (endpoint_x >= cols * CELL - radius and dx >= 0.) or (endpoint_y <= radius and dy <= 0.) or (endpoint_y >= rows * CELL - radius and dy >= 0.):
            return True
        for row in range(rows):
            for col in range(cols):
                if grid[row][col]:
                    box = (col * CELL - radius, (rows - row - 1) * CELL - radius, (col + 1) * CELL + radius, (rows - row) * CELL + radius)
                    if self._segment_hits_box(pose['x'], pose['y'], dx, dy, box):
                        return True
        return False

    def action(self, index: int, stop_when: Callable[[dict], bool] | None = None) -> dict:
        if index not in (0, 1, 2):
            raise ValueError('Only three discrete actions (0,1,2) are defined')
        self._check()
        before = self.pose()
        poses = []
        distance = 0.
        collided = False
        engine_rewards = []
        mode = self.calibration.get('mode', 'delta')
        for _ in range(4):
            angle = math.radians(before['yaw'])
            speed = float(self.calibration.get('velocity_per_tick', 6.4))
            dx, dy = speed * math.cos(angle), speed * math.sin(angle)
            if mode == 'velocity':
                self._set_velocity(dx if index == 2 else 0., dy if index == 2 else 0.)
            elif self.calibration.get('clear_velocity_each_tick', False):
                self._set_velocity(0., 0.)
            turn = float(self.calibration.get('turn_delta', 1.25))
            # ViZDoom's positive yaw delta is clockwise; measured by calibrate.
            rotation = -turn if index == 0 else turn if index == 1 else 0.
            movement = float(self.calibration.get('forward_delta', 30.)) if index == 2 and mode != 'velocity' else 0.
            engine_rewards.append(float(self.game.make_action([rotation, movement], 1)))
            after = self.pose()
            distance += math.hypot(after['x'] - before['x'], after['y'] - before['y'])
            if index == 2:
                collided |= self._would_contact(before, dx, dy)
            poses.append(after)
            before = after
            if stop_when is not None and stop_when(after):
                break
        self.game.set_action([0., 0.])
        if mode == 'velocity':
            self._set_velocity(0., 0.)
        if any(abs(r) > 1e-8 for r in engine_rewards):
            raise RuntimeError(f'Unexpected nonzero native engine reward: {engine_rewards}')
        return dict(image=self.image(), poses=poses, collided=bool(collided), distance=distance, ticks=len(poses))

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def calibration_probe(output_dir: str | Path, parameters: dict | None = None) -> dict:
    """Measure actual rendering/actions; writes raw evidence, never trains."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = debug_tasks()['corridor']
    path = build_scene(task, output_dir / 'corridor.wad')
    rows = []
    with Engine(parameters) as engine:
        engine.load_task(task, path)
        image = engine.reset()
        np.save(output_dir / 'first_person.npy', image)
        for name, actions in [('left', [0]), ('right', [1]), ('forward', [2]), ('continuous_forward', [2] * 10), ('forward_then_turn', [2, 0, 0, 0]), ('turn_then_forward', [0, 2, 2]), ('wall', [0] * 18 + [2] * 5), ('sliding_wall', [0] * 9 + [2] * 5)]:
            engine.reset()
            for i, action in enumerate(actions):
                start = engine.pose()
                outcome = engine.action(action)
                end = engine.pose()
                rows.append(dict(scenario=name, action_index=i, action=action, before=start, after=end, dx=end['x'] - start['x'], dy=end['y'] - start['y'], displacement=math.hypot(end['x'] - start['x'], end['y'] - start['y']), angle_delta=(end['yaw'] - start['yaw'] + 180) % 360 - 180, collided=outcome['collided'], ticks=outcome['ticks']))
        params = dict(engine.calibration)
    result = dict(parameters=params, measurements=rows)
    (output_dir / 'calibration_raw.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def calibrate(output_dir: str | Path) -> dict:
    """Freeze physical parameters only after measured tolerance checks pass."""
    import vizdoom
    from PIL import Image
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected = calibration_probe(output_dir / 'delta_probe', dict(mode='delta', turn_delta=1.25, forward_delta=30., velocity_per_tick=6.4))
    parameters = dict(mode='velocity', turn_delta=1.25, velocity_per_tick=6.4, ticks_per_action=4, spawn_settle_ticks=10)
    result = calibration_probe(output_dir / 'velocity_probe', parameters)
    rows = result['measurements']
    # Four cardinal starting headings also verify array/world orientation and
    # render the actual camera images, independently of the corridor tests.
    for yaw in (0, 90, 180, 270):
        task = dict(task_id=f'debug_heading_{yaw}', occupancy=[[0] * 7 for _ in range(7)], start_cell=[3, 3], goal_cell=[1, 1], start_yaw_deg=yaw)
        path = build_scene(task, output_dir / f'heading_{yaw}.wad')
        with Engine(parameters) as engine:
            engine.load_task(task, path)
            Image.fromarray(engine.reset()).save(output_dir / f'heading_{yaw}.png')
            for i in range(5):
                start = engine.pose()
                outcome = engine.action(2)
                end = engine.pose()
                rows.append(dict(scenario=f'heading_{yaw}', action_index=i, action=2, before=start, after=end, dx=end['x']-start['x'], dy=end['y']-start['y'], displacement=math.hypot(end['x']-start['x'], end['y']-start['y']), angle_delta=(end['yaw']-start['yaw']+180)%360-180, collided=outcome['collided'], ticks=outcome['ticks']))
    forward = [r for r in rows if r['action'] == 2 and r['scenario'] not in ('wall', 'sliding_wall')]
    turns = [r for r in rows if r['action'] != 2]
    expected_angle = lambda r: 5. if r['action'] == 0 else -5.
    checks = dict(
        forward_max_relative_error=max(abs(r['displacement'] / 25.6 - 1.) for r in forward),
        turn_max_absolute_error_deg=max(abs(r['angle_delta'] - expected_angle(r)) for r in turns),
        turn_max_drift_units=max(r['displacement'] for r in turns),
        all_four_heading_directions=all((abs(r['dx'] - 25.6 * math.cos(math.radians(r['before']['yaw']))) < 1.28 and abs(r['dy'] - 25.6 * math.sin(math.radians(r['before']['yaw']))) < 1.28) for r in forward),
        wall_collision_observed=any(r['collided'] for r in rows if r['scenario'] == 'wall'),
        wall_not_crossed=all(r['after']['y'] <= 112.01 for r in rows if r['scenario'] == 'wall'),
        sliding_contact_detected=any(r['collided'] and abs(r['dx']) > 10. and abs(r['dy']) < 1. for r in rows if r['scenario'] == 'sliding_wall'),
        camera_height_64=all(abs(r['before']['camera_z'] - 64.) < .01 for r in rows),
        fov_90=all(abs(r['before']['fov'] - 90.) < .01 for r in rows),
        pitch_roll_zero=all(abs(r['before']['pitch']) < .01 and abs(r['before']['roll']) < .01 for r in rows),
    )
    passed = checks['forward_max_relative_error'] <= .05 and checks['turn_max_absolute_error_deg'] <= .1 and checks['turn_max_drift_units'] <= 1.28 and all(v for k, v in checks.items() if k not in ('forward_max_relative_error', 'turn_max_absolute_error_deg', 'turn_max_drift_units'))
    result.update(protocol='rl2-task3-from-scratch-v2', passed=bool(passed), engine_version=vizdoom.__version__, checks=checks, rejected_delta_probe=str(output_dir / 'delta_probe' / 'calibration_raw.json'), method='ACS_ExecuteWithResult + SetActorVelocity before each physical tick; engine collision retained; no movement teleportation', source_note='Four tick velocity control is an explicitly permitted v2 local physical calibration design, not an original RL2 engine detail.')
    Image.fromarray(np.load(output_dir / 'velocity_probe' / 'first_person.npy')).save(output_dir / 'first_person_corridor.png')
    (output_dir / 'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not passed:
        raise RuntimeError(f'Physical calibration failed: {checks}; raw measurements: {output_dir}')
    return result
