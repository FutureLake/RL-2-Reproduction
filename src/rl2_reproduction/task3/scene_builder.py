"""Rebuildable UDMF scenes. Geometry and textures are generated locally.

No Doom editor, node builder, or training code is required. ViZDoom builds BSP
nodes when loading TEXTMAP. ACC is only used for the tiny velocity-control hook.
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path

CELL = 128

ACS_SOURCE = '''#include "zcommon.acs"
// Values passed by the wrapper are fixed-point Doom units per engine tick.
script 1 (int vx, int vy, int vz)
{
    SetActorVelocity(0, vx, vy, vz, FALSE, FALSE);
}
'''


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _png(color: tuple[int, int, int], checker: bool = False) -> bytes:
    """A deterministic, RGB PNG using only the Python standard library."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    rows = []
    for y in range(64):
        row = bytearray([0])
        for x in range(64):
            scale = .82 if checker and ((x // 16 + y // 16) % 2) else 1.
            row.extend(round(c * scale) for c in color)
        rows.append(bytes(row))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b'')


def _write_wad(path: Path, lumps: list[tuple[str, bytes]]) -> None:
    body = bytearray()
    entries = []
    for name, data in lumps:
        entries.append((12 + len(body), len(data), name.encode('ascii').ljust(8, b'\0')))
        body.extend(data)
    directory = b''.join(struct.pack('<ii8s', *entry) for entry in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<4sii', b'PWAD', len(lumps), 12 + len(body)) + body + directory)


def compile_behavior(cache_dir: Path | None = None) -> bytes:
    """Compile inspectable ACS source with the pinned official ACC release."""
    cache_dir = cache_dir or project_root() / 'data' / 'task3' / 'tools'
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = cache_dir / 'velocity.acs'
    output = cache_dir / 'velocity.o'
    compiler_dir = project_root() / 'third_party' / 'acc' / '1.60'
    compiler = compiler_dir / 'acc.exe'
    if not compiler.is_file():
        raise FileNotFoundError(f'Pinned ACC 1.60 compiler is missing: {compiler}')
    if not source.exists() or source.read_text(encoding='utf-8') != ACS_SOURCE or not output.exists():
        source.write_text(ACS_SOURCE, encoding='utf-8')
        result = subprocess.run([str(compiler), '-i', str(compiler_dir), str(source), str(output)], capture_output=True, text=True)
        (cache_dir / 'acc_compile.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode != 0 or not output.exists():
            raise RuntimeError(f'ACC compilation failed ({result.returncode}): {result.stdout}\n{result.stderr}')
    return output.read_bytes()


def build_scene(task: dict, path: str | Path, *, behavior: bytes | None = None) -> Path:
    """Build one task-specific scene; start/goal coordinates use row,col order."""
    path = Path(path)
    grid = task['occupancy']
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError('Occupancy must be rectangular')
    start = tuple(task['start_cell'])
    goal = tuple(task['goal_cell'])
    if grid[start[0]][start[1]] or grid[goal[0]][goal[1]]:
        raise ValueError('Start and goal must be open cells')
    vertices: list[tuple[int, int]] = []
    vertex_ids: dict[tuple[int, int], int] = {}
    sides = []
    sectors = []
    edges: dict[tuple[int, int], dict] = {}
    def vid(xy):
        if xy not in vertex_ids:
            vertex_ids[xy] = len(vertices)
            vertices.append(xy)
        return vertex_ids[xy]
    for row in range(rows):
        for col in range(cols):
            if grid[row][col]:
                continue
            sid = len(sectors)
            sectors.append((row, col))
            x0, y0 = col * CELL, (rows - row - 1) * CELL
            # Clockwise: the sector is on the linedef's right/front side.
            points = [(x0, y0), (x0, y0 + CELL), (x0 + CELL, y0 + CELL), (x0 + CELL, y0)]
            for a, b in zip(points, points[1:] + points[:1]):
                v1, v2 = vid(a), vid(b)
                key = tuple(sorted((v1, v2)))
                side_id = len(sides)
                sides.append(sid)
                if key in edges:
                    edge = edges[key]
                    if edge['back'] != -1:
                        raise ValueError('Non-manifold grid edge')
                    edge['back'] = side_id
                else:
                    edges[key] = dict(v1=v1, v2=v2, front=side_id, back=-1)
    lines = ['namespace = "ZDoom";']
    lines += [f'vertex {{ x = {x}.0; y = {y}.0; }}' for x, y in vertices]
    for e in edges.values():
        dual = e['back'] >= 0
        lines.append(f'linedef {{ v1 = {e["v1"]}; v2 = {e["v2"]}; sidefront = {e["front"]}; sideback = {e["back"]}; blocking = {str(not dual).lower()}; twosided = {str(dual).lower()}; }}')
    portal_sides = {e[k] for e in edges.values() if e['back'] >= 0 for k in ('front', 'back')}
    for i, sid in enumerate(sides):
        texture = '-' if i in portal_sides else 'RLWALL'
        lines.append(f'sidedef {{ sector = {sid}; texturemiddle = "{texture}"; }}')
    for rc in sectors:
        floor = 'RLGOAL' if rc == goal else 'RLFLOOR'
        lines.append(f'sector {{ heightfloor = 0; heightceiling = 128; texturefloor = "{floor}"; textureceiling = "RLCEIL"; lightlevel = 240; }}')
    sx, sy = (start[1] + .5) * CELL, (rows - start[0] - .5) * CELL
    yaw = int(task['start_yaw_deg']) % 360
    lines.append(f'thing {{ x = {sx}; y = {sy}; angle = {yaw}; type = 1; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; single=true; }}')
    decorate = b'actor RL2Player : DoomPlayer\n{\n Radius 16\n Height 56\n Player.ViewHeight 64\n Player.JumpZ 0\n}\n'
    mapinfo = b'gameinfo { playerclasses = "RL2Player" }\nmap MAP01 "RL2 Maze" { levelnum = 1 nointermission }\n'
    lumps = [('DECORATE', decorate), ('MAPINFO', mapinfo), ('TX_START', b''), ('RLWALL', _png((132, 140, 148), True)), ('RLFLOOR', _png((94, 96, 96), True)), ('RLCEIL', _png((118, 118, 118))), ('RLGOAL', _png((240, 24, 24))), ('TX_END', b''), ('MAP01', b''), ('TEXTMAP', '\n'.join(lines).encode('ascii')), ('BEHAVIOR', behavior if behavior is not None else compile_behavior()), ('SCRIPTS', ACS_SOURCE.encode('ascii')), ('ENDMAP', b'')]
    _write_wad(path, lumps)
    return path


def debug_tasks() -> dict[str, dict]:
    return {
        'corridor': dict(task_id='debug_corridor', occupancy=[[0] * 9], start_cell=[0, 1], goal_cell=[0, 7], start_yaw_deg=0),
        'l_corridor': dict(task_id='debug_l', occupancy=[[0, 1, 1], [0, 1, 1], [0, 0, 0]], start_cell=[2, 2], goal_cell=[0, 0], start_yaw_deg=180),
        't_junction': dict(task_id='debug_t', occupancy=[[0, 0, 0], [1, 0, 1], [1, 0, 1]], start_cell=[2, 1], goal_cell=[0, 0], start_yaw_deg=90),
    }


def scene_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
