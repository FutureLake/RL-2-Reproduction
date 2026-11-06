"""Opaque, non-solid goal faces strictly inside the existing goal cell.

Four self-referencing two-sided lines render middle textures. Both sides refer
to the original goal sector: no sector/actor, solid volume, floor/ceiling or
gameplay special is added. This is a local reconstruction, not recovered RL2
scene geometry. Success remains owned exclusively by MazeEpisodeEnv._goal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .scene_builder import CELL, _png, _write_wad
from .target_preview import read_lumps

VERSION = "rl2-task3-visual-stone-goal-v2"
ALIAS = "RLTARG"
SPEC = {
    "method": "four opaque self-referencing two-sided middle-texture faces",
    "footprint_units": 48,
    "inset_from_cell_boundary_units": 40,
    "height_units": 128,
    "texture_rgb": [144, 0, 0],
    "alpha": 1.0,
    "blocking": False,
    "goal_region": "unchanged full original 128x128 cell, player-center rule",
    "red_floor": "original RLGOAL resource and sector reference preserved",
    "lighting": "original lightlevel 240 and renderer directional shading",
    "source": "deterministic local solid-color PNG; visual reference only to RL2 Figure 4 / Duan Figure 8",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks(text, kind):
    return re.findall(rf"\b{kind}\s*\{{([^{{}}]*)\}}", text)


def transform_internal_goal(lumps, task):
    """Append visual lines without editing any pre-existing UDMF block."""
    if ALIAS in {name for name, _ in lumps}:
        raise ValueError("Goal marker already present")
    result = list(lumps)
    index = next(i for i, (name, _) in enumerate(result) if name == "TEXTMAP")
    original = result[index][1].decode("ascii")
    grid = task["occupancy"]
    cells = [(r, c) for r, row in enumerate(grid) for c, wall in enumerate(row) if not wall]
    goal = tuple(task["goal_cell"])
    sector = cells.index(goal)
    sectors = blocks(original, "sector")
    if len(sectors) != len(cells):
        raise ValueError("Expected original one-sector-per-open-cell scene")
    if not re.search(r'texturefloor\s*=\s*"RLGOAL"', sectors[sector]):
        raise ValueError("Task goal does not match source red-floor sector")
    if not re.search(r'heightfloor\s*=\s*0\s*;', sectors[sector]) or not re.search(r'heightceiling\s*=\s*128\s*;', sectors[sector]):
        raise ValueError("Expected original 0..128 sector heights")
    nv, ns = len(blocks(original, "vertex")), len(blocks(original, "sidedef"))
    x0, y0 = goal[1] * CELL, (len(grid) - goal[0] - 1) * CELL
    inset = SPEC["inset_from_cell_boundary_units"]
    lo_x, lo_y, hi_x, hi_y = x0 + inset, y0 + inset, x0 + CELL - inset, y0 + CELL - inset
    corners = [(lo_x, lo_y), (lo_x, hi_y), (hi_x, hi_y), (hi_x, lo_y)]
    additions = [f'vertex {{ x = {x}.0; y = {y}.0; }}' for x, y in corners]
    for i in range(4):
        additions.append(f'linedef {{ v1 = {nv+i}; v2 = {nv+(i+1)%4}; sidefront = {ns+2*i}; sideback = {ns+2*i+1}; twosided = true; blocking = false; blockplayers = false; blockeverything = false; midtex3d = false; midtex3dimpassible = false; special = 0; alpha = 1.0; }}')
        for _ in range(2):
            additions.append(f'sidedef {{ sector = {sector}; texturemiddle = "{ALIAS}"; wrapmidtex = true; clipmidtex = true; }}')
    result[index] = ("TEXTMAP", (original + "\n" + "\n".join(additions)).encode("ascii"))
    texture = _png(tuple(SPEC["texture_rgb"]))
    tx_end = next(i for i, (name, _) in enumerate(result) if name == "TX_END")
    result.insert(tx_end, (ALIAS, texture))
    report = {"visual_version": VERSION, **SPEC, "goal_sector": sector,
              "bounds_xy": [lo_x, lo_y, hi_x, hi_y],
              "added": {"vertex": 4, "linedef": 4, "sidedef": 8, "sector": 0, "thing": 0},
              "texture_sha256": hashlib.sha256(texture).hexdigest(),
              "existing_udmf_prefix_unchanged": True}
    return result, report


def validate_structure(source_lumps, target_lumps, task):
    """Strict whitelist: original blocks/lumps stay identical; four visual faces only."""
    expected, report = transform_internal_goal(source_lumps, task)
    if expected != target_lumps:
        raise AssertionError("Scene differs from the frozen visual-only transform")
    a, b = dict(source_lumps), dict(target_lumps)
    old, new = a["TEXTMAP"].decode("ascii"), b["TEXTMAP"].decode("ascii")
    for kind, extra in report["added"].items():
        before, after = blocks(old, kind), blocks(new, kind)
        assert after[:len(before)] == before
        assert len(after) == len(before) + extra
    assert set(b) - set(a) == {ALIAS}
    assert all(b[k] == v for k, v in a.items() if k != "TEXTMAP")
    return report


def write_internal_goal(source, destination, task):
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("Preserve source and previous visual assets")
    original = read_lumps(source)
    result, report = transform_internal_goal(original, task)
    validate_structure(original, result, task)
    _write_wad(destination, result)
    return {**report, "source_sha256": sha(source), "scene_sha256": sha(destination)}
