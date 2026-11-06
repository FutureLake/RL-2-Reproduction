"""Preview-only renderable goal curtains on existing, nonblocking portals.

This is not enabled in the default visual dataset. It changes only the middle
texture/alpha/wrapping of existing two-sided lines surrounding the goal sector.
No things, vertices, geometry, collision flags or scripts are introduced.
UDMF render fields: https://github.com/rheit/zdoom/blob/master/specs/udmf_zdoom.txt
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct

from .scene_builder import _write_wad


def read_lumps(path):
    data = Path(path).read_bytes()
    magic, count, directory = struct.unpack_from("<4sii", data)
    if magic not in (b"PWAD", b"IWAD"):
        raise ValueError("Not a WAD")
    result = []
    for i in range(count):
        offset, size, name = struct.unpack_from("<ii8s", data, directory + i * 16)
        result.append((name.rstrip(b"\0").decode("ascii"), data[offset:offset + size]))
    return result


def _integer(block, key):
    match = re.search(rf"\b{key}\s*=\s*(-?\d+)\s*;", block)
    if match is None:
        raise ValueError(f"Missing UDMF integer {key}")
    return int(match.group(1))


def _set_field(block, key, value):
    pattern = rf"\b{key}\s*=\s*[^;]+;"
    replacement = f"{key} = {value};"
    return re.sub(pattern, replacement, block) if re.search(pattern, block) else block.rstrip() + " " + replacement + " "


def transform_goal_curtains(lumps, task, *, alpha=0.65):
    """Return new lumps/report; alpha=1 is opaque but remains physically passable."""
    if not 0 < alpha <= 1:
        raise ValueError("Goal curtain alpha must be in (0,1]")
    text_index = next(i for i, (name, _) in enumerate(lumps) if name == "TEXTMAP")
    text = lumps[text_index][1].decode("ascii")
    sectors = re.findall(r"\bsector\s*\{([^{}]*)\}", text)
    sides = re.findall(r"\bsidedef\s*\{([^{}]*)\}", text)
    lines = re.findall(r"\blinedef\s*\{([^{}]*)\}", text)
    cells = [(r, c) for r, row in enumerate(task["occupancy"]) for c, wall in enumerate(row) if not wall]
    goal_sector = cells.index(tuple(task["goal_cell"]))
    if len(sectors) != len(cells):
        raise ValueError("Preview expects one existing sector per open cell")
    side_sector = [_integer(side, "sector") for side in sides]
    selected_lines, selected_sides = set(), set()
    for index, line in enumerate(lines):
        front, back = _integer(line, "sidefront"), _integer(line, "sideback")
        if back < 0:
            continue
        if goal_sector in (side_sector[front], side_sector[back]) and side_sector[front] != side_sector[back]:
            if re.search(r"\bblocking\s*=\s*true\s*;", line):
                raise ValueError("A goal portal was already blocking; not a supported preview")
            selected_lines.add(index)
            selected_sides.update((front, back))
    if not selected_lines:
        raise ValueError("Goal has no passable portal")
    counters = {"linedef": 0, "sidedef": 0}
    def replace(match):
        kind, block = match.group(1), match.group(2)
        index = counters[kind]
        counters[kind] += 1
        if kind == "sidedef" and index in selected_sides:
            block = _set_field(block, "texturemiddle", '"RLGOAL"')
            block = _set_field(block, "wrapmidtex", "true")
            block = _set_field(block, "clipmidtex", "true")
        elif kind == "linedef" and index in selected_lines:
            block = _set_field(block, "alpha", repr(float(alpha)))
        return f"{kind} {{{block}}}"
    updated = re.sub(r"\b(linedef|sidedef)\s*\{([^{}]*)\}", replace, text)
    result = list(lumps)
    result[text_index] = ("TEXTMAP", updated.encode("ascii"))
    return result, {"preview_only": True, "method": "existing nonblocking two-sided portal middle textures",
                    "material": "RLGOAL", "alpha": alpha, "goal_sector": goal_sector,
                    "portal_linedef_indices": sorted(selected_lines), "sidedef_indices": sorted(selected_sides),
                    "new_geometry_or_things": 0, "collision_flags_changed": False,
                    "goal_trigger": "unchanged: player center inside original goal occupancy cell",
                    "caveat": "Adds a visible red surface and changes visual occlusion; not validated as original RL2 target geometry."}


def write_goal_preview(source, destination, task, *, alpha=0.65):
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("Preserve source and prior preview files")
    result, report = transform_goal_curtains(read_lumps(source), task, alpha=alpha)
    _write_wad(destination, result)
    report.update(source_scene=str(source.resolve()), preview_scene=str(destination.resolve()),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  preview_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    return report
