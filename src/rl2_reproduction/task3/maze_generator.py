"""Finite 5x5 maze support and reproducible random-Kruskal tree mazes.

Rows increase downwards; world x increases right and world y increases up.
Random Kruskal is deliberately not described as a uniform tree sampler.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from typing import Iterable

import numpy as np

Cell = tuple[int, int]


def _node_edges(nodes: int) -> list[tuple[Cell, Cell]]:
    edges = []
    for row in range(nodes):
        for col in range(nodes):
            if row + 1 < nodes:
                edges.append(((row, col), (row + 1, col)))
            if col + 1 < nodes:
                edges.append(((row, col), (row, col + 1)))
    return edges


def _tree_grid(size: int, edges: Iterable[tuple[Cell, Cell]]) -> np.ndarray:
    grid = np.ones((size, size), dtype=np.uint8)
    grid[::2, ::2] = 0
    for first, second in edges:
        grid[first[0] + second[0], first[1] + second[1]] = 0
    return grid


def _validate_size(size: int) -> None:
    if not isinstance(size, int) or size < 3 or size % 2 == 0:
        raise ValueError("maze_size must be an odd integer >= 3")


def random_kruskal(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a connected tree on the specified odd occupancy-grid size."""
    _validate_size(size)
    nodes = (size + 1) // 2
    parent = list(range(nodes * nodes))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = _node_edges(nodes)
    accepted = []
    for index in rng.permutation(len(edges)):
        first, second = edges[int(index)]
        a = find(first[0] * nodes + first[1])
        b = find(second[0] * nodes + second[1])
        if a != b:
            parent[a] = b
            accepted.append((first, second))
        if len(accepted) == nodes * nodes - 1:
            break
    return _tree_grid(size, accepted)


def layout_key(grid: np.ndarray | list) -> str:
    array = np.asarray(grid)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("occupancy must be square")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("occupancy must contain only 0 and 1")
    return "".join(str(int(item)) for item in array.flat)


def layout_hash(grid: np.ndarray | list) -> str:
    return sha256(layout_key(grid).encode("ascii")).hexdigest()


def canonical_key(grid: np.ndarray | list) -> str:
    array = np.asarray(grid, dtype=np.uint8)
    return min(
        layout_key(transformed)
        for turns in range(4)
        for transformed in (np.rot90(array, turns), np.fliplr(np.rot90(array, turns)))
    )


def canonical_group_id(grid: np.ndarray | list) -> str:
    return "g-" + sha256(canonical_key(grid).encode("ascii")).hexdigest()[:16]


@lru_cache(maxsize=1)
def _enumerated_keys() -> tuple[str, ...]:
    edges = _node_edges(3)
    keys = []
    for selected in combinations(edges, 8):
        parent = list(range(9))

        def find(index: int) -> int:
            while parent[index] != index:
                index = parent[index]
            return index

        for first, second in selected:
            a = find(first[0] * 3 + first[1])
            b = find(second[0] * 3 + second[1])
            if a == b:
                break
            parent[a] = b
        else:
            keys.append(layout_key(_tree_grid(5, selected)))
    return tuple(sorted(keys))


def enumerate_layouts() -> list[np.ndarray]:
    """Return all 192 fixed-orientation 5x5 layouts, each as a fresh array."""
    return [np.fromiter(map(int, key), dtype=np.uint8).reshape(5, 5) for key in _enumerated_keys()]


def split_layouts(split_seed: int = 0) -> dict[str, list[np.ndarray]]:
    layouts = enumerate_layouts()
    groups = sorted({canonical_key(grid) for grid in layouts})
    if len(layouts) != 192 or len(groups) != 28:
        raise AssertionError("The finite maze support has changed")
    order = np.random.default_rng(split_seed).permutation(len(groups))
    split_indices = {"train": order[:20], "val": order[20:23], "test": order[23:]}
    return {
        split: [grid for grid in layouts if canonical_key(grid) in {groups[int(i)] for i in indices}]
        for split, indices in split_indices.items()
    }


def shortest_path(grid: np.ndarray | list, start: Cell, goal: Cell) -> list[Cell]:
    array = np.asarray(grid)
    start, goal = tuple(start), tuple(goal)
    rows, cols = array.shape

    def open_cell(cell: Cell) -> bool:
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols and array[cell] == 0

    if not open_cell(start) or not open_cell(goal):
        raise ValueError("start and goal must be passable cells")
    parents: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return list(reversed(path))
        for dr, dc in ((0, 1), (-1, 0), (0, -1), (1, 0)):
            nxt = (current[0] + dr, current[1] + dc)
            if open_cell(nxt) and nxt not in parents:
                parents[nxt] = current
                queue.append(nxt)
    raise ValueError("goal is disconnected from start")


def path_turn_count(path: list[Cell]) -> int:
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def cell_center(cell: Cell, size: int, cell_size: float = 128.0) -> tuple[float, float]:
    row, col = cell
    return (col + 0.5) * cell_size, (size - row - 0.5) * cell_size


def cell_from_position(x: float, y: float, size: int, cell_size: float = 128.0) -> Cell:
    """World cells use half-open [lower, upper) intervals on both axes."""
    return size - 1 - int(np.floor(y / cell_size)), int(np.floor(x / cell_size))


def reference_action_plan(task: dict) -> list[int]:
    """Nominal centerline actions, never supplied to the learned policy.

    Positive world yaw is counter-clockwise. Left/right turns are 0/1 and
    forward is 2. The controller may stop before this plan ends at goal entry.
    """
    path = shortest_path(task["occupancy"], tuple(task["start_cell"]), tuple(task["goal_cell"]))
    yaw = int(task["start_yaw_deg"]) % 360
    if yaw % 90:
        raise ValueError("reference plans require cardinal initial yaw")
    yaw_by_delta = {(0, 1): 0, (-1, 0): 90, (0, -1): 180, (1, 0): 270}
    actions = []
    for first, second in zip(path, path[1:]):
        target = yaw_by_delta[second[0] - first[0], second[1] - first[1]]
        delta = (target - yaw + 180) % 360 - 180
        actions.extend([0 if delta > 0 else 1] * (abs(delta) // 5))
        actions.extend([2] * 5)
        yaw = target
    return actions


def legal_configurations(grid: np.ndarray | list) -> list[tuple[Cell, Cell, int, int]]:
    """Finite (start, goal, yaw, nominal_steps) candidates, before engine checks."""
    array = np.asarray(grid, dtype=np.uint8)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(array == 0)]
    candidates = []
    yaw_by_delta = {(0, 1): 0, (-1, 0): 90, (0, -1): 180, (1, 0): 270}
    for start in cells:
        for goal in cells:
            path = shortest_path(array, start, goal)
            turns = path_turn_count(path)
            if len(path) < 4 or turns == 0:
                continue
            first_yaw = yaw_by_delta[path[1][0] - path[0][0], path[1][1] - path[0][1]]
            for yaw in (0, 90, 180, 270):
                delta = abs((first_yaw - yaw + 180) % 360 - 180)
                steps = 5 * (len(path) - 1) + 18 * turns + delta // 5
                candidates.append((start, goal, yaw, steps))
    return candidates
