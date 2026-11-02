"""2D hidden occupancy grid, ray-cast observations, and known-free pathfinding.

The planner MUST only see `known`, NEVER `terrain`. `terrain` is simulator-only.
Grid cells are 1 m squares; row increases toward positive map-y.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import hypot

UNKNOWN = -1
FREE = 0
OCCUPIED = 1
Cell = tuple[int, int]
NEIGHBORS: tuple[Cell, ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))


def ray_cells(a: Cell, b: Cell) -> list[Cell]:
    """Bresenham: include both endpoints, with deterministic integer traversal."""
    y0, x0 = a
    y1, x1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    cells = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


@dataclass
class GridMap:
    terrain: tuple[str, ...]
    home: Cell
    sensor_radius: int = 4

    def __post_init__(self):
        if not self.terrain or len({len(line) for line in self.terrain}) != 1:
            raise ValueError('terrain must be a nonempty rectangular grid')
        if self.sensor_radius < 1:
            raise ValueError('sensor radius must be positive')
        if any(char not in '#.H' for line in self.terrain for char in line):
            raise ValueError('terrain must use # wall, . free, H home')
        homes = [(r, c) for r, line in enumerate(self.terrain) for c, x in enumerate(line) if x == 'H']
        if homes != [self.home]:
            raise ValueError('exactly one H must match home')
        self.known = [[UNKNOWN for _ in line] for line in self.terrain]

    @property
    def height(self) -> int:
        return len(self.terrain)

    @property
    def width(self) -> int:
        return len(self.terrain[0])

    def inside(self, p: Cell) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width

    def wall(self, p: Cell) -> bool:
        return self.terrain[p[0]][p[1]] == '#'

    def distance(self, a: Cell, b: Cell) -> float:
        return hypot(a[0]-b[0], a[1]-b[1])

    def reveal(self, position: Cell) -> bool:
        """Simulated 360° range scan. A wall is visible, the space behind is not."""
        if not self.inside(position) or self.wall(position):
            raise ValueError('sensor position must be traversable ground truth')
        changed = False
        radius = self.sensor_radius
        for r in range(max(0, position[0]-radius), min(self.height, position[0]+radius+1)):
            for c in range(max(0, position[1]-radius), min(self.width, position[1]+radius+1)):
                target = (r, c)
                if self.distance(position, target) > radius:
                    continue
                for pr, pc in ray_cells(position, target):
                    val = OCCUPIED if self.wall((pr, pc)) else FREE
                    if self.known[pr][pc] != val:
                        self.known[pr][pc] = val
                        changed = True
                    if val == OCCUPIED:
                        break
        return changed

    def clear_line_of_sight(self, start: Cell, end: Cell) -> bool:
        """Ground-truth visibility, used ONLY in the sensor simulator."""
        return (self.inside(start) and self.inside(end)
                and all(not self.wall(p) for p in ray_cells(start, end)))

    def neighbors(self, p: Cell):
        for dr, dc in NEIGHBORS:
            q = (p[0]+dr, p[1]+dc)
            if self.inside(q):
                yield q

    def path(self, start: Cell, goal: Cell) -> list[Cell] | None:
        """BFS over KNOWN FREE cells only; never plans through unknown or occupied."""
        if not self.inside(start) or not self.inside(goal):
            return None
        if self.known[start[0]][start[1]] != FREE or self.known[goal[0]][goal[1]] != FREE:
            return None
        parent: dict[Cell, Cell | None] = {start: None}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            if p == goal:
                path = []
                while p is not None:
                    path.append(p)
                    p = parent[p]
                return path[::-1]
            for q in self.neighbors(p):
                if q not in parent and self.known[q[0]][q[1]] == FREE:
                    parent[q] = p
                    queue.append(q)
        return None

    def reachable(self, start: Cell) -> dict[Cell, int]:
        if self.known[start[0]][start[1]] != FREE:
            return {}
        steps = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for q in self.neighbors(p):
                if q not in steps and self.known[q[0]][q[1]] == FREE:
                    steps[q] = steps[p] + 1
                    queue.append(q)
        return steps

    def frontier_cells(self, current: Cell) -> list[Cell]:
        reachable = self.reachable(current)
        return sorted(p for p in reachable if any(self.known[r][c] == UNKNOWN for r,c in self.neighbors(p)))

    def gain(self, viewpoint: Cell) -> int:
        """Unknown cells potentially observable; only known map used for occlusion."""
        targets: set[Cell] = set()
        radius = self.sensor_radius
        for r in range(max(0, viewpoint[0]-radius), min(self.height, viewpoint[0]+radius+1)):
            for c in range(max(0, viewpoint[1]-radius), min(self.width, viewpoint[1]+radius+1)):
                q=(r,c)
                if self.distance(viewpoint, q) > radius: continue
                for p in ray_cells(viewpoint, q)[1:]:
                    cell = self.known[p[0]][p[1]]
                    if cell == OCCUPIED:
                        break
                    if cell == UNKNOWN:
                        targets.add(p)
        return len(targets)

    def coverage(self) -> float:
        """Known-cell fraction of a predefined rectangular ROI (full grid)."""
        return sum(cell != UNKNOWN for row in self.known for cell in row) / (self.height * self.width)

    def ascii(self, position: Cell, frontiers: set[Cell] | None = None,
              found: set[Cell] | None = None, path: list[Cell] | None = None) -> str:
        """Visualize ONLY observed occupancy and discovered objects."""
        frontiers, found = frontiers or set(), found or set()
        trail = set(path or [])
        lines = []
        for r, line in enumerate(self.known):
            chars = []
            for c, val in enumerate(line):
                p=(r,c)
                ch = '?' if val == UNKNOWN else ('#' if val == OCCUPIED else '.')
                if p in trail and val == FREE: ch='+'
                if p in frontiers: ch='F'
                if p in found: ch='T'
                if p == self.home and val == FREE: ch='H'
                if p == position: ch='D'
                chars.append(ch)
            lines.append(''.join(chars))
        return '\n'.join(reversed(lines))
