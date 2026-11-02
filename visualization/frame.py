"""Optional visualization message, separate from the compact planning schema.

Only observed occupancy and discovered entities are allowed; no ground truth.
The same message can later be created by a ROS 2 adapter and displayed in RViz.
"""
from __future__ import annotations
from datetime import datetime
from math import isfinite
from pydantic import BaseModel, Field, ConfigDict, model_validator


class _FrameMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class VisualPoint(_FrameMessage):
    x: float
    y: float


class VisualEntity(_FrameMessage):
    id: str
    class_id: str
    position: VisualPoint
    confidence: float = Field(ge=0, le=1)


class VisualFrontier(_FrameMessage):
    id: str
    position: VisualPoint
    information_gain: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class VisualSafety(_FrameMessage):
    """Local debug state only; not part of the offboard planning contract."""
    mode: str
    return_route_known: bool
    feasible_optional_count: int = Field(ge=0)
    rejected_candidate_count: int = Field(ge=0)
    last_status: str | None = None
    last_reason: str | None = None
    last_source: str | None = None


class VisualPlanner(_FrameMessage):
    """Untrusted model recommendation and debug metadata only, never control."""
    mode: str = 'SCRIPTED'
    status: str = 'READY'
    model: str | None = None
    action_id: str | None = None
    rationale: str = ''
    latency_s: float | None = Field(default=None, ge=0)


class VisualizationFrame(_FrameMessage):
    """Renderer input: a safe, observed-state-only view of the running mission.

    A grid coordinate (r,c) is centered at (c+0.5,r+0.5) in `frame_id`.
    `occupancy` uses -1 unknown, 0 free, 1 occupied, row 0 at bottom.
    Unlike MissionSnapshot, contains display-sized occupancy and path history.
    Don't send full frames to an LLM over a low-bandwidth link.
    """
    schema_version: str = 'viz-1.0'
    mission_id: str
    frame_id: str = 'map'
    state_seq: int = Field(ge=0)
    map_version: int = Field(ge=0)
    captured_at: datetime
    occupancy: tuple[tuple[int, ...], ...]
    cell_size_m: float = Field(default=1.0, gt=0)
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    drone: VisualPoint
    home: VisualPoint
    path: tuple[VisualPoint, ...]
    frontiers: tuple[VisualFrontier, ...] = ()
    discovered_entities: tuple[VisualEntity, ...] = ()
    goals: dict[str, str]
    remaining_energy_wh: float = Field(ge=0)
    reserve_energy_wh: float = Field(ge=0)
    remaining_time_s: float = Field(ge=0)
    reserve_time_s: float = Field(ge=0)
    return_energy_wh: float = Field(ge=0)
    return_time_s: float = Field(ge=0)
    selected_action_id: str | None = None
    candidate_action_count: int = Field(ge=0)
    safety: VisualSafety | None = None
    planner: VisualPlanner | None = None

    @model_validator(mode='after')
    def check_grid_and_coordinates(self):
        if self.schema_version != 'viz-1.0':
            raise ValueError('Unknown visualization message version')
        if not self.occupancy or not self.occupancy[0]:
            raise ValueError('Expected a nonempty observed occupancy grid')
        w = len(self.occupancy[0])
        if any(len(row) != w for row in self.occupancy):
            raise ValueError('Occupancy grid must be rectangular')
        if any(cell not in (-1, 0, 1) for row in self.occupancy for cell in row):
            raise ValueError('Grid supports only unknown -1, free 0, occupied 1')
        pts = (self.drone, self.home, *self.path,
               *(f.position for f in self.frontiers),
               *(e.position for e in self.discovered_entities))
        if any(not isfinite(p.x) or not isfinite(p.y) for p in pts):
            raise ValueError('All coordinates must be finite')
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError('Captured time must be timezone-aware')
        return self

    @property
    def coverage(self) -> float:
        observed = sum(cell != -1 for row in self.occupancy for cell in row)
        return observed / (len(self.occupancy) * len(self.occupancy[0]))
