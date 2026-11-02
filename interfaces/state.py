"""Dynamic mission snapshot. Observations are short facts, not images or clouds."""
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import Field, model_validator
from .common import Message, Id, Probability, FiniteFloat, NonNegative, Scalar
from .spec import TargetKind

class GoalStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"

class GoalProgress(Message):
    goal_id: Id
    status: GoalStatus
    evidence_ids: list[Id] = Field(default_factory=list)
    failure_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def consistency(self):
        if self.status == GoalStatus.VERIFIED and not self.evidence_ids:
            raise ValueError("VERIFIED goal requires evidence IDs")
        if (self.status == GoalStatus.FAILED) != (self.failure_reason is not None):
            raise ValueError("FAILED iff failure_reason is provided")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Duplicate goal evidence IDs")
        return self

class Position(Message):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    frame_id: Id  # e.g. map, not an unlabelled xyz

class WorldEntity(Message):
    id: Id
    class_id: Id
    confidence: Probability
    position: Position | None = None

class Frontier(Message):
    id: Id
    map_version: int = Field(ge=0)
    observation_pose: Position
    expected_information_gain: NonNegative
    confidence: Probability  # geometric, NOT target-existence probability

class Observation(Message):
    id: Id
    subject_id: Id
    subject_class: Id  # e.g. fire_extinguisher, robot, exit_sign
    property: Id
    value: Scalar
    confidence: Probability
    producer_id: Id
    producer_kind: Literal["SIMULATOR", "PERCEPTION", "GEOMETRY", "NAVIGATION", "EXECUTOR"]
    observed_at: datetime
    frame_id: Id | None = None
    artifact_ref: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def require_tz(self):
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("Observation time must include UTC offset")
        return self

class ActionTarget(Message):
    kind: TargetKind
    ref: Id | None = None

    @model_validator(mode="after")
    def target(self):
        if (self.kind == TargetKind.NONE) != (self.ref is None):
            raise ValueError("NONE must have no ref; every other target requires a ref")
        return self

class ActionCost(Message):
    execute_energy_wh: NonNegative
    return_energy_wh_from_endpoint: NonNegative
    execute_time_s: NonNegative
    return_time_s_from_endpoint: NonNegative

class CandidateAction(Message):
    id: Id
    capability_id: Id
    target: ActionTarget
    relevant_goal_ids: list[Id]
    cost: ActionCost
    geometric_gain: NonNegative | None = None
    evidence_gain_hint: Probability | None = None  # heuristic; NOT verified progress

class Resources(Message):
    remaining_energy_wh: NonNegative
    reserve_energy_wh: NonNegative
    remaining_flight_time_s: NonNegative
    reserve_time_s: NonNegative
    current_return_energy_wh: NonNegative
    current_return_time_s: NonNegative

class MissionSnapshot(Message):
    schema_version: Literal["2.0"]
    mission_id: Id
    spec_revision: int = Field(ge=1)
    state_seq: int = Field(ge=0)
    map_version: int = Field(ge=0)
    captured_at: datetime
    robot_pose: Position
    goals: list[GoalProgress]
    bindings: dict[Id, Id] = Field(default_factory=dict)
    entities: list[WorldEntity]
    frontiers: list[Frontier]
    viewpoint_ids: list[Id]
    region_ids: list[Id]
    observations: list[Observation]
    resources: Resources
    candidates: list[CandidateAction]

    @model_validator(mode="after")
    def internal_integrity(self):
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("Snapshot time must include UTC offset")
        for label, seq, key in [
            ("goal", self.goals, "goal_id"), ("entity", self.entities, "id"),
            ("frontier", self.frontiers, "id"), ("observation", self.observations, "id"),
            ("candidate", self.candidates, "id")]:
            ids = [getattr(x, key) for x in seq]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {label} ID")
        if len(self.viewpoint_ids) != len(set(self.viewpoint_ids)):
            raise ValueError("Duplicate viewpoint IDs")
        if len(self.region_ids) != len(set(self.region_ids)):
            raise ValueError("Duplicate region IDs")
        if any(f.map_version != self.map_version for f in self.frontiers):
            raise ValueError("Frontiers must use current map version")
        if any(e.observed_at > self.captured_at for e in self.observations):
            raise ValueError("Observation newer than snapshot")
        return self

class PlannerDecision(Message):
    schema_version: Literal["2.0"]
    mission_id: Id
    spec_revision: int = Field(ge=1)
    state_seq: int = Field(ge=0)
    map_version: int = Field(ge=0)
    action_id: Id
    rationale: str = Field(max_length=240)
