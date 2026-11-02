"""Static mission specification. Same schema for many mission types."""
from enum import Enum
from typing import Literal
from pydantic import Field, model_validator
from .common import Message, Id, Probability, NonNegative, PositiveWeight, Scalar

SCHEMA_VERSION = "2.0"

class SubjectKind(str, Enum):
    ENTITY = "ENTITY"       # An instance by stable ID
    CLASS = "CLASS"         # Any instance of a semantic class
    BINDING = "BINDING"     # Instance selected by an earlier goal
    ROBOT = "ROBOT"         # Robot by ID
    REGION = "REGION"       # Region by ID

class SubjectSelector(Message):
    kind: SubjectKind
    ref: Id

class Compare(str, Enum):
    EQ = "EQ"
    ONE_OF = "ONE_OF"
    GTE = "GTE"
    LTE = "LTE"
    EXISTS = "EXISTS"

class EvidenceRule(Message):
    """Predicate matched against an actual observation, not an LLM assertion."""
    id: Id
    subject: SubjectSelector
    property: Id
    operator: Compare
    expected: Scalar | list[Scalar] | None = None
    min_confidence: Probability
    allowed_producers: list[Id] = Field(min_length=1)
    max_age_s: NonNegative | None = None

    @model_validator(mode="after")
    def check_operator(self):
        v = self.expected
        if self.operator == Compare.EXISTS:
            if v is not None:
                raise ValueError("EXISTS must not include expected")
        elif self.operator == Compare.ONE_OF:
            if not isinstance(v, list) or not v:
                raise ValueError("ONE_OF requires nonempty expected list")
        elif self.operator in (Compare.GTE, Compare.LTE):
            if type(v) not in (float, int):
                raise ValueError("numeric comparison requires a number")
        elif isinstance(v, list) or v is None:
            raise ValueError("EQ requires one expected scalar")
        if len(self.allowed_producers) != len(set(self.allowed_producers)):
            raise ValueError("Duplicate allowed producer")
        return self

class GoalDefinition(Message):
    id: Id
    description: str = Field(min_length=1, max_length=400)
    priority: PositiveWeight = 1.0
    depends_on: list[Id] = Field(default_factory=list)
    evidence_logic: Literal["ALL", "ANY"] = "ALL"
    evidence_rules: list[EvidenceRule] = Field(min_length=1)
    bind_entity_as: Id | None = None

    @model_validator(mode="after")
    def validate_goal(self):
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("Duplicate goal dependencies")
        ids = [r.id for r in self.evidence_rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence rule IDs")
        if self.bind_entity_as and len(self.evidence_rules) != 1:
            raise ValueError("Entity binding currently requires one evidence rule")
        return self

class TargetKind(str, Enum):
    NONE = "NONE"
    FRONTIER = "FRONTIER"
    ENTITY = "ENTITY"
    VIEWPOINT = "VIEWPOINT"
    GOAL = "GOAL"
    REGION = "REGION"

class ActionCapability(Message):
    id: Id
    allowed_target_kinds: list[TargetKind] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=240)

class MissionSpec(Message):
    schema_version: Literal["2.0"]
    mission_id: Id
    revision: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=500)
    robot_id: Id
    goals: list[GoalDefinition] = Field(min_length=1)
    capabilities: list[ActionCapability] = Field(min_length=1)

    @model_validator(mode="after")
    def check_graph(self):
        goals = {g.id: g for g in self.goals}
        if len(goals) != len(self.goals):
            raise ValueError("Duplicate goal IDs")
        capabilities = {c.id for c in self.capabilities}
        if len(capabilities) != len(self.capabilities):
            raise ValueError("Duplicate capability IDs")
        binding_sources = {g.bind_entity_as: g.id for g in self.goals if g.bind_entity_as}
        if len(binding_sources) != len([g for g in self.goals if g.bind_entity_as]):
            raise ValueError("Duplicate binding names")
        for g in self.goals:
            for dep in g.depends_on:
                if dep not in goals:
                    raise ValueError(f"Unknown dependency {dep}")
                if dep == g.id:
                    raise ValueError("Self dependency")
            for rule in g.evidence_rules:
                if rule.subject.kind == SubjectKind.BINDING:
                    if rule.subject.ref not in binding_sources:
                        raise ValueError(f"Unknown binding name {rule.subject.ref}")
                    if binding_sources[rule.subject.ref] not in g.depends_on:
                        raise ValueError("Binding-producing goal must be a dependency")
        visiting, visited = set(), set()
        def visit(key):
            if key in visiting: raise ValueError("Cyclic goal dependencies")
            if key in visited: return
            visiting.add(key)
            for dep in goals[key].depends_on: visit(dep)
            visiting.remove(key)
            visited.add(key)
        for key in goals: visit(key)
        return self
