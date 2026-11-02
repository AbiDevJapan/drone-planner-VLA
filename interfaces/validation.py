"""Deterministic referential/evidence validation. NOT sensor authentication or flight safety."""
from datetime import datetime
from .spec import MissionSpec, EvidenceRule, SubjectKind, Compare
from .state import MissionSnapshot, PlannerDecision, GoalStatus, Observation, CandidateAction
from .spec import TargetKind

class InvalidMissionState(ValueError): pass
class StaleDecisionError(ValueError): pass
class UnknownActionError(ValueError): pass


def _equal(a, b):
    # Avoid Python's `True == 1` accident; treat int and float as compatible numerics.
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    if type(a) in (int, float) and type(b) in (int, float):
        return a == b
    return type(a) is type(b) and a == b


def _matching(obs: Observation, rule: EvidenceRule, snap: MissionSnapshot):
    selector = rule.subject
    if selector.kind == SubjectKind.CLASS and obs.subject_class != selector.ref:
        return False
    if selector.kind == SubjectKind.BINDING and snap.bindings.get(selector.ref) != obs.subject_id:
        return False
    if selector.kind in (SubjectKind.ENTITY, SubjectKind.ROBOT, SubjectKind.REGION) and obs.subject_id != selector.ref:
        return False
    if obs.property != rule.property or obs.producer_id not in rule.allowed_producers:
        return False
    if obs.confidence < rule.min_confidence:
        return False
    if rule.max_age_s is not None and (snap.captured_at - obs.observed_at).total_seconds() > rule.max_age_s:
        return False
    if rule.operator == Compare.EXISTS:
        return True
    if rule.operator == Compare.EQ:
        return _equal(obs.value, rule.expected)
    if rule.operator == Compare.ONE_OF:
        return any(_equal(obs.value, candidate) for candidate in rule.expected)
    if rule.operator == Compare.GTE:
        return type(obs.value) in (int, float) and obs.value >= rule.expected
    if rule.operator == Compare.LTE:
        return type(obs.value) in (int, float) and obs.value <= rule.expected
    return False


def validate_spec_snapshot(spec: MissionSpec, snap: MissionSnapshot) -> None:
    """Validate state claims against immutable spec. Does not decide sensor trust."""
    if (spec.schema_version != snap.schema_version or spec.mission_id != snap.mission_id
            or spec.revision != snap.spec_revision):
        raise InvalidMissionState("Mismatched specification identity/revision")
    goals = {g.id: g for g in spec.goals}
    progress = {p.goal_id: p for p in snap.goals}
    if set(goals) != set(progress):
        raise InvalidMissionState("Snapshot must contain exactly the specification goals")
    observations = {o.id: o for o in snap.observations}
    entities = {e.id: e for e in snap.entities}
    for obs in snap.observations:
        if obs.subject_id in entities and obs.subject_class != entities[obs.subject_id].class_id:
            raise InvalidMissionState("Observation subject_class disagrees with world entity")
    for name, entity_id in snap.bindings.items():
        bind_sources = [g for g in spec.goals if g.bind_entity_as == name]
        if not bind_sources or entity_id not in entities:
            raise InvalidMissionState("Unknown binding or entity")
        source_goal = bind_sources[0]
        source_state = progress[source_goal.id]
        if source_state.status != GoalStatus.VERIFIED:
            raise InvalidMissionState("Binding-producing goal has not been verified")
        if not any(
            observations.get(eid) is not None
            and observations[eid].subject_id == entity_id
            and _matching(observations[eid], source_goal.evidence_rules[0], snap)
            for eid in source_state.evidence_ids
        ):
            raise InvalidMissionState("Bound entity not supported by valid goal evidence")
    for g in spec.goals:
        p = progress[g.id]
        if any(eid not in observations for eid in p.evidence_ids):
            raise InvalidMissionState(f"Goal {g.id}: unknown evidence ID")
        if p.status == GoalStatus.VERIFIED:
            if any(progress[d].status != GoalStatus.VERIFIED for d in g.depends_on):
                raise InvalidMissionState(f"Goal {g.id}: dependencies not verified")
            supporting = [observations[eid] for eid in p.evidence_ids]
            passed = [any(_matching(obs, rule, snap) for obs in supporting) for rule in g.evidence_rules]
            if not (all(passed) if g.evidence_logic == "ALL" else any(passed)):
                raise InvalidMissionState(f"Goal {g.id}: insufficient matching evidence")
            if g.bind_entity_as and g.bind_entity_as not in snap.bindings:
                raise InvalidMissionState(f"Goal {g.id}: verified but binding missing")
    caps = {c.id: c for c in spec.capabilities}
    frontier_ids = {f.id for f in snap.frontiers}
    goal_ids = set(goals)
    target_refs = {
        TargetKind.FRONTIER: frontier_ids,
        TargetKind.ENTITY: set(entities),
        TargetKind.VIEWPOINT: set(snap.viewpoint_ids),
        TargetKind.GOAL: goal_ids,
        TargetKind.REGION: set(snap.region_ids),
    }
    for a in snap.candidates:
        cap = caps.get(a.capability_id)
        if cap is None:
            raise InvalidMissionState(f"Candidate {a.id}: unavailable capability")
        if a.target.kind not in cap.allowed_target_kinds:
            raise InvalidMissionState(f"Candidate {a.id}: wrong target kind")
        if a.target.kind != TargetKind.NONE and a.target.ref not in target_refs[a.target.kind]:
            raise InvalidMissionState(f"Candidate {a.id}: unknown target")
        if any(goal_id not in goal_ids for goal_id in a.relevant_goal_ids):
            raise InvalidMissionState(f"Candidate {a.id}: unknown relevant goal")
        # Step 3: actual feasibility checked by independent onboard supervisor,
        # including return estimates from the action's predicted endpoint.


def validate_decision(spec: MissionSpec, snap: MissionSnapshot, decision: PlannerDecision) -> CandidateAction:
    validate_spec_snapshot(spec, snap)
    if (decision.schema_version != snap.schema_version
            or decision.mission_id != snap.mission_id
            or decision.spec_revision != snap.spec_revision
            or decision.state_seq != snap.state_seq
            or decision.map_version != snap.map_version):
        raise StaleDecisionError("Decision made for stale or different mission state")
    for a in snap.candidates:
        if a.id == decision.action_id:
            return a
    raise UnknownActionError("Decision action_id not present in supplied candidates")
