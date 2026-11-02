"""Deterministic high-level exploration simulator for the generalized v2 contracts.

Not a flight dynamics simulator or mission-value planner; Step 3 adds a deterministic safety gate.
The offboard side sees only `MissionSnapshot`; ground truth stays private.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

from interfaces import MissionSpec, MissionSnapshot, PlannerDecision, validate_decision, validate_spec_snapshot
from interfaces.spec import SubjectKind, TargetKind
from interfaces.state import (
    Position, GoalProgress, GoalStatus, WorldEntity, Frontier, Observation,
    ActionTarget, ActionCost, CandidateAction, Resources
)
from interfaces.validation import _matching, StaleDecisionError, UnknownActionError
from .grid import Cell, GridMap, UNKNOWN, FREE, OCCUPIED, ray_cells
from safety import SafetySupervisor, SafetyEvent, SafetyViolation
from .scenario import Scenario, HiddenObject


class ExecutionError(ValueError):
    pass


class ExplorationSimulator:
    """2D occupancy/raycast simulator with no hidden truth in published messages."""
    def __init__(self, spec: MissionSpec, scenario: Scenario, *, seed: int = 0):
        import random
        self.spec = spec
        self.scenario = scenario
        self._rng = random.Random(seed)
        self.map: GridMap = scenario.create_map()
        self.home = scenario.home
        self.position = self.home
        self.state_seq = 0
        self.map_version = 0
        self.now = datetime(2026, 11, 2, tzinfo=timezone.utc)
        self.remaining_energy_wh = scenario.initial_energy_wh
        self.remaining_time_s = scenario.initial_time_s
        self._observations: list[Observation] = []
        self._entities: dict[str, WorldEntity] = {}
        self._bindings: dict[str, str] = {}
        self._obs_counter = 0
        self._home_return_executed = False
        self.path_history: list[Cell] = [self.home]
        self.action_history: list[str] = []
        self.safety = SafetySupervisor()
        self.safety_events: list[SafetyEvent] = []
        self.safety_mode = 'NORMAL'
        self.terminal_contingency = False
        # Sample hidden target locations from scenario-approved alternatives.
        # The planner never sees either the choices or the sampled truth.
        objects = []
        for obj in scenario.objects:
            options = obj.placement_options or [obj.cell]
            row, col = self._rng.choice(options)
            objects.append(obj.model_copy(update={'row': row, 'col': col}))
        self._hidden_objects = objects
        self._object_index = {obj.id: obj for obj in objects}
        if spec.schema_version != '2.0' or spec.robot_id == '':
            raise ValueError('expected v2 schema and valid robot')
        self._scan()  # initial real observation, but NO automatic home verification
        self.snapshot()  # fail early if scenario/spec are inconsistent

    def _pos(self, p: Cell) -> Position:
        return Position(x=float(p[1]) + 0.5, y=float(p[0]) + 0.5,
                        z=float(self.scenario.altitude_m), frame_id='map')

    def _from_pos(self, p: Position) -> Cell:
        return (int(p.y), int(p.x))

    def _record(self, subject_id: str, subject_class: str, prop: str, value,
                confidence: float, producer: str) -> None:
        # Never allow a publisher from the planner to inject trusted observations.
        self._obs_counter += 1
        self._observations.append(Observation(
            id=f'obs_{self._obs_counter:05d}', subject_id=subject_id,
            subject_class=subject_class, property=prop, value=value,
            confidence=float(confidence), producer_id=producer,
            producer_kind='SIMULATOR', observed_at=self.now,
            frame_id='map', artifact_ref=None))

    def _scan(self) -> None:
        """Observe geometry and visible object presence. Properties need INSPECT."""
        if self.map.reveal(self.position):
            self.map_version += 1
        for obj in sorted(self._hidden_objects, key=lambda o: o.id):
            if obj.id in self._entities:
                continue
            if (self.map.distance(self.position, obj.cell) <= self.scenario.detection_radius
                    and self.map.clear_line_of_sight(self.position, obj.cell)):
                self._entities[obj.id] = WorldEntity(
                    id=obj.id, class_id=obj.class_id, confidence=0.96,
                    position=self._pos(obj.cell))
                self._record(obj.id, obj.class_id, 'detected', True, 0.96, 'sim_detector')
        if self.scenario.include_coverage:
            new_coverage = self.map.coverage()
            old_coverage = next((o.value for o in reversed(self._observations)
                                 if o.property == 'coverage_fraction'), None)
            if old_coverage != new_coverage:
                for rid in self.scenario.regions:
                    self._record(rid, 'region', 'coverage_fraction', float(new_coverage),
                                 1.0, self.scenario.coverage_producer)

    def _goal_progress(self) -> tuple[list[GoalProgress], dict[str, str]]:
        """Deterministically infer VERIFIED only from matching evidence and dependencies."""
        progress: dict[str, GoalProgress] = {}
        previous_bindings = dict(self._bindings)
        bindings = {}  # Rebuild from CURRENT matching evidence; expired facts remove bindings.
        # Static topological ordering; never infer status from the LLM.
        remaining = list(self.spec.goals)
        while remaining:
            ready = [g for g in remaining if all(d in progress for d in g.depends_on)]
            assert ready, 'MissionSpec validates acyclic goal graph'
            for goal in ready:
                remaining.remove(goal)
                if any(progress[d].status != GoalStatus.VERIFIED for d in goal.depends_on):
                    progress[goal.id] = GoalProgress(goal_id=goal.id, status=GoalStatus.UNKNOWN)
                    continue
                # _matching requires the real bindings; create a minimal snapshot for it.
                proxy = self._comparison_snapshot(bindings)
                supported = []
                for rule in goal.evidence_rules:
                    valid = [o for o in reversed(self._observations) if _matching(o, rule, proxy)]
                    if goal.bind_entity_as and previous_bindings.get(goal.bind_entity_as):
                        old_id = previous_bindings[goal.bind_entity_as]
                        valid.sort(key=lambda obs: obs.subject_id != old_id)
                    supported.append(valid[0] if valid else None)
                satisfied = (all(o is not None for o in supported) if goal.evidence_logic == 'ALL'
                             else any(o is not None for o in supported))
                if satisfied:
                    chosen = [o for o in supported if o is not None]
                    if goal.bind_entity_as:
                        bindings[goal.bind_entity_as] = chosen[0].subject_id
                    progress[goal.id] = GoalProgress(
                        goal_id=goal.id, status=GoalStatus.VERIFIED,
                        evidence_ids=list(dict.fromkeys(o.id for o in chosen)))
                else:
                    progress[goal.id] = GoalProgress(goal_id=goal.id, status=GoalStatus.PENDING)
        self._bindings = bindings
        return [progress[g.id] for g in self.spec.goals], bindings

    def _comparison_snapshot(self, bindings: dict[str, str]) -> MissionSnapshot:
        """Temporary runtime shell: only used by the v2 observation matcher."""
        return MissionSnapshot(
            schema_version='2.0', mission_id=self.spec.mission_id,
            spec_revision=self.spec.revision, state_seq=self.state_seq,
            map_version=self.map_version, captured_at=self.now,
            robot_pose=self._pos(self.position),
            goals=[GoalProgress(goal_id=g.id, status=GoalStatus.PENDING) for g in self.spec.goals],
            bindings=bindings, entities=list(self._entities.values()),
            frontiers=[], viewpoint_ids=[], region_ids=self.scenario.regions,
            observations=self._observations,
            resources=self._resources(), candidates=[])

    def _cost(self, endpoint: Cell, *, dwell_s: float = 2.0,
              return_action: bool = False) -> ActionCost | None:
        path = self.map.path(self.position, endpoint)
        home_path = self.map.path(endpoint, self.home)
        if path is None or home_path is None:
            return None
        distance = len(path)-1
        return_distance = len(home_path)-1
        return ActionCost(
            execute_energy_wh=float(distance * 0.40 + (0 if return_action else 0.15)),
            return_energy_wh_from_endpoint=float(return_distance * 0.40),
            execute_time_s=float(distance * 2.0 + (0 if return_action else dwell_s)),
            return_time_s_from_endpoint=float(return_distance * 2.0))

    def _resources(self) -> Resources:
        path = self.map.path(self.position, self.home)
        distance = len(path)-1 if path is not None else 0
        return Resources(
            remaining_energy_wh=float(self.remaining_energy_wh),
            reserve_energy_wh=float(self.scenario.reserve_energy_wh),
            remaining_flight_time_s=float(self.remaining_time_s),
            reserve_time_s=float(self.scenario.reserve_time_s),
            current_return_energy_wh=float(distance * 0.4),
            current_return_time_s=float(distance * 2.0))

    def _frontiers(self, *, max_count: int = 7) -> list[Frontier]:
        cells = self.map.frontier_cells(self.position)
        distances = self.map.reachable(self.position)
        scored = sorted(
            ((-self.map.gain(p), distances[p], p) for p in cells),
            key=lambda x: (x[0], x[1], x[2]))
        selected: list[Cell] = []
        for negative_gain, steps, p in scored:
            if len(selected) >= max_count:
                break
            if -negative_gain < 1:
                continue
            if any(self.map.distance(p, q) < 3 for q in selected):
                continue
            selected.append(p)
        return [Frontier(
            id=f'frontier_{p[0]}_{p[1]}', map_version=self.map_version,
            observation_pose=self._pos(p),
            expected_information_gain=float(self.map.gain(p)),
            # Proxy for local map support, not calibrated or target-likelihood.
            confidence=float(min(0.95, 0.55 + 0.1 * sum(
                self.map.known[r][c] != UNKNOWN for r, c in self.map.neighbors(p)))) )
            for p in selected]

    def _estimated_clear_los(self, start: Cell, end: Cell) -> bool:
        """Viewpoint estimation uses only the KNOWN map, never hidden terrain."""
        return all(self.map.known[r][c] != OCCUPIED for r, c in ray_cells(start, end))

    def _inspect_endpoint(self, obj: HiddenObject) -> Cell | None:
        """Choose a known-reachable free viewpoint with estimated visibility.

        This function cannot access hidden terrain: an occluded candidate can
        be proposed, then yield no property observation when executed.
        """
        reachable = self.map.reachable(self.position)
        # Location is obtained from the published detection, not from unrevealed truth.
        target_cell = self._from_pos(self._entities[obj.id].position)
        options = [(d, p) for p, d in reachable.items()
                   if self.map.distance(p, target_cell) <= self.scenario.inspection_radius
                   and self._estimated_clear_los(p, target_cell)]
        return min(options)[1] if options else None

    def _candidates(self, progress: list[GoalProgress], frontiers: list[Frontier]) -> list[CandidateAction]:
        caps = {cap.id for cap in self.spec.capabilities}
        outstanding = [g.goal_id for g in progress if g.status == GoalStatus.PENDING and g.goal_id != 'return']
        actions: list[CandidateAction] = []
        if not self._home_return_executed and 'uav.explore' in caps:
            for frontier in frontiers:
                endpoint = self._from_pos(frontier.observation_pose)
                cost = self._cost(endpoint)
                if cost is None: continue
                actions.append(CandidateAction(
                    id=f'explore_{frontier.id}', capability_id='uav.explore',
                    target=ActionTarget(kind=TargetKind.FRONTIER, ref=frontier.id),
                    relevant_goal_ids=outstanding, cost=cost,
                    geometric_gain=frontier.expected_information_gain))
        if not self._home_return_executed and 'uav.inspect' in caps:
            for obj_id in sorted(self._entities):
                obj = self._object_index[obj_id]
                needs = [g.id for g in self.spec.goals for rule in g.evidence_rules
                         if g.id in outstanding and rule.property in obj.properties
                         and (rule.subject.kind == SubjectKind.CLASS and rule.subject.ref == obj.class_id
                              or rule.subject.kind == SubjectKind.ENTITY and rule.subject.ref == obj.id
                              or rule.subject.kind == SubjectKind.BINDING
                              and self._bindings.get(rule.subject.ref) == obj.id)]
                if not needs: continue
                endpoint = self._inspect_endpoint(obj)
                if endpoint is None: continue
                cost = self._cost(endpoint, dwell_s=5.0)
                if cost is None: continue
                actions.append(CandidateAction(
                    id=f'inspect_{obj_id}', capability_id='uav.inspect',
                    target=ActionTarget(kind=TargetKind.ENTITY, ref=obj_id),
                    relevant_goal_ids=sorted(set(needs)), cost=cost, evidence_gain_hint=0.8))
        if not self._home_return_executed and 'uav.return_home' in caps:
            cost = self._cost(self.home, return_action=True)
            if cost is not None:
                actions.append(CandidateAction(
                    id='return_home', capability_id='uav.return_home',
                    target=ActionTarget(kind=TargetKind.NONE), relevant_goal_ids=[
                        g.id for g in self.spec.goals if g.id == 'return'], cost=cost))
        return actions

    def snapshot(self) -> MissionSnapshot:
        progress, bindings = self._goal_progress()
        frontiers = [] if self._home_return_executed else self._frontiers()
        snap = MissionSnapshot(
            schema_version='2.0', mission_id=self.spec.mission_id,
            spec_revision=self.spec.revision, state_seq=self.state_seq,
            map_version=self.map_version, captured_at=self.now,
            robot_pose=self._pos(self.position), goals=progress, bindings=bindings,
            entities=list(self._entities.values()), frontiers=frontiers,
            viewpoint_ids=[], region_ids=self.scenario.regions,
            observations=list(self._observations), resources=self._resources(),
            candidates=self._candidates(progress, frontiers))
        validate_spec_snapshot(self.spec, snap)
        return snap

    def _known_return_route(self) -> bool:
        return self.map.path(self.position, self.home) is not None

    def safety_assessments(self, snap: MissionSnapshot | None = None):
        snap = snap if snap is not None else self.snapshot()
        return {a.id: self.safety.assess(snap, a, return_route_known=self._known_return_route())
                for a in snap.candidates}

    def _event(self, action_id, status, reason, source='SYSTEM'):
        self.safety_events.append(SafetyEvent(self.state_seq, action_id, status, reason, source))

    def autonomous_recover(self, reason: str = 'PLANNER_UNAVAILABLE') -> MissionSnapshot:
        """Local fallback after timeout, invalid planner output, or no safe optional actions."""
        if self._home_return_executed or self.terminal_contingency:
            return self.snapshot()
        snap = self.snapshot()
        fallback = self.safety.recover(snap, return_route_known=self._known_return_route(), reason=reason)
        if fallback.kind == 'CONTINGENCY_LAND':
            self.terminal_contingency = True
            self.safety_mode = 'CONTINGENCY_LAND'
            self._event(None, 'CONTINGENCY_LAND', fallback.reason)
            self.state_seq += 1
            # This is a terminal simulator state, NOT verified home arrival.
            return self.snapshot()
        self._event(fallback.action_id, 'AUTONOMOUS_RETURN', fallback.reason)
        return self.execute(PlannerDecision(
            schema_version=snap.schema_version, mission_id=snap.mission_id,
            spec_revision=snap.spec_revision, state_seq=snap.state_seq,
            map_version=snap.map_version, action_id=fallback.action_id,
            rationale='Locally supervised recovery'), source='SUPERVISOR')

    def handle_planner_decision(self, decision: PlannerDecision | None) -> MissionSnapshot:
        """Fail-safe boundary for the future offboard LLM integration (Step 4).

        A missing, stale, unknown or resource-infeasible planner decision never
        reaches an unchecked motion command. On error, independently recover.
        Use `execute()` for low-level tests expecting an explicit rejection.
        """
        if decision is None:
            return self.autonomous_recover('PLANNER_UNAVAILABLE')
        try:
            return self.execute(decision)
        except (StaleDecisionError, UnknownActionError, SafetyViolation, ExecutionError) as exc:
            if not isinstance(exc, SafetyViolation):
                self._event(getattr(decision, 'action_id', None), 'REJECTED',
                            type(exc).__name__, 'PLANNER')
            return self.autonomous_recover('PLANNER_DECISION_REJECTED:' + type(exc).__name__)

    def inject_resource_loss(self, *, energy_wh: float = 0.0, time_s: float = 0.0):
        """Test-only telemetry disturbance; always invalidates outstanding decisions."""
        import math
        if not all(math.isfinite(x) and x >= 0 for x in (energy_wh, time_s)):
            raise ValueError('Resource loss must be nonnegative and finite')
        if self._home_return_executed or self.terminal_contingency:
            raise ExecutionError('Mission already terminal')
        self.remaining_energy_wh = max(0.0, self.remaining_energy_wh - energy_wh)
        self.remaining_time_s = max(0.0, self.remaining_time_s - time_s)
        self.state_seq += 1
        self._event(None, 'TELEMETRY_UPDATE', 'SIMULATED_RESOURCE_LOSS')
        return self.snapshot()

    def execute(self, decision: PlannerDecision, *, source: str = 'PLANNER') -> MissionSnapshot:
        """Validate identity, resource safety and local route BEFORE any movement."""
        snap = self.snapshot()
        if self.terminal_contingency or self._home_return_executed:
            raise ExecutionError('Mission already terminal')
        action = validate_decision(self.spec, snap, decision)
        assessment = self.safety.assess(snap, action, return_route_known=self._known_return_route())
        if not assessment.allowed:
            self._event(action.id, 'REJECTED', assessment.reason, source)
            raise SafetyViolation(f'{action.id}: {assessment.reason}')
        if action.capability_id == 'uav.explore':
            frontier = next(f for f in snap.frontiers if f.id == action.target.ref)
            endpoint = self._from_pos(frontier.observation_pose)
        elif action.capability_id == 'uav.inspect':
            obj = self._object_index[action.target.ref]
            endpoint = self._inspect_endpoint(obj)
            if endpoint is None:
                raise ExecutionError('No known-free inspection viewpoint')
        elif action.capability_id == 'uav.return_home':
            endpoint = self.home
        else:
            raise ExecutionError('Step 2 executor supports EXPLORE, INSPECT, RETURN_HOME only')
        route = self.map.path(self.position, endpoint)
        if route is None or any(self.map.known[r][c] != FREE for r,c in route):
            raise ExecutionError('Route must be entirely known free')
        # Recompute from the local known-free route, not remotely supplied costs.
        trusted = self._cost(endpoint, dwell_s=(5.0 if action.capability_id=='uav.inspect' else 2.0),
                             return_action=(action.capability_id=='uav.return_home'))
        if trusted is None or trusted != action.cost:
            self._event(action.id, 'REJECTED', 'COST_OR_ROUTE_CHANGED', source)
            raise SafetyViolation('Route or locally calculated cost changed; fail closed')
        e, t = trusted.execute_energy_wh, trusted.execute_time_s
        if e > self.remaining_energy_wh or t > self.remaining_time_s:
            raise SafetyViolation('Insufficient execution resources')
        self.remaining_energy_wh -= e
        self.remaining_time_s -= t
        self.now += timedelta(seconds=t)
        self.position = endpoint
        self.path_history.extend(route[1:])
        self.state_seq += 1
        self.action_history.append(action.id)
        self._event(action.id, 'EXECUTED' if assessment.mode=='NORMAL' else 'EMERGENCY_RETURN',
                    assessment.reason, source)
        self._scan()
        if action.capability_id == 'uav.inspect':
            obj = self._object_index[action.target.ref]
            if (self.map.distance(self.position, obj.cell) <= self.scenario.inspection_radius
                    and self.map.clear_line_of_sight(self.position, obj.cell)):
                for prop, value in sorted(obj.properties.items()):
                    self._record(obj.id, obj.class_id, prop, value, 0.95,
                                 obj.producers[prop])
        if action.capability_id == 'uav.return_home':
            self._home_return_executed = True
            self.safety_mode = 'HOME'
            self._record(self.spec.robot_id, 'robot', 'at_home', True,
                         1.0, 'sim_executor')
        if not self._home_return_executed:
            current = self.snapshot()
            feasible_optional = any(a.capability_id != 'uav.return_home' and assessment.allowed
                                    for a in current.candidates
                                    for assessment in [self.safety.assess(current, a, return_route_known=self._known_return_route())])
            self.safety_mode = 'NORMAL' if feasible_optional else 'RETURN_REQUIRED'
        return self.snapshot()

    def select_action(self, action_id: str) -> MissionSnapshot:
        """Helper for deterministic demonstrations, not an autonomous planner."""
        snap = self.snapshot()
        return self.execute(PlannerDecision(
            schema_version='2.0', mission_id=snap.mission_id,
            spec_revision=snap.spec_revision, state_seq=snap.state_seq,
            map_version=snap.map_version, action_id=action_id,
            rationale='Scripted Step 2 demonstration'))

    def known_ascii(self) -> str:
        snap = self.snapshot()
        return self.map.ascii(self.position,
                              {self._from_pos(f.observation_pose) for f in snap.frontiers},
                              {self._object_index[e.id].cell for e in snap.entities},
                              self.path_history)
