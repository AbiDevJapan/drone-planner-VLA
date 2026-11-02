"""Step 3: independent execution gate for high-level UAV actions.

Takes locally generated, version-checked MissionSnapshot/CandidateAction messages.
No LLM, visualization, simulator terrain, or remote planner dependencies.

Assumptions: positive, deterministic, optimistic energy/time estimates. Real-flight
use requires calibrated conservative estimates, telemetry freshness, route validity,
flight-controller failsafes and an actual landing safety assessment.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from interfaces.state import CandidateAction, MissionSnapshot


class SafetyViolation(ValueError):
    """Planner requested a valid candidate that fails independent safety checks."""


@dataclass(frozen=True)
class SafetyAssessment:
    action_id: str
    allowed: bool
    mode: Literal['NORMAL', 'EMERGENCY_RETURN', 'REJECTED']
    reason: str
    energy_margin_wh: float
    time_margin_s: float


@dataclass(frozen=True)
class RecoveryDecision:
    kind: Literal['RETURN_HOME', 'CONTINGENCY_LAND']
    reason: str
    action_id: str | None = None


@dataclass(frozen=True)
class SafetyEvent:
    state_seq: int
    action_id: str | None
    status: str
    reason: str
    source: Literal['PLANNER', 'SUPERVISOR', 'SYSTEM'] = 'SYSTEM'


class SafetySupervisor:
    """Fail-closed checks and autonomous recovery recommendation.

    This is not the simulator engine: the engine must call this gate before
    moving/consuming resources. Remote clients may provide ONLY an action ID.
    Return-route reachability must come from a trusted local path estimator.
    """
    EPS = 1e-8

    def assess(self, snapshot: MissionSnapshot, action: CandidateAction,
               *, return_route_known: bool) -> SafetyAssessment:
        r, c = snapshot.resources, action.cost
        energy_margin = (r.remaining_energy_wh - c.execute_energy_wh
                         - c.return_energy_wh_from_endpoint - r.reserve_energy_wh)
        time_margin = (r.remaining_flight_time_s - c.execute_time_s
                       - c.return_time_s_from_endpoint - r.reserve_time_s)
        if not return_route_known:
            return SafetyAssessment(action.id, False, 'REJECTED',
                                    'NO_KNOWN_RETURN_ROUTE', energy_margin, time_margin)
        # For return-home actions the flight is still needed if the reserve has
        # been breached. Execute if physically modeled energy/time can cover it.
        if action.capability_id == 'uav.return_home':
            if c.return_energy_wh_from_endpoint > self.EPS or c.return_time_s_from_endpoint > self.EPS:
                return SafetyAssessment(action.id, False, 'REJECTED',
                                        'BAD_RETURN_ENDPOINT', energy_margin, time_margin)
            if c.execute_energy_wh > r.remaining_energy_wh + self.EPS:
                return SafetyAssessment(action.id, False, 'REJECTED',
                                        'INSUFFICIENT_ENERGY_FOR_RETURN', energy_margin, time_margin)
            if c.execute_time_s > r.remaining_flight_time_s + self.EPS:
                return SafetyAssessment(action.id, False, 'REJECTED',
                                        'INSUFFICIENT_TIME_FOR_RETURN', energy_margin, time_margin)
            if energy_margin < -self.EPS or time_margin < -self.EPS:
                return SafetyAssessment(action.id, True, 'EMERGENCY_RETURN',
                                        'RETURN_POSSIBLE_BUT_RESERVE_BREACHED', energy_margin, time_margin)
            return SafetyAssessment(action.id, True, 'NORMAL',
                                    'RETURN_FEASIBLE', energy_margin, time_margin)

        if energy_margin < -self.EPS or time_margin < -self.EPS:
            reason = ('ENERGY_RESERVE_VIOLATION' if energy_margin < -self.EPS
                      else 'TIME_RESERVE_VIOLATION')
            return SafetyAssessment(action.id, False, 'REJECTED',
                                    reason, energy_margin, time_margin)
        return SafetyAssessment(action.id, True, 'NORMAL',
                                'RESOURCE_FEASIBLE', energy_margin, time_margin)

    def safe_action_ids(self, snapshot: MissionSnapshot, *, return_route_known: bool) -> set[str]:
        return {a.id for a in snapshot.candidates
                if self.assess(snapshot, a, return_route_known=return_route_known).allowed}

    def recover(self, snapshot: MissionSnapshot, *, return_route_known: bool,
                reason: str) -> RecoveryDecision:
        """Recommend a locally supervised return or model-level landing contingency.

        A landing here TERMINATES THE SIMULATOR, not a claim that the location
        is safe to land a real UAV. Separate landing perception/control is required.
        """
        home = next((a for a in snapshot.candidates
                     if a.capability_id == 'uav.return_home'), None)
        if home is not None:
            assessment = self.assess(snapshot, home, return_route_known=return_route_known)
            if assessment.allowed:
                return RecoveryDecision('RETURN_HOME', reason, home.id)
        return RecoveryDecision('CONTINGENCY_LAND', reason + ';NO_FEASIBLE_KNOWN_RETURN')
