"""Simulator -> observation-only visualization frame.

The renderer and recorders must never import simulation or access private terrain.
This adapter intentionally reads `sim.map.known`, NOT `sim.map.terrain`.
"""
from __future__ import annotations
from interfaces.state import MissionSnapshot
from .frame import VisualizationFrame, VisualPoint, VisualEntity, VisualFrontier, VisualSafety


def frame_from_observed(
    snapshot: MissionSnapshot,
    observed_grid: tuple[tuple[int, ...], ...] | list[list[int]],
    *,
    home: VisualPoint,
    executed_path: tuple[VisualPoint, ...] | list[VisualPoint] = (),
    action_id: str | None = None,
    cell_size_m: float = 1.0,
    safety: VisualSafety | None = None,
) -> VisualizationFrame:
    """Generic adapter for either simulator or future ROS 2 occupancy producer.

    `snapshot` itself does not contain a grid; a source must supply *observed*
    occupancy and the actual executed path separately. Do not pass true maps.
    """
    return VisualizationFrame(
        mission_id=snapshot.mission_id,
        frame_id=snapshot.robot_pose.frame_id,
        state_seq=snapshot.state_seq,
        map_version=snapshot.map_version,
        captured_at=snapshot.captured_at,
        occupancy=tuple(tuple(row) for row in observed_grid),
        cell_size_m=cell_size_m,
        drone=VisualPoint(x=snapshot.robot_pose.x, y=snapshot.robot_pose.y),
        home=home,
        path=tuple(executed_path),
        frontiers=tuple(VisualFrontier(
            id=f.id,
            position=VisualPoint(x=f.observation_pose.x, y=f.observation_pose.y),
            information_gain=f.expected_information_gain,
            confidence=f.confidence,
        ) for f in snapshot.frontiers),
        discovered_entities=tuple(VisualEntity(
            id=e.id,
            class_id=e.class_id,
            position=VisualPoint(x=e.position.x, y=e.position.y),
            confidence=e.confidence,
        ) for e in snapshot.entities if e.position is not None),
        goals={g.goal_id: g.status.value for g in snapshot.goals},
        remaining_energy_wh=snapshot.resources.remaining_energy_wh,
        reserve_energy_wh=snapshot.resources.reserve_energy_wh,
        remaining_time_s=snapshot.resources.remaining_flight_time_s,
        reserve_time_s=snapshot.resources.reserve_time_s,
        return_energy_wh=snapshot.resources.current_return_energy_wh,
        return_time_s=snapshot.resources.current_return_time_s,
        selected_action_id=action_id,
        candidate_action_count=len(snapshot.candidates),
        safety=safety,
    )


def frame_from_simulator(sim, snapshot: MissionSnapshot | None = None,
                         *, action_id: str | None = None) -> VisualizationFrame:
    """A single boundary uses simulator state; outputs only publicly observed data."""
    snapshot = snapshot if snapshot is not None else sim.snapshot()
    safety = None
    if hasattr(sim, 'safety_assessments'):
        assessments = sim.safety_assessments(snapshot)
        ev = sim.safety_events[-1] if sim.safety_events else None
        safety = VisualSafety(
            mode=sim.safety_mode,
            return_route_known=sim._known_return_route(),
            feasible_optional_count=sum(a.id in assessments and assessments[a.id].allowed
                                        for a in snapshot.candidates if a.capability_id != 'uav.return_home'),
            rejected_candidate_count=sum(not a.allowed for a in assessments.values()),
            last_status=ev.status if ev else None,
            last_reason=ev.reason if ev else None,
            last_source=ev.source if ev else None,
        )
    return frame_from_observed(
        snapshot,
        sim.map.known,  # deliberately NOT `sim.map.terrain`
        home=VisualPoint(x=sim.home[1]+0.5, y=sim.home[0]+0.5),
        executed_path=[VisualPoint(x=c+0.5, y=r+0.5) for r,c in sim.path_history],
        action_id=action_id,
        safety=safety,
    )
