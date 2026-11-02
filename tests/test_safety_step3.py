"""Step 3: regressions for energy/time safety, return fallback and observability."""
from dataclasses import asdict
import json
import pytest
from interfaces.state import PlannerDecision, GoalStatus
from interfaces.validation import StaleDecisionError, UnknownActionError
from safety import SafetyViolation
from simulation import ExplorationSimulator, load_scenario
from scripts.run_step3 import run, make_decision
from visualization.adapters import frame_from_simulator
from visualization.frame import VisualizationFrame


def sim_with(name='warehouse', *, energy=None, time=None, seed=0):
    spec, scenario = load_scenario(name)
    change = {}
    if energy is not None: change['initial_energy_wh'] = energy
    if time is not None: change['initial_time_s'] = time
    if change:
        scenario = type(scenario).model_validate(scenario.model_copy(update=change).model_dump())
    return ExplorationSimulator(spec, scenario, seed=seed)


def first_explore(snap):
    return next(a for a in snap.candidates if a.capability_id == 'uav.explore')


def test_gate_rejects_energy_insufficient_to_execute_and_return_with_reserve():
    sim = sim_with(energy=11.0)
    snap = sim.snapshot()
    action = first_explore(snap)
    check = sim.safety.assess(snap, action, return_route_known=True)
    assert not check.allowed and check.reason == 'ENERGY_RESERVE_VIOLATION'
    pose, energy, seq, action_history = sim.position, sim.remaining_energy_wh, sim.state_seq, list(sim.action_history)
    with pytest.raises(SafetyViolation, match='ENERGY_RESERVE_VIOLATION'):
        sim.execute(make_decision(snap, action.id))
    assert (sim.position, sim.remaining_energy_wh, sim.state_seq, sim.action_history) == (pose, energy, seq, action_history)
    assert sim.safety_events[-1].status == 'REJECTED'


def test_gate_rejects_time_budget_even_when_battery_full():
    sim = sim_with(time=96.0)
    snap = sim.snapshot()
    action = first_explore(snap)
    report = sim.safety.assess(snap, action, return_route_known=True)
    assert report.reason == 'TIME_RESERVE_VIOLATION'
    with pytest.raises(SafetyViolation, match='TIME_RESERVE_VIOLATION'):
        sim.execute(make_decision(snap, action.id))


def test_cost_budget_uses_candidate_endpoint_not_current_position():
    sim = sim_with(energy=20.0)
    snap = sim.snapshot()
    explore = max((a for a in snap.candidates if a.capability_id=='uav.explore'),
                  key=lambda a: a.cost.return_energy_wh_from_endpoint)
    current_only = explore.cost.execute_energy_wh + snap.resources.current_return_energy_wh + snap.resources.reserve_energy_wh
    from_endpoint = (explore.cost.execute_energy_wh + explore.cost.return_energy_wh_from_endpoint
                     + snap.resources.reserve_energy_wh)
    assert from_endpoint > current_only
    assert explore.cost.return_energy_wh_from_endpoint > snap.resources.current_return_energy_wh
    assert sim.safety.assess(snap, explore, return_route_known=True).energy_margin_wh == pytest.approx(20-from_endpoint)


def test_return_allowed_even_if_reserve_already_breached():
    sim = sim_with(energy=11.0)
    sim.position = (sim.home[0]+1, sim.home[1])  # observed free at initial scan
    assert sim.map.path(sim.position,sim.home) is not None
    sim.remaining_energy_wh = 0.5  # physically enough for 1 grid-cell return, not for reserve
    snap = sim.snapshot()
    home = next(a for a in snap.candidates if a.capability_id=='uav.return_home')
    a = sim.safety.assess(snap,home,return_route_known=True)
    assert a.allowed and a.mode == 'EMERGENCY_RETURN'
    after = sim.execute(make_decision(snap,home.id))
    assert sim.position == sim.home
    assert after.resources.remaining_energy_wh == pytest.approx(0.1)
    assert any(e.status == 'EMERGENCY_RETURN' for e in sim.safety_events)
    assert next(g for g in after.goals if g.goal_id=='return').status == GoalStatus.VERIFIED


def test_auto_return_under_low_budget_leaves_nonreturn_goals_incomplete():
    sim, states = run('warehouse', initial_energy_wh=14.0, verbose=False)
    assert sim.position == sim.home
    assert states[-1].resources.remaining_energy_wh >= 10 - 1e-8
    assert states[-1].resources.remaining_flight_time_s >= 90 - 1e-8
    assert next(g for g in states[-1].goals if g.goal_id=='return').status == GoalStatus.VERIFIED
    assert next(g for g in states[-1].goals if g.goal_id=='locate').status != GoalStatus.VERIFIED
    assert any(e.source == 'SUPERVISOR' for e in sim.safety_events)


def test_remote_timeout_returns_locally():
    sim, states = run('warehouse', fault='timeout', fault_step=2, verbose=False)
    assert sim.position == sim.home
    assert any(e.status == 'AUTONOMOUS_RETURN' for e in sim.safety_events)
    assert sim.safety_events[-1].source == 'SUPERVISOR'
    assert next(g for g in states[-1].goals if g.goal_id=='return').status == GoalStatus.VERIFIED


def test_invalid_remote_action_never_executes_and_falls_back():
    sim, states = run('warehouse', fault='invalid_action', fault_step=2, verbose=False)
    assert sim.position == sim.home
    assert 'invented_remote_action' not in sim.action_history
    assert any(e.status == 'REJECTED' for e in sim.safety_events)
    assert sim.safety_events[-1].source == 'SUPERVISOR'


def test_stale_decision_cannot_execute_after_telemetry_loss():
    sim = sim_with()
    old = sim.snapshot()
    decision = make_decision(old, first_explore(old).id)
    sim.inject_resource_loss(energy_wh=60.0)
    with pytest.raises(StaleDecisionError): sim.execute(decision)
    assert sim.state_seq == 1
    assert sim.position == sim.home


def test_no_route_home_causes_terminal_contingency_not_false_at_home():
    sim = sim_with()
    snap = sim.snapshot()
    sim.execute(make_decision(snap,first_explore(snap).id))
    assert sim.position != sim.home
    # Invalidate every known cell adjacent to home; a locally measured obstruction.
    for r,c in sim.map.neighbors(sim.home):
        if sim.map.known[r][c] == 0:
            sim.map.known[r][c] = 1
    sim.map_version += 1
    assert sim.map.path(sim.position, sim.home) is None
    result = sim.autonomous_recover('RETURN_ROUTE_BLOCKED')
    assert sim.terminal_contingency
    assert sim.safety_mode == 'CONTINGENCY_LAND'
    assert sim.position != sim.home
    assert not any(o.property == 'at_home' for o in result.observations)
    assert next(g for g in result.goals if g.goal_id=='return').status != GoalStatus.VERIFIED
    with pytest.raises(ValueError): sim.select_action('return_home')


def test_unaffordable_return_stops_in_contingency_without_claiming_success():
    sim = sim_with()
    snap = sim.snapshot()
    sim.execute(make_decision(snap, first_explore(snap).id))
    sim.inject_resource_loss(energy_wh=119.0)
    end = sim.autonomous_recover('BATTERY_EMERGENCY')
    assert sim.terminal_contingency
    assert sim.position != sim.home
    assert next(g for g in end.goals if g.goal_id=='return').status != GoalStatus.VERIFIED


def test_resources_dropped_mid_mission_invalidate_decisions_and_trigger_return():
    sim, states = run('warehouse', fault='battery_drop', fault_step=2,
                      energy_loss_wh=100.0, verbose=False)
    assert sim._home_return_executed or sim.terminal_contingency
    assert any(e.status=='TELEMETRY_UPDATE' for e in sim.safety_events)
    assert states[-1].resources.remaining_energy_wh >= 0


def test_safety_overlay_replays_and_contains_no_hidden_truth():
    sim = sim_with(energy=11.0)
    snap = sim.snapshot()
    action = first_explore(snap)
    with pytest.raises(SafetyViolation): sim.execute(make_decision(snap,action.id))
    frame = frame_from_simulator(sim)
    assert frame.safety.mode == 'NORMAL'
    assert frame.safety.rejected_candidate_count > 0
    assert frame.safety.last_status == 'REJECTED'
    assert frame.safety.last_reason == 'ENERGY_RESERVE_VIOLATION'
    assert 'ext_2' not in frame.model_dump_json()
    assert VisualizationFrame.model_validate_json(frame.model_dump_json()) == frame


@pytest.mark.parametrize('name', ['warehouse','exit_sign','thermal_inspection','mapping'])
def test_normal_step3_mission_can_finish(name):
    sim, states = run(name, verbose=False, max_steps=50)
    assert sim._home_return_executed
    assert all(g.status == GoalStatus.VERIFIED for g in states[-1].goals)
    assert not sim.terminal_contingency
    assert all(e.status in ('EXECUTED','AUTONOMOUS_RETURN','EMERGENCY_RETURN') for e in sim.safety_events)


def test_offboard_boundary_recovers_when_planner_times_out():
    sim = sim_with()
    sim.execute(make_decision(sim.snapshot(),first_explore(sim.snapshot()).id))
    result = sim.handle_planner_decision(None)
    assert sim.position == sim.home
    assert any(e.source == 'SUPERVISOR' for e in sim.safety_events)
    assert next(g for g in result.goals if g.goal_id=='return').status == GoalStatus.VERIFIED


def test_offboard_boundary_recovers_from_unknown_or_stale_id():
    for mode in ('unknown', 'stale'):
        sim = sim_with()
        before = sim.snapshot()
        old = make_decision(before, first_explore(before).id)
        sim.execute(old)
        if mode == 'unknown':
            snap = sim.snapshot()
            bad = make_decision(snap, 'invalid_action')
        else:
            bad = old
        final = sim.handle_planner_decision(bad)
        assert sim.position == sim.home
        assert any(e.status=='REJECTED' for e in sim.safety_events)
        assert final.goals[-1].status == GoalStatus.VERIFIED
