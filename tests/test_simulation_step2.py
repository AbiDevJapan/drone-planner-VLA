"""Step 2 acceptance tests: observed-only mapping, frontiers, contracts, determinism."""
from copy import deepcopy
import pytest

from interfaces import PlannerDecision, validate_spec_snapshot
from interfaces.state import GoalStatus
from interfaces.validation import StaleDecisionError, UnknownActionError
from simulation import GridMap, Scenario, ExplorationSimulator, load_scenario
from simulation.grid import UNKNOWN, FREE, OCCUPIED
from simulation.engine import ExecutionError
from scripts.run_step2 import run, scripted_choice


def start(name='warehouse', seed=0):
    return ExplorationSimulator(*load_scenario(name), seed=seed)


def test_sensor_does_not_see_through_obstacle():
    grid = GridMap(('#######','#H.#..#','#######'), (1,1), sensor_radius=6)
    grid.reveal((1,1))
    assert grid.known[1][3] == OCCUPIED
    assert grid.known[1][4] == UNKNOWN
    assert grid.known[1][2] == FREE


def test_known_free_routing_never_uses_unknown_or_wall():
    grid = GridMap(('#########','#H..#...#','#.......#','#########'), (1,1), sensor_radius=2)
    grid.reveal((1,1))
    assert grid.path((1,1),(1,7)) is None  # unknown beyond sensor range
    assert grid.path((1,1),(1,4)) is None  # wall
    assert grid.path((1,1),(1,2)) == [(1,1),(1,2)]
    assert all(grid.known[r][c] == FREE for r,c in grid.path((1,1),(1,2)))


def test_targets_are_not_leaked_in_initial_state():
    sim=start()
    snap=sim.snapshot()
    assert not snap.entities and not snap.observations
    assert not any(a.capability_id=='uav.inspect' for a in snap.candidates)
    blob=snap.model_dump_json()
    assert 'fire_extinguisher' not in blob
    assert 'BLOCKED' not in blob
    assert 'ext_2' not in blob
    assert sim._hidden_objects[0].id == 'ext_2'  # simulator knows what planner cannot


def test_frontier_viewpoints_reachable_and_information_positive():
    sim=start()
    snap=sim.snapshot()
    assert snap.frontiers
    for frontier in snap.frontiers:
        dest=sim._from_pos(frontier.observation_pose)
        path=sim.map.path(sim.position,dest)
        assert path is not None
        assert all(sim.map.known[r][c] == FREE for r,c in path)
        assert any(sim.map.known[r][c] == UNKNOWN for r,c in sim.map.neighbors(dest))
        assert frontier.expected_information_gain > 0
        assert frontier.map_version == snap.map_version
    for a in snap.candidates:
        if a.capability_id == 'uav.explore':
            assert a.target.ref in {f.id for f in snap.frontiers}


def test_frontier_and_map_evolve_after_explore():
    sim=start()
    before=sim.snapshot()
    a=next(a for a in before.candidates if a.capability_id=='uav.explore')
    after=sim.select_action(a.id)
    assert after.map_version > before.map_version
    assert after.state_seq == before.state_seq + 1
    assert sim.map.coverage() > 0.1
    assert [a.id for a in after.candidates] != [a.id for a in before.candidates]
    validate_spec_snapshot(sim.spec,after)


def test_action_cost_return_is_from_predicted_endpoint():
    sim=start()
    snap=sim.snapshot()
    candidate=next(a for a in snap.candidates if a.capability_id=='uav.explore')
    endpoint=next(sim._from_pos(f.observation_pose) for f in snap.frontiers if f.id==candidate.target.ref)
    return_path=sim.map.path(endpoint,sim.home)
    assert candidate.cost.return_energy_wh_from_endpoint == pytest.approx(0.4*(len(return_path)-1))
    assert candidate.cost.return_time_s_from_endpoint == pytest.approx(2.0*(len(return_path)-1))
    assert sim.position == sim.home
    assert snap.resources.current_return_energy_wh == 0.0


def test_stale_decision_is_rejected_after_map_update():
    sim=start()
    snap=sim.snapshot()
    a=next(a for a in snap.candidates if a.capability_id=='uav.explore')
    stale=PlannerDecision(schema_version='2.0',mission_id=snap.mission_id,
                          spec_revision=snap.spec_revision,state_seq=snap.state_seq,
                          map_version=snap.map_version,action_id=a.id,rationale='old')
    sim.execute(stale)
    with pytest.raises(StaleDecisionError):
        sim.execute(stale)


def test_unknown_action_rejected():
    sim=start()
    snap=sim.snapshot()
    decision=PlannerDecision(schema_version='2.0',mission_id=snap.mission_id,
                             spec_revision=snap.spec_revision,state_seq=snap.state_seq,
                             map_version=snap.map_version,action_id='invented_action',rationale='no')
    with pytest.raises(UnknownActionError):
        sim.execute(decision)


def test_direct_route_to_hidden_unmapped_target_not_possible():
    sim=start()
    target=sim._hidden_objects[0].cell
    assert sim.map.path(sim.home,target) is None
    assert all(a.target.ref != 'ext_2' for a in sim.snapshot().candidates)


@pytest.mark.parametrize('name', ['warehouse','exit_sign','thermal_inspection','mapping'])
def test_step2_demo_completes_mission_with_valid_snapshots(name):
    sim, states=run(name,seed=0,max_steps=40,verbose=False)
    assert len(states)>2
    for state in states:
        validate_spec_snapshot(sim.spec,state)
    assert sim.position == sim.home
    assert all(goal.status==GoalStatus.VERIFIED for goal in states[-1].goals)
    assert states[-1].resources.remaining_energy_wh < states[0].resources.remaining_energy_wh
    if name == 'warehouse':
        assert 'ext_2' in {e.id for e in states[-1].entities}
        assert any(o.property == 'access_state' and o.value == 'BLOCKED'
                   for o in states[-1].observations)
    if name == 'exit_sign':
        assert any(o.property == 'recognized_text' and o.value == 'EXIT'
                   for o in states[-1].observations)
    if name == 'thermal_inspection':
        assert any(o.property == 'temperature_c' and o.value == 68.5
                   for o in states[-1].observations)
    if name == 'mapping':
        assert any(o.property == 'coverage_fraction' and o.value >= 0.60
                   for o in states[-1].observations)


def test_same_seed_identical_trace_and_different_seeds_can_change_placement():
    _, a=run('warehouse',seed=12, max_steps=40,verbose=False)
    _, b=run('warehouse',seed=12, max_steps=40,verbose=False)
    assert [s.model_dump_json() for s in a] == [s.model_dump_json() for s in b]
    placements={start(seed=k)._hidden_objects[0].cell for k in range(10)}
    assert len(placements)>1


def test_target_observation_requires_line_of_sight():
    spec,scenario=load_scenario('warehouse')
    data=scenario.model_dump()
    data['terrain']=['#########','#H..#...#','#.......#','#########']
    data['objects']=[{'id':'ext_2','class_id':'fire_extinguisher','row':1,'col':6,
                      'properties':{'access_state':'BLOCKED'},
                      'producers':{'access_state':'sim_geometry'}}]
    data['sensor_radius']=6
    data['detection_radius']=8.0
    sim=ExplorationSimulator(spec, Scenario.model_validate(data))
    assert sim.map.clear_line_of_sight(sim.home,(1,6)) is False
    assert sim.snapshot().entities == []
    assert all(o.property != 'detected' for o in sim.snapshot().observations)


def test_inspection_only_generates_properties_after_inspection_action():
    sim=start()
    for _ in range(30):
        snap=sim.snapshot()
        if snap.entities:
            assert not any(o.property == 'access_state' for o in snap.observations)
            inspect=next(a for a in snap.candidates if a.capability_id=='uav.inspect')
            observed=sim.select_action(inspect.id)
            assert any(o.property == 'access_state' for o in observed.observations)
            assert next(g for g in observed.goals if g.goal_id=='access').status==GoalStatus.VERIFIED
            return
        sim.select_action(next(a.id for a in snap.candidates if a.capability_id=='uav.explore'))
    pytest.fail('scripted sequence never discovered target')


def test_mapping_coverage_is_reachable_observation_not_hidden_truth():
    sim=start('mapping')
    initial=sim.snapshot()
    assert next(g for g in initial.goals if g.goal_id=='survey').status==GoalStatus.PENDING
    assert initial.observations[0].value==pytest.approx(sim.map.coverage())
    assert 0.0<=initial.observations[0].value<0.6


def test_optional_matplotlib_map_export(tmp_path):
    pytest.importorskip('matplotlib')
    from simulation.visualize import save_discovered_map
    sim=start()
    output=tmp_path/'map.png'
    save_discovered_map(sim,output)
    assert output.exists() and output.stat().st_size>2000
