"""Step 3 demo: deterministic safety-gated exploration and failure recovery.

No real flight control. Uses the same v2 mission-state schema as Steps 1-2.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from interfaces.state import GoalStatus, PlannerDecision
from safety import SafetyViolation
from simulation import ExplorationSimulator, load_scenario
from visualization.adapters import frame_from_simulator


def make_decision(snapshot, action_id, rationale='Step 3 deterministic policy'):
    return PlannerDecision(
        schema_version=snapshot.schema_version, mission_id=snapshot.mission_id,
        spec_revision=snapshot.spec_revision, state_seq=snapshot.state_seq,
        map_version=snapshot.map_version, action_id=action_id, rationale=rationale)


def choose_safe_action(sim: ExplorationSimulator, snapshot):
    """This is not a mission-aware research planner, just a safe smoke-test policy."""
    allowed = sim.safety.safe_action_ids(snapshot, return_route_known=sim._known_return_route())
    pending = any(g.goal_id != 'return' and g.status != GoalStatus.VERIFIED
                  for g in snapshot.goals)
    if pending:
        for capability in ('uav.inspect', 'uav.explore'):
            matching = [a for a in snapshot.candidates
                        if a.capability_id == capability and a.id in allowed]
            if matching:
                return matching[0].id
    return None  # Supervisor independently decides return or contingency.


def run(name='warehouse', *, seed=0, max_steps=50, initial_energy_wh=None,
        initial_time_s=None, fault='none', fault_step=2, energy_loss_wh=0.0,
        output_dir: Path | None = None, frame_sink=None, verbose=True):
    spec, scenario = load_scenario(name)
    changes = {}
    if initial_energy_wh is not None:
        changes['initial_energy_wh'] = initial_energy_wh
    if initial_time_s is not None:
        changes['initial_time_s'] = initial_time_s
    if changes:
        scenario = scenario.model_copy(update=changes)
        # Revalidate overrides, not merely a low-level model_copy.
        scenario = type(scenario).model_validate(scenario.model_dump())
    sim = ExplorationSimulator(spec, scenario, seed=seed)
    states = [sim.snapshot()]
    def render(snap, action_id=None):
        if frame_sink is not None:
            frame_sink.on_frame(frame_from_simulator(sim, snap, action_id=action_id))
    render(states[-1])
    fault_injected = False
    for step in range(max_steps):
        before = sim.snapshot()
        if sim._home_return_executed or sim.terminal_contingency:
            break
        if not fault_injected and step == fault_step and fault != 'none':
            fault_injected = True
            if fault == 'battery_drop':
                before = sim.inject_resource_loss(energy_wh=energy_loss_wh)
                states.append(before)
                render(before)
            elif fault == 'timeout':
                after = sim.autonomous_recover('SIMULATED_PLANNER_TIMEOUT')
                states.append(after)
                render(after, sim.action_history[-1] if sim.action_history else None)
                break
            elif fault == 'invalid_action':
                try:
                    sim.execute(make_decision(before, 'invented_remote_action'))
                except ValueError as exc:
                    sim._event('invented_remote_action', 'REJECTED', type(exc).__name__, 'PLANNER')
                    render(before, 'invented_remote_action')
                    after = sim.autonomous_recover('REMOTE_INVALID_ACTION')
                    states.append(after)
                    render(after, sim.action_history[-1] if sim.action_history else None)
                    break
                raise AssertionError('Invalid remote action unexpectedly accepted')
        before = sim.snapshot()
        choice = choose_safe_action(sim, before)
        if choice is None:
            after = sim.autonomous_recover('NO_SAFE_OPTIONAL_ACTION_OR_MISSION_COMPLETE')
        else:
            try:
                after = sim.execute(make_decision(before, choice))
            except SafetyViolation:
                render(before, choice)  # Show rejection before autonomous recovery.
                after = sim.autonomous_recover('SAFETY_GATE_REJECTED_ACTION')
        states.append(after)
        render(after, sim.action_history[-1] if sim.action_history else None)
        if verbose:
            event = sim.safety_events[-1]
            print(f'{step + 1:02d} {event.status:18s} '
                  f'{event.action_id or "-":30s} '
                  f'{after.resources.remaining_energy_wh:6.1f} Wh | '
                  f'{after.resources.remaining_flight_time_s:6.0f} s | '
                  f'{sim.safety_mode} | {event.reason}')
        if sim._home_return_executed or sim.terminal_contingency:
            break
    # A demo watchdog is also a safety fallback: do not leave a simulated
    # mission indefinitely airborne after a bounded decision loop expires.
    if not sim._home_return_executed and not sim.terminal_contingency:
        final = sim.autonomous_recover('DEMO_WATCHDOG_MAX_STEPS')
        states.append(final)
        render(final, sim.action_history[-1] if sim.action_history else None)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'mission_spec.json').write_text(spec.model_dump_json(indent=2)+'\n')
        (output_dir / 'snapshots.jsonl').write_text('\n'.join(s.model_dump_json() for s in states)+'\n')
        (output_dir / 'safety_events.jsonl').write_text('\n'.join(json.dumps(asdict(e)) for e in sim.safety_events)+'\n')
        (output_dir / 'final_snapshot.json').write_text(states[-1].model_dump_json(indent=2)+'\n')
    if verbose:
        print('Outcome:', 'HOME' if sim._home_return_executed else
              'CONTINGENCY_LAND (simulation only)' if sim.terminal_contingency else 'MAX_STEPS')
        print('Goals:', {g.goal_id: g.status.value for g in states[-1].goals})
    return sim, states


def main():
    parser = argparse.ArgumentParser(description='Step 3 resource safety supervisor demo')
    parser.add_argument('--mission', choices=['warehouse','exit_sign','thermal_inspection','mapping'],default='warehouse')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-steps',type=int,default=50)
    parser.add_argument('--initial-energy-wh',type=float)
    parser.add_argument('--initial-time-s',type=float)
    parser.add_argument('--fault',choices=['none','timeout','invalid_action','battery_drop'],default='none')
    parser.add_argument('--fault-step',type=int,default=2)
    parser.add_argument('--energy-loss-wh',type=float,default=30.0)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--frame-delay',type=float,default=0.35)
    parser.add_argument('--viz-log',type=Path)
    parser.add_argument('--viz-gif',type=Path)
    parser.add_argument('--viz-png',type=Path)
    args = parser.parse_args()
    from visualization.sinks import JsonlFrameRecorder, GifFrameRecorder, LiveMatplotlibSink, FrameSinkGroup
    sinks = []
    if args.live: sinks.append(LiveMatplotlibSink(args.frame_delay))
    if args.viz_log: sinks.append(JsonlFrameRecorder(args.viz_log))
    if args.viz_gif: sinks.append(GifFrameRecorder(args.viz_gif))
    sink = FrameSinkGroup(*sinks) if sinks else None
    try:
        sim, states = run(args.mission,seed=args.seed,max_steps=args.max_steps,
                          initial_energy_wh=args.initial_energy_wh,
                          initial_time_s=args.initial_time_s,
                          fault=args.fault,fault_step=args.fault_step,
                          energy_loss_wh=args.energy_loss_wh,
                          output_dir=args.output_dir,frame_sink=sink)
    finally:
        if sink: sink.close()
    if args.viz_png:
        from visualization.matplotlib_view import DashboardRenderer
        renderer = DashboardRenderer()
        try:
            renderer.save_png(frame_from_simulator(sim, states[-1],
                              action_id=sim.action_history[-1] if sim.action_history else None),args.viz_png)
        finally:
            renderer.close()
        print('Dashboard saved:',args.viz_png)


if __name__=='__main__':
    main()
