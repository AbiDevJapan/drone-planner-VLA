"""Step 4: safety-gated high-level LLM planner in a simulated mission.

The Ollama network call is made OFFBOARD. All motion and task verification stay local.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from interfaces.state import GoalStatus, PlannerDecision
from offboard import OllamaPlanner, OllamaSettings, PlannerFailure
from simulation import ExplorationSimulator, load_scenario
from visualization.adapters import frame_from_simulator
from visualization.frame import VisualPlanner


def scripted_action(sim, snap):
    """The original Step 3 non-LLM baseline for comparable scenarios."""
    allowed = sim.safety.safe_action_ids(snap, return_route_known=sim._known_return_route())
    pending = any(g.goal_id != 'return' and g.status != GoalStatus.VERIFIED for g in snap.goals)
    if pending:
        for cap in ('uav.inspect', 'uav.explore'):
            actions = [a for a in snap.candidates if a.capability_id == cap and a.id in allowed]
            if actions:
                return actions[0].id
    return None


def run(name='warehouse', *, seed=0, max_steps=50, planner_mode='ollama',
        planner=None, initial_energy_wh=None, initial_time_s=None,
        output_dir: Path | None = None, frame_sink=None, verbose=True):
    spec, scenario = load_scenario(name)
    changes = {}
    if initial_energy_wh is not None: changes['initial_energy_wh'] = initial_energy_wh
    if initial_time_s is not None: changes['initial_time_s'] = initial_time_s
    if changes:
        scenario = type(scenario).model_validate(scenario.model_copy(update=changes).model_dump())
    sim = ExplorationSimulator(spec, scenario, seed=seed)
    if planner_mode == 'ollama' and planner is None:
        planner = OllamaPlanner(OllamaSettings.from_env())
    if planner_mode not in ('ollama', 'scripted'):
        raise ValueError('Unknown planner_mode')

    snapshots = [sim.snapshot()]
    planner_events = []
    def render(snapshot, action_id=None, info=None):
        if frame_sink is not None:
            frame = frame_from_simulator(sim, snapshot, action_id=action_id)
            if info is not None:
                frame = frame.model_copy(update={'planner': VisualPlanner.model_validate(info)})
            frame_sink.on_frame(frame)
    render(snapshots[-1], info={'mode': planner_mode.upper(), 'status': 'READY'})

    for step in range(max_steps):
        before = sim.snapshot()
        if sim._home_return_executed or sim.terminal_contingency:
            break
        optional_allowed = any(a.capability_id != 'uav.return_home' and assess.allowed
                               for a in before.candidates
                               for assess in [sim.safety.assess(before, a, return_route_known=sim._known_return_route())])
        unfinished = any(g.goal_id != 'return' and g.status != GoalStatus.VERIFIED
                         for g in before.goals)
        event = {'step': step + 1, 'state_seq': before.state_seq,
                 'map_version': before.map_version, 'planner': planner_mode,
                 'selected_action_id': None, 'status': 'NOT_REQUESTED',
                 'rationale': '', 'latency_s': None, 'prompt_bytes': 0, 'response_bytes': 0}
        if not optional_allowed or not unfinished:
            event['status'] = 'LOCAL_RETURN'
            event['rationale'] = 'No optional feasible action or all mission work verified'
            after = sim.autonomous_recover('NO_SAFE_OPTIONAL_ACTION_OR_MISSION_COMPLETE')
        elif planner_mode == 'scripted':
            choice = scripted_action(sim, before)
            if choice is None:
                event['status'] = 'LOCAL_RETURN'
                after = sim.autonomous_recover('SCRIPTED_NO_SAFE_ACTION')
            else:
                event.update(selected_action_id=choice, status='SCRIPTED',
                             rationale='Original Step 3 scripted policy')
                decision = PlannerDecision(
                    schema_version=before.schema_version, mission_id=before.mission_id,
                    spec_revision=before.spec_revision, state_seq=before.state_seq,
                    map_version=before.map_version, action_id=choice,
                    rationale=event['rationale'])
                after = sim.handle_planner_decision(decision)
        else:
            assessed = sim.safety_assessments(before)
            render(before, info={'mode': 'OLLAMA', 'status': 'WAITING', 'model': planner.settings.model})
            try:
                result = planner.plan(spec, before, assessed)
                event.update(selected_action_id=result.decision.action_id,
                             status='LLM_SELECTED', rationale=result.decision.rationale,
                             latency_s=round(result.latency_s, 3),
                             prompt_bytes=result.prompt_bytes,
                             response_bytes=result.response_bytes,
                             model=result.model, eval_count=result.eval_count,
                             eval_duration_ns=result.eval_duration_ns)
                # The engine independently revalidates candidate identity/version
                # and runs its full resource safety gate before moving.
                after = sim.handle_planner_decision(result.decision)
                if sim.safety_events and sim.safety_events[-1].status in ('REJECTED', 'AUTONOMOUS_RETURN'):
                    event['status'] = 'LLM_REJECTED_RECOVERY'
            except (PlannerFailure, TimeoutError) as exc:
                event['status'] = 'LLM_FAILED_RECOVERY'
                event['rationale'] = f'{type(exc).__name__}: {str(exc)[:180]}'
                after = sim.autonomous_recover('REMOTE_PLANNER_FAILURE')
        snapshots.append(after)
        planner_events.append(event)
        render(after, sim.action_history[-1] if sim.action_history else None, info={
            'mode': planner_mode.upper(), 'status': event['status'],
            'model': event.get('model', getattr(getattr(planner, 'settings', None), 'model', None)),
            'rationale': event['rationale'],
            'latency_s': event['latency_s'],
            'action_id': event['selected_action_id']})
        if verbose:
            print(f'{step+1:02d} [{event["status"]:23s}] '
                  f'{event["selected_action_id"] or "-":30s} '
                  f'{(event["latency_s"] or 0):5.2f}s | '
                  f'{after.resources.remaining_energy_wh:6.1f} Wh | {sim.safety_mode}')
            if event['rationale']:
                print('    ', event['rationale'][:180])
        if sim._home_return_executed or sim.terminal_contingency:
            break
    if not sim._home_return_executed and not sim.terminal_contingency:
        end = sim.autonomous_recover('DEMO_WATCHDOG_MAX_STEPS')
        snapshots.append(end)
        render(end, sim.action_history[-1] if sim.action_history else None,
               info={'mode': planner_mode.upper(), 'status': 'WATCHDOG_RETURN'})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/'mission_spec.json').write_text(spec.model_dump_json(indent=2) + '\n')
        (output_dir/'snapshots.jsonl').write_text('\n'.join(s.model_dump_json() for s in snapshots) + '\n')
        (output_dir/'planner_events.jsonl').write_text('\n'.join(json.dumps(e) for e in planner_events) + '\n')
        (output_dir/'safety_events.jsonl').write_text('\n'.join(json.dumps(asdict(e)) for e in sim.safety_events) + '\n')
        (output_dir/'final_snapshot.json').write_text(snapshots[-1].model_dump_json(indent=2) + '\n')
    if verbose:
        print('Outcome:', 'HOME' if sim._home_return_executed else 'CONTINGENCY_LAND')
        print('Goals:', {g.goal_id: g.status.value for g in snapshots[-1].goals})
    return sim, snapshots, planner_events


def main():
    parser = argparse.ArgumentParser(description='Step 4 safety-gated Ollama planning demo')
    parser.add_argument('--mission', choices=['warehouse','exit_sign','thermal_inspection','mapping'], default='warehouse')
    parser.add_argument('--planner', choices=['ollama','scripted'], default='ollama')
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--check-ollama', action='store_true', help='List models and confirm selected model is installed, without running a mission')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--initial-energy-wh', type=float)
    parser.add_argument('--initial-time-s', type=float)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--frame-delay', type=float, default=0.35)
    parser.add_argument('--viz-log', type=Path)
    parser.add_argument('--viz-gif', type=Path)
    parser.add_argument('--viz-png', type=Path)
    args = parser.parse_args()
    planner = None
    if args.planner == 'ollama':
        settings = OllamaSettings.from_env(str(args.env_file))
        planner = OllamaPlanner(settings)
        if args.check_ollama:
            try:
                models = planner.check_connection()
            except PlannerFailure as exc:
                parser.error(str(exc))
            print('Server:', settings.base_url)
            print('Models installed:', ', '.join(models) or '(none)')
            print('Selected:', settings.model)
            if settings.model not in models:
                parser.error(f'Model {settings.model!r} not installed on Ollama server. Run: ollama pull {settings.model}')
            return
    elif args.check_ollama:
        parser.error('--check-ollama requires --planner ollama')
    from visualization.sinks import JsonlFrameRecorder, GifFrameRecorder, LiveMatplotlibSink, FrameSinkGroup
    sinks = []
    if args.live: sinks.append(LiveMatplotlibSink(args.frame_delay))
    if args.viz_log: sinks.append(JsonlFrameRecorder(args.viz_log))
    if args.viz_gif: sinks.append(GifFrameRecorder(args.viz_gif))
    sink = FrameSinkGroup(*sinks) if sinks else None
    sim, states = None, []
    try:
        sim, states, events = run(args.mission, seed=args.seed, max_steps=args.max_steps,
                                  planner_mode=args.planner, planner=planner,
                                  initial_energy_wh=args.initial_energy_wh,
                                  initial_time_s=args.initial_time_s,
                                  output_dir=args.output_dir, frame_sink=sink)
    finally:
        if sink: sink.close()
    if args.viz_png:
        from visualization.matplotlib_view import DashboardRenderer
        renderer = DashboardRenderer()
        try:
            frame = frame_from_simulator(sim, states[-1], action_id=sim.action_history[-1] if sim.action_history else None)
            if events:
                e = events[-1]
                frame = frame.model_copy(update={'planner': VisualPlanner(
                    mode=args.planner.upper(), status=e['status'], rationale=e['rationale'],
                    latency_s=e['latency_s'], action_id=e['selected_action_id'],
                    model=e.get('model', getattr(getattr(planner, 'settings', None), 'model', None)))})
            renderer.save_png(frame, args.viz_png)
        finally:
            renderer.close()
        print('Dashboard saved:', args.viz_png)


if __name__ == '__main__':
    main()
