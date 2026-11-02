"""Step 2 demonstration. The greedy selection policy is NOT the proposed research planner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from simulation import load_scenario, ExplorationSimulator
from interfaces.state import GoalStatus


def scripted_choice(snap):
    """Fixed integration demo: inspect known target, otherwise explore, then return."""
    remaining = any(g.goal_id != 'return' and g.status != GoalStatus.VERIFIED for g in snap.goals)
    if remaining:
        inspect = [a for a in snap.candidates if a.capability_id == 'uav.inspect']
        if inspect:
            return inspect[0].id
        explore = [a for a in snap.candidates if a.capability_id == 'uav.explore']
        if explore:
            # Frontiers already ranked by geometric gain and distance in the simulator.
            return explore[0].id
    return 'return_home'


def run(name: str, seed: int = 0, max_steps: int = 35, verbose: bool = True,
        output_dir: Path | None = None, plot_path: Path | None = None,
        frame_sink=None):
    spec, scenario = load_scenario(name)
    sim = ExplorationSimulator(spec, scenario, seed=seed)
    start_snap = sim.snapshot()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'mission_spec.json').write_text(spec.model_dump_json(indent=2) + '\n')
        (output_dir / 'initial_snapshot.json').write_text(start_snap.model_dump_json(indent=2) + '\n')
    if verbose:
        print(f'=== {name} (seed={seed}) ===')
        print('Legend: D drone, H home, F frontier, T discovered target, # occupied, . free, ? unknown')
        print('INITIAL KNOWN MAP:')
        print(sim.known_ascii())
    states = [start_snap]
    if frame_sink is not None:
        from visualization.adapters import frame_from_simulator
        frame_sink.on_frame(frame_from_simulator(sim, start_snap))
    for i in range(max_steps):
        before = sim.snapshot()
        choice = scripted_choice(before)
        if choice not in {a.id for a in before.candidates}:
            if verbose: print('No executable Step 2 candidate remaining.')
            break
        snap = sim.select_action(choice)
        states.append(snap)
        if frame_sink is not None:
            frame_sink.on_frame(frame_from_simulator(sim, snap, action_id=choice))
        if verbose:
            goal_text = ', '.join(f'{g.goal_id}:{g.status.value}' for g in snap.goals)
            print(f'{i+1:02d}. {choice:32s} map_v={snap.map_version:02d} '
                  f'known={sim.map.coverage():.1%} energy={snap.resources.remaining_energy_wh:.1f}Wh '
                  f'goals=({goal_text})')
        if choice == 'return_home':
            break
    final = states[-1]
    if verbose:
        print('\nFINAL KNOWN MAP:')
        print(sim.known_ascii())
        print('Final goal states:', {g.goal_id: g.status.value for g in final.goals})
        print('Hidden ground truth was not exported to the planner snapshots.')
    if output_dir:
        (output_dir / 'final_snapshot.json').write_text(final.model_dump_json(indent=2) + '\n')
        (output_dir / 'snapshots.jsonl').write_text(
            '\n'.join(s.model_dump_json() for s in states) + '\n')
        (output_dir / 'known_map.txt').write_text(sim.known_ascii() + '\n')
        (output_dir / 'decisions.json').write_text(json.dumps(sim.action_history,indent=2) + '\n')
        if verbose:
            print(f'Planner-visible data written to {output_dir.resolve()}')
    if plot_path:
        from simulation.visualize import save_discovered_map
        save_discovered_map(sim, plot_path)
        if verbose: print(f'Observed-map plot written to {plot_path.resolve()}')
    return sim, states


def main():
    parser = argparse.ArgumentParser(description='Step 2: lightweight occupancy exploration with generalized mission-state output')
    parser.add_argument('--mission', choices=['warehouse','exit_sign','thermal_inspection','mapping'], default='warehouse')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=35)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plot', type=Path, help='Save original observed map PNG')
    parser.add_argument('--live', action='store_true',
                        help='Open an updating Matplotlib dashboard (desktop GUI required)')
    parser.add_argument('--frame-delay', type=float, default=0.30,
                        help='Seconds to display each simulation update in live mode')
    parser.add_argument('--viz-gif', type=Path,
                        help='Record all visualization updates to an animated GIF')
    parser.add_argument('--viz-log', type=Path,
                        help='Record replayable observed-only visualization frames as JSONL')
    parser.add_argument('--viz-png', type=Path,
                        help='Save a final dashboard PNG (different from --plot)')
    parser.add_argument('--viz-fps', type=float, default=2.0,
                        help='Playback FPS for --viz-gif')
    args = parser.parse_args()
    from visualization.sinks import (
        JsonlFrameRecorder, GifFrameRecorder, LiveMatplotlibSink, FrameSinkGroup)
    sinks = []
    if args.live:
        sinks.append(LiveMatplotlibSink(args.frame_delay))
    if args.viz_log:
        sinks.append(JsonlFrameRecorder(args.viz_log))
    if args.viz_gif:
        sinks.append(GifFrameRecorder(args.viz_gif, args.viz_fps))
    sink = FrameSinkGroup(*sinks) if sinks else None
    try:
        sim, states = run(args.mission, seed=args.seed, max_steps=args.max_steps,
                          output_dir=args.output_dir, plot_path=args.plot,
                          frame_sink=sink)
    finally:
        if sink:
            sink.close()
    if args.viz_png:
        from visualization.adapters import frame_from_simulator
        from visualization.matplotlib_view import DashboardRenderer
        renderer = DashboardRenderer()
        try:
            renderer.save_png(frame_from_simulator(
                sim, states[-1], action_id=sim.action_history[-1] if sim.action_history else None),
                args.viz_png)
        finally:
            renderer.close()
        print(f'Final dashboard saved to {args.viz_png.resolve()}')


if __name__ == '__main__':
    main()
