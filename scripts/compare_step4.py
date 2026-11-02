"""Small, exploratory policy comparison (not a statistical research benchmark)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from offboard import OllamaPlanner, OllamaSettings
from scripts.run_step4 import run


def main():
    p = argparse.ArgumentParser(description='Compare original scripted vs Ollama planners on the same random seeds')
    p.add_argument('--mission', choices=['warehouse','exit_sign','thermal_inspection','mapping'], default='warehouse')
    p.add_argument('--seeds', type=str, default='0,1,2')
    p.add_argument('--env-file', type=Path, default=Path('.env'))
    p.add_argument('--output', type=Path, default=Path('outputs/step4_comparison.json'))
    args = p.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not seeds or len(seeds) > 100:
        p.error('Provide between 1 and 100 seeds')
    ollama = OllamaPlanner(OllamaSettings.from_env(str(args.env_file)))
    records = []
    for seed in seeds:
        for method in ('scripted', 'ollama'):
            sim, snapshots, events = run(args.mission, seed=seed, planner_mode=method,
                                         planner=ollama if method == 'ollama' else None, verbose=False)
            goals = {g.goal_id: g.status.value for g in snapshots[-1].goals}
            records.append({
                'mission': args.mission, 'seed': seed, 'method': method,
                'returned_home': sim._home_return_executed,
                'all_goals_verified': all(g == 'VERIFIED' for g in goals.values()),
                'goals': goals,
                'steps': len(events),
                'energy_used_wh': round(sim.scenario.initial_energy_wh - sim.remaining_energy_wh, 3),
                'llm_calls': sum(e['status'] in ('LLM_SELECTED', 'LLM_REJECTED_RECOVERY', 'LLM_FAILED_RECOVERY') for e in events),
                'llm_failures': sum(e['status'] in ('LLM_FAILED_RECOVERY','LLM_REJECTED_RECOVERY') for e in events),
                'prompt_bytes': sum(e['prompt_bytes'] for e in events),
                'llm_latency_s': round(sum(e['latency_s'] or 0 for e in events), 3),
            })
            print(f'seed={seed:3d} {method:9s} home={sim._home_return_executed} '
                  f'complete={records[-1]["all_goals_verified"]} '
                  f'energy={records[-1]["energy_used_wh"]:.1f}Wh steps={len(events)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
