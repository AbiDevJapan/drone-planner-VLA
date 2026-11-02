import json
from pathlib import Path
from interfaces import MissionSpec, MissionSnapshot, validate_spec_snapshot
root = Path(__file__).resolve().parents[1] / "examples"
for name in ("warehouse", "exit_sign", "thermal_inspection"):
    spec = MissionSpec.model_validate_json((root / f"{name}_spec.json").read_text())
    snap = MissionSnapshot.model_validate_json((root / f"{name}_state.json").read_text())
    validate_spec_snapshot(spec, snap)
    print(f"{name}: OK; verified goals {[g.goal_id for g in snap.goals if g.status == 'VERIFIED']}")
