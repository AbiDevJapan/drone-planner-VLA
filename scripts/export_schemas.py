import json
from pathlib import Path
from interfaces import MissionSpec, MissionSnapshot, PlannerDecision
out = Path(__file__).resolve().parents[1] / "schemas"
out.mkdir(exist_ok=True)
for model in (MissionSpec, MissionSnapshot, PlannerDecision):
    path = out / f"{model.__name__}.schema.json"
    path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
    print(path)
