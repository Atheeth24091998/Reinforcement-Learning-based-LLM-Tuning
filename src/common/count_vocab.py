from src.dataset.io import load_json
from src.dataset.normalize import normalize_record

train_path = "data/raw/train.json"

recs = [normalize_record(r) for r in load_json(train_path)]

problems = set()
causes = set()
actions = set()

for r in recs:
    if "problem" in r and r["problem"]:
        problems.add(r["problem"])

    for c in r.get("causes", []) or []:
        causes.add(c)

    for a in r.get("actions", []) or []:
        actions.add(a)

print("Unique PROBLEMS:", len(problems))
print("Unique CAUSES:", len(causes))
print("Unique ACTIONS:", len(actions))

# (Optional) peek at a few examples
print("\nSample PROBLEMS:", list(problems)[:10])
print("Sample CAUSES:", list(causes)[:10])
print("Sample ACTIONS:", list(actions)[:10])