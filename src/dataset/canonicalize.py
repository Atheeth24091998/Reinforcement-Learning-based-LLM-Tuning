import json
from pathlib import Path

with open("data/alias_map.json") as f:
    ALIAS = json.load(f)

def canon(x):
    return ALIAS.get(x, x)

def rewrite(in_path, out_path):
    with open(in_path) as f:
        recs = json.load(f)

    for r in recs:
        if "explicit_symptoms" in r:
            r["explicit_symptoms"] = [canon(x) for x in r["explicit_symptoms"]]
        if "implicit_symptoms" in r:
            r["implicit_symptoms"] = [canon(x) for x in r["implicit_symptoms"]]
        if "possible_causes" in r:
            r["possible_causes"] = [canon(x) for x in r["possible_causes"]]
        if "recommended_actions" in r:
            r["recommended_actions"] = [canon(x) for x in r["recommended_actions"]]

    with open(out_path, "w") as f:
        json.dump(recs, f, indent=2)

def main():
    rewrite("data/raw/train.json", "data/clean/train.json")
    rewrite("data/raw/val.json", "data/clean/val.json")
    rewrite("data/raw/test.json", "data/clean/test.json")

if __name__ == "__main__":
    main()