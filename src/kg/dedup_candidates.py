import json, re
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path

def norm(s):
    s = s.lower().strip()
    s = s.replace("%", " percent ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

def load_all():
    recs = []
    for p in ["data/clean/train.json", "data/clean/val.json", "data/clean/test.json"]:
        with open(p) as f:
            recs += json.load(f)
    return recs

def collect(recs):
    S, C, A = set(), set(), set()
    for r in recs:
        for x in r.get("explicit_symptoms", []): S.add(x)
        for x in r.get("implicit_symptoms", []): S.add(x)
        for x in r.get("possible_causes", []): C.add(x)
        for x in r.get("recommended_actions", []): A.add(x)
    return sorted(S), sorted(C), sorted(A)

def cluster(items, thresh=0.9):
    clusters = []
    used = set()

    for i, a in enumerate(items):
        if a in used: 
            continue
        group = [a]
        used.add(a)
        for b in items[i+1:]:
            if b in used:
                continue
            if sim(a, b) >= thresh:
                group.append(b)
                used.add(b)
        if len(group) > 1:
            clusters.append(group)
    return clusters

def main():
    recs = load_all()
    S, C, A = collect(recs)

    for name, items in [("SYMPTOMS", S), ("CAUSES", C), ("ACTIONS", A)]:
        clusters = cluster(items, thresh=0.88)
        print(f"\n{name} duplicate clusters ({len(clusters)} clusters):\n")
        for g in clusters:
            print("  -> ", g)

if __name__ == "__main__":
    main()