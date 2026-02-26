import random
from src.dataset.io import load_json
from src.dataset.normalize import normalize_record
from src.kg.build_triples import build_triples_from_record
from src.kg.graph import KG
from src.kg.reward import compute_reward

def main():
    recs_raw = load_json("data/clean/train.json")
    recs = [normalize_record(r) for r in recs_raw]

    # Build a tiny KG from first N examples
    triples = []
    for r in recs[:200]:
        triples.extend(build_triples_from_record(r))
    kg = KG(triples)

    r = random.choice(recs)
    # Fake model output (in your enforced format)
    fake = f"""PROBLEM: {r['problem']}
CAUSES: {', '.join('CAUSE::'+c for c in r['possible_causes'][:2])}
ACTIONS: {', '.join('ACTION::'+a for a in r['recommended_actions'][:2])}
"""
    rew = compute_reward(kg, r, fake, max_hops=3)
    print("Example symptoms:", r["symptoms"])
    print("Fake output:\n", fake)
    print("Reward:", rew)

if __name__ == "__main__":
    main()