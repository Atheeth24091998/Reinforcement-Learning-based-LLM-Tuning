from collections import Counter
from typing import Iterable, Tuple, List
from src.dataset.io import load_json
from src.dataset.normalize import normalize_record
from src.kg.build_triples import build_triples_from_record
from src.kg.graph import KG

Triple = Tuple[str, str, str]

def build_kg_from_json(path: str) -> tuple[KG, List[Triple]]:
    recs = [normalize_record(r) for r in load_json(path)]
    triples: List[Triple] = []
    for r in recs:
        triples.extend(build_triples_from_record(r))
    return KG(triples), triples

def main():
    kg, triples = build_kg_from_json("data/raw/train.json")

    # degree
    out_deg = Counter()
    for h, _, _ in triples:
        out_deg[h] += 1

    print("Nodes:", len(kg.nodes))
    print("Triples:", len(triples))
    print("\nTop 20 nodes by outgoing degree:")
    for n, d in out_deg.most_common(20):
        print(f"{d:>4}  {n}")

if __name__ == "__main__":
    main()