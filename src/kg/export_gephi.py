import networkx as nx
from src.dataset.io import load_json
from src.dataset.normalize import normalize_record
from src.kg.build_triples import build_triples_from_record

def main():
    recs = [normalize_record(r) for r in load_json("data/raw/train.json")]

    G = nx.DiGraph()
    for r in recs:
        triples = build_triples_from_record(r)
        for h, rel, t in triples:
            G.add_node(h, kind=h.split("::")[0])
            G.add_node(t, kind=t.split("::")[0])
            # keep relation as edge label
            G.add_edge(h, t, rel=rel)

    out_path = "outputs/kg.graphml"
    nx.write_graphml(G, out_path)
    print("Saved:", out_path)
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

if __name__ == "__main__":
    main()