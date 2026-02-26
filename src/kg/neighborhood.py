from collections import deque
from typing import List, Tuple
from src.kg.graph import KG

def print_neighbors(kg: KG, node: str, max_hops: int = 2, max_print: int = 50):
    if node not in kg.nodes:
        print("Node not in KG:", node)
        return

    print(f"\nNeighborhood for {node} (<= {max_hops} hops)")
    q = deque([(node, 0)])
    seen = {node}
    printed = 0

    while q and printed < max_print:
        n, d = q.popleft()
        for rel, nxt in kg.out.get(n, []):
            print(f"{'  '*d}{n} -[{rel}]-> {nxt}")
            printed += 1
            if d + 1 <= max_hops and nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
            if printed >= max_print:
                break