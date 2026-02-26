from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]

class KG:
    def __init__(self, triples: Iterable[Triple]):
        self.out = defaultdict(list)  # node -> list[(rel, node)]
        self.nodes: Set[str] = set()
        for h, r, t in triples:
            self.out[h].append((r, t))
            self.nodes.add(h); self.nodes.add(t)

    def reachable(self, sources: List[str], target: str, max_hops: int = 3) -> bool:
        srcs = [s for s in sources if s in self.nodes]
        if not srcs:
            return False
        q = deque([(s, 0) for s in srcs])
        seen = set(srcs)
        while q:
            n, d = q.popleft()
            if n == target:
                return True
            if d >= max_hops:
                continue
            for _, nxt in self.out.get(n, []):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return False