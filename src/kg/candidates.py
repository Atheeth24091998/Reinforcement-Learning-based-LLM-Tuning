from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]

class CandidateIndex:
    """
    Fast lookup index:
      symptom -> problems
      problem -> causes/actions
    Assumes your triples use:
      SYMPTOM::<s> indicates PROBLEM::<p>
      PROBLEM::<p> caused_by CAUSE::<c>
      PROBLEM::<p> mitigated_by ACTION::<a>
    """
    def __init__(self, triples: Iterable[Triple]):
        self.symptom_to_problems: Dict[str, Set[str]] = defaultdict(set)
        self.problem_to_causes: Dict[str, Set[str]] = defaultdict(set)
        self.problem_to_actions: Dict[str, Set[str]] = defaultdict(set)
        self.all_problems: Set[str] = set()
        self.all_causes: Set[str] = set()
        self.all_actions: Set[str] = set()

        for h, rel, t in triples:
            if rel == "indicates" and h.startswith("SYMPTOM::") and t.startswith("PROBLEM::"):
                self.symptom_to_problems[h].add(t)
                self.all_problems.add(t)
            elif rel == "caused_by" and h.startswith("PROBLEM::") and t.startswith("CAUSE::"):
                self.problem_to_causes[h].add(t)
                self.all_causes.add(t)
            elif rel == "mitigated_by" and h.startswith("PROBLEM::") and t.startswith("ACTION::"):
                self.problem_to_actions[h].add(t)
                self.all_actions.add(t)

def _strip_prefix(x: str) -> str:
    # PROBLEM::foo -> foo
    return x.split("::", 1)[1] if "::" in x else x

def get_candidates(
    idx: CandidateIndex,
    symptoms: List[str],
    k_problems: int = 30,
    k_causes: int = 50,
    k_actions: int = 50,
) -> Dict[str, List[str]]:
    # symptoms are raw ids like "wire_slips_in_tensioner"
    symptom_nodes = [f"SYMPTOM::{s}" for s in symptoms]

    # collect candidate problems from symptoms
    cand_problems: Set[str] = set()
    for s in symptom_nodes:
        cand_problems |= idx.symptom_to_problems.get(s, set())

    # If nothing found (cold start), fall back to empty (don’t inject huge list)
    problems = sorted([_strip_prefix(p) for p in cand_problems])[:k_problems]

    # collect causes/actions from those problems
    cand_causes: Set[str] = set()
    cand_actions: Set[str] = set()
    for p in cand_problems:
        cand_causes |= idx.problem_to_causes.get(p, set())
        cand_actions |= idx.problem_to_actions.get(p, set())

    causes = sorted([_strip_prefix(c) for c in cand_causes])[:k_causes]
    actions = sorted([_strip_prefix(a) for a in cand_actions])[:k_actions]

    return {"problems": problems, "causes": causes, "actions": actions}