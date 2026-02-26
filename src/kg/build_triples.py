from typing import Dict, List, Tuple

Triple = Tuple[str, str, str]

def build_triples_from_record(rec: Dict) -> List[Triple]:
    problem = rec["problem"]
    triples: List[Triple] = []

    for s in rec["symptoms"]:
        triples.append((f"SYMPTOM::{s}", "indicates", f"PROBLEM::{problem}"))

    for c in rec["possible_causes"]:
        triples.append((f"PROBLEM::{problem}", "caused_by", f"CAUSE::{c}"))

    for a in rec["recommended_actions"]:
        # simplest: problem -> action
        triples.append((f"PROBLEM::{problem}", "mitigated_by", f"ACTION::{a}"))

    return triples