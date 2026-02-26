import re
from typing import Dict, List
from .graph import KG

CAUSE_RE = re.compile(r"CAUSE::<?([a-zA-Z0-9_\-]+)>?")
ACTION_RE = re.compile(r"ACTION::<?([a-zA-Z0-9_\-]+)>?")
PROBLEM_RE = re.compile(r"PROBLEM:\s*(PROBLEM::[a-zA-Z0-9_\-]+)", re.IGNORECASE)

def extract_entities(text: str) -> Dict[str, List[str]]:
    causes = [m.group(1) for m in CAUSE_RE.finditer(text)]
    actions = [m.group(1) for m in ACTION_RE.finditer(text)]
    return {"causes": causes, "actions": actions}

def compute_reward(kg: KG, rec: Dict, model_text: str, max_hops: int = 3) -> float:
    sources = [f"SYMPTOM::{s}" for s in rec["symptoms"]] + [f"PROBLEM::{rec['problem']}"]

    # --- format reward (dense) ---
    format_reward = 0.0
    if "PROBLEM:" in model_text: format_reward += 0.1
    if "CAUSES:" in model_text:  format_reward += 0.1
    if "ACTIONS:" in model_text: format_reward += 0.1

    # penalize placeholders / numeric ids (common failure)
    bad = 0
    bad += len(re.findall(r"CAUSE::<?\d+>?", model_text))
    bad += len(re.findall(r"ACTION::<?\d+>?", model_text))
    bad += len(re.findall(r"CAUSE::<", model_text))
    bad += len(re.findall(r"ACTION::<", model_text))
    placeholder_penalty = 0.05 * bad  # mild but present

    ents = extract_entities(model_text)
    pred_causes = [f"CAUSE::{c}" for c in ents["causes"]]
    pred_actions = [f"ACTION::{a}" for a in ents["actions"]]

    # if nothing extracted, return only format - penalty (still gives signal)
    if not pred_causes and not pred_actions:
        return max(0.0, format_reward - placeholder_penalty)

    total = len(pred_causes) + len(pred_actions)

    # entity validity
    valid = sum(1 for n in pred_causes + pred_actions if n in kg.nodes)
    r_entity = valid / max(total, 1)

    # reachability
    reach = sum(1 for n in pred_causes + pred_actions if kg.reachable(sources, n, max_hops=max_hops))
    r_path = reach / max(total, 1)

    # final
    R = 0.15 * format_reward + 0.5 * r_entity + 0.35 * r_path - placeholder_penalty
    return float(max(0.0, min(1.0, R)))