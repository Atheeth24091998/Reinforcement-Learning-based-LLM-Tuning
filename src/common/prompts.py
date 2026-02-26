SYSTEM = """You are a troubleshooting assistant for an industrial machine manual.

You MUST answer in EXACTLY this format (3 lines only):

PROBLEM: PROBLEM::<problem_id>
CAUSES: CAUSE::<id>, CAUSE::<id>, ...
ACTIONS: ACTION::<id>, ACTION::<id>, ...

Rules:
- Use ONLY ids that exist in the knowledge graph.
- Never output numbers like CAUSE::1 or placeholders like <1>.
- If unsure, output empty lists: CAUSES:  and ACTIONS:
- Do not add any other text.
"""

def build_user_prompt(symptoms: list[str], candidates: dict | None = None) -> str:
    s = ", ".join(symptoms)
    base = f"Symptoms: {s}\n"

    if candidates and (candidates.get("problems") or candidates.get("causes") or candidates.get("actions")):
        # Keep these lists reasonably small
        p = ", ".join(candidates.get("problems", []))
        c = ", ".join(candidates.get("causes", []))
        a = ", ".join(candidates.get("actions", []))
        base += (
            "Choose ONLY from these candidates (do not invent new IDs):\n"
            f"- PROBLEMS: {p}\n"
            f"- CAUSES: {c}\n"
            f"- ACTIONS: {a}\n"
        )

    return base + "Return the best PROBLEM/CAUSES/ACTIONS in the required format."