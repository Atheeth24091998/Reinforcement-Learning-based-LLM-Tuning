from typing import Any, Dict, List

def normalize_record(r: Dict[str, Any]) -> Dict[str, Any]:
    # Supports both:
    # - {explicit_symptoms:[], implicit_symptoms:[]}
    # - {symptoms:[]}
    explicit = r.get("explicit_symptoms", []) or []
    implicit = r.get("implicit_symptoms", []) or []
    symptoms = r.get("symptoms", None)
    if symptoms is None:
        symptoms = list(explicit) + list(implicit)

    return {
        "problem": r.get("problem"),
        "symptoms": symptoms,
        "possible_causes": r.get("possible_causes", []) or [],
        "recommended_actions": r.get("recommended_actions", []) or [],
    }