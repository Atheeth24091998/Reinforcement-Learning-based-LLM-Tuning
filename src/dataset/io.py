import json
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Expected a list of records in {p}, got {type(obj)}")