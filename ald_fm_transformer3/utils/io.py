from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)
