from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_output_dir(root: Path, prefix: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_directory(root / f"{prefix}-{stamp}")


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
