"""Locate the paper's selected run for a reproduction notebook."""
from __future__ import annotations

import json
from pathlib import Path


def published_run(case_id: str, root: Path) -> Path:
    """Select the same completed case run used by the paper's result pages."""
    index = json.loads((root / "results/index.json").read_text(encoding="utf-8"))
    case = next(item for item in index["cases"] if item["case_id"] == case_id)
    run = next(item for item in case["runs"] if item["run_id"] == case["selected_run"])
    if run["status"] != "completed":
        raise ValueError(f"The selected paper run for {case_id} is incomplete")
    return root / run["directory"]
