"""Shared paper titles, case ordering and result figures."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def publication():
    return json.loads((ROOT / "docs" / "paper_cases.json").read_text(encoding="utf-8"))


def paper_cases():
    return {case["id"]: case for case in publication()["cases"]}


def collapse_axes(case_id: str, panel: str | None = None) -> tuple[str, str]:
    case = paper_cases().get(case_id, {})
    axes = case.get("collapse_axes_by_panel", {}).get(panel, case.get("collapse_axes", ["linear", "linear"]))
    if not isinstance(axes, (list, tuple)) or len(axes) != 2 or any(axis not in {"log", "linear"} for axis in axes):
        raise ValueError(f"Invalid collapse axes for {case_id}/{panel}.")
    return tuple(axes)
