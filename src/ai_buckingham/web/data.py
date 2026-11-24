"""Small file adapters shared by the six Streamlit views.

Loading a view never imports the numerical engine, initializes a model, or sends
an API request. The evidence archive is read without rewriting historical files.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CASES = {
    "mhd": ("MHD power generator", "Type I", "dataset_mhd.csv"),
    "pressure_gradient": ("Pressure-gradient skin friction", "Type I", "dataset_presGrad.csv"),
    "rayleigh_benard": ("Rayleigh–Bénard convection", "Type I", "dataset_rayben.csv"),
    "laser_keyhole": ("Laser keyhole depth", "Type I", "dataset_keyhole.csv"),
    "blasius": ("Blasius boundary layer", "Type I", "dataset_blasius.csv"),
    "sandgrain": ("Equivalent sand-grain roughness", "Type I", "dataset_sandgrain.csv"),
    "velocity_scaling": ("Compressible velocity scaling", "Type II", "dataset_veloTrans.csv"),
    "goldenfeld": ("Goldenfeld rough-pipe scaling", "Type II", "dataset_goldenfeld.csv"),
    "cess_channel": ("Cess eddy-viscosity model", "Multiple regimes", "dataset_cess_channel.csv"),
    "rough_pipe": ("Rough circular pipe", "Multiple regimes", "dataset_pipe.csv"),
}
ALIASES = {"presgrad": "pressure_gradient", "keyhole": "laser_keyhole", "rayleigh": "rayleigh_benard",
           "velotrans": "velocity_scaling", "cess": "cess_channel", "pipe": "rough_pipe"}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {} if default is None else default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def json_lines(path: Path) -> list[dict]:
    """Also tolerates the last, partially written line of a running trace."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except ValueError:
            continue
    return records


def case_configs() -> dict[str, dict]:
    configs = {}
    for path in sorted((ROOT / "configs" / "cases").glob("*.json")):
        content = read_json(path)
        case_id = content.get("case_id", path.stem)
        configs[case_id] = content
    return configs


def case_catalog() -> dict[str, dict]:
    configs = case_configs()
    result = {}
    for key, config in configs.items():
        canonical = ALIASES.get(key, key)
        title, kind, filename = CASES.get(canonical, (key.replace("_", " ").title(), "Recorded case", ""))
        result[key] = {"title": config.get("title", config.get("name", title)), "kind": kind,
                       "filename": filename, "config": config}
    if not result:
        result = {key: {"title": value[0], "kind": value[1], "filename": value[2], "config": {}}
                  for key, value in CASES.items()}
    return result


def dataset_path(case_id: str) -> Path:
    catalog = case_catalog()
    item = catalog.get(case_id, {})
    config = item.get("config", {})
    configured = config.get("dataset", config.get("dataset_path", config.get("csv_path")))
    if isinstance(configured, dict):
        configured = configured.get("path", configured.get("file"))
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_file():
            return candidate
    filename = item.get("filename", "")
    for candidate in [ROOT / "data" / "datasets" / filename,
                      ROOT / "data" / "datasets" / f"{case_id}.csv"]:
        if filename and candidate.is_file():
            return candidate
    return ROOT / "data" / "datasets" / filename


def runs(case_id: str | None = None) -> list[Path]:
    found = []
    for base in [ROOT / "results" / "cases", ROOT / "generated" / "cases"]:
        patterns = [base / case_id] if case_id else list(base.glob("*"))
        for folder in patterns:
            for manifest in folder.glob("*/manifest.json"):
                found.append(manifest.parent)
            if (folder / "manifest.json").is_file():
                found.append(folder)
    return sorted(found, key=lambda p: p.name, reverse=True)


def load_run(path: Path) -> dict:
    return {"path": path, "manifest": read_json(path / "manifest.json"),
            "config": read_json(path / "resolved_config.json"),
            "state": read_json(path / "state.json"), "events": json_lines(path / "events.jsonl")}


def find_values(value: Any, *keys: str) -> list[Any]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys:
                found.append(child)
            elif isinstance(child, (dict, list)):
                found.extend(find_values(child, *keys))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                found.extend(find_values(child, *keys))
    return found


def first(value: Any, *keys: str, default: Any = None) -> Any:
    # Plans and results share names such as regime_detection. Numerical views
    # must use the executed result before planner settings and earlier attempts.
    if isinstance(value, dict) and isinstance(value.get("workflow_result"), dict):
        executed = find_values(value["workflow_result"], *keys)
        if executed:
            return executed[0]
    values = find_values(value, *keys)
    return values[0] if values else default


def scrub(value: Any) -> Any:
    """Prevent credential fields from appearing in JSON previews/downloads."""
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("api_key", "authorization", "password", "secret", "access_token"))
                    else scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def archive_bytes(folder: Path) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(folder)
            if any(part.startswith(".") or part in {"cache", "__pycache__"} for part in relative.parts):
                continue
            if path.suffix.lower() in {".json", ".jsonl"}:
                if path.suffix == ".json":
                    payload = json.dumps(scrub(read_json(path)), ensure_ascii=False, indent=2)
                else:
                    payload = "\n".join(json.dumps(scrub(r), ensure_ascii=False) for r in json_lines(path))
                archive.writestr(relative.as_posix(), payload)
            else:
                archive.write(path, relative.as_posix())
    return output.getvalue()


def reproduction(name: str) -> Path:
    return ROOT / "reproduction" / name
