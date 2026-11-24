"""Portable, inspectable evidence bundles shared by notebooks and the web app.

This module deliberately has no model, graph, or numerical-library imports.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[2]
_SECRET_KEYS = {"api_key", "openai_api_key", "paratera_api_key", "authorization", "cookie", "access_token", "secret", "password"}


def repository_root() -> Path:
    import os
    configured = os.environ.get("AI_BUCKINGHAM_PROJECT_ROOT") or os.environ.get("AI_BUCKINGHAM_ROOT")
    if configured:
        return Path(configured).resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").is_dir():
        return candidate
    return Path.cwd().resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(value: Any, root: Path | None = None) -> Any:
    """Convert NumPy/SymPy/Pydantic/messages to JSON without pickled objects."""
    root = Path(root or repository_root()).resolve()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (str, Path)):
        text = str(value)
        for prefix in (str(root), str(root).replace("\\", "/")):
            text = text.replace(prefix + "\\", "").replace(prefix + "/", "")
        return text
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower() in _SECRET_KEYS else portable(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [portable(item, root) for item in value]
    if hasattr(value, "model_dump"):
        return portable(value.model_dump(), root)
    if hasattr(value, "tolist"):
        return portable(value.tolist(), root)
    if hasattr(value, "item"):
        return portable(value.item(), root)
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(portable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def find_runs(root: Path | str | None = None, case_id: str | None = None) -> list[dict]:
    root = Path(root or repository_root())
    base = root / "results" / "cases"
    manifests = base.glob(f"{case_id or '*'}/*/manifest.json")
    found = []
    for path in sorted(manifests):
        record = read_json(path, {})
        record.update(run_dir=str(path.parent), _run_dir=str(path.parent))
        found.append(record)
    return found


def load_run(path: Path | str | dict) -> dict:
    if isinstance(path, dict):
        path = path.get("run_dir") or path.get("_run_dir")
    run_dir = Path(path)
    if run_dir.name == "manifest.json":
        run_dir = run_dir.parent
    events = []
    event_path = run_dir / "events.jsonl"
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return {"run_dir": str(run_dir), "manifest": read_json(run_dir / "manifest.json", {}),
            "state": read_json(run_dir / "state.json", {}),
            "config": read_json(run_dir / "resolved_config.json", {}), "events": events,
            "figures": [str(p) for p in sorted((run_dir / "figures").glob("*"))]}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def saved_figures_html(run_dir: Path) -> str:
    """Embed the exact saved PNGs, including when Matplotlib uses Agg."""
    figures = sorted((Path(run_dir) / "figures").glob("*.png"))
    originals = [path for path in figures if not path.name.startswith("display_")]
    if not figures:
        return ""
    parts = ['<section id="saved-figures"><h2>Saved evidence figures</h2>']
    for path in originals or figures:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append('<figure><img src="data:image/png;base64,' + encoded + '" alt="' + html.escape(path.stem) + '"><figcaption>' + html.escape(path.name) + '</figcaption></figure>')
    return "\n".join(parts) + "</section>"


class RunRecorder:
    """Append node events and atomically finalize a new run; never overwrite one."""

    def __init__(self, run_dir: Path, case_id: str, config: dict, mode: str = "live"):
        self.run_dir = Path(run_dir).resolve()
        if (self.run_dir / "manifest.json").exists():
            raise FileExistsError(f"Existing run is immutable: {self.run_dir}")
        self.case_id, self.config, self.mode = case_id, config, mode
        for name in ("evidence", "tables", "figures"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        root = repository_root()
        dataset = root / config.get("dataset", "")
        packages = {}
        for name in ("numpy", "pandas", "scipy", "sympy", "scikit-learn", "cma", "openai", "langgraph", "langchain-openai", "pydantic"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        source_files = sorted((root / "src" / "ai_buckingham").rglob("*.py"))
        self.manifest = {"schema_version": SCHEMA_VERSION, "case_id": case_id,
                         "run_id": self.run_dir.name, "mode": mode, "status": "running",
                         "started_at": utc_now(), "dataset": config.get("dataset"),
                         "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
                         "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in source_files},
                         "python": platform.python_version(), "platform": platform.platform(),
                         "packages": packages, "interpretation": "New execution with the supplied implementation and case settings; not the historical manuscript run.",
                         "coverage": {"node_events": True, "optimization_generations": True}}
        self.event_count = 0
        self.last_state = {}
        self.rich_outputs = []
        write_json(self.run_dir / "manifest.json", self.manifest)
        write_json(self.run_dir / "resolved_config.json", config)

    def record_event(self, node: str, update: Any) -> None:
        self.event_count += 1
        event = {"sequence": self.event_count, "timestamp": utc_now(), "node": node,
                 "update": portable(update)}
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        if isinstance(update, dict):
            write_json(self.run_dir / "evidence" / f"node_{self.event_count:02d}_{re.sub('[^A-Za-z0-9_]', '_', node)}.json", update)

    def save_rich_outputs(self, outputs: list) -> None:
        """Persist actual IPython display bundles, not reconstructed cards."""
        for index, output in enumerate(outputs):
            data = output.get("data", {}) if isinstance(output, dict) else getattr(output, "data", {})
            if data:
                self.rich_outputs.append(portable(data))
                if "image/png" in data:
                    (self.run_dir / "figures" / f"display_{index:03d}.png").write_bytes(base64.b64decode(data["image/png"]))
        write_json(self.run_dir / "evidence" / "display_bundles.json", self.rich_outputs)

    def finalize(self, state: dict, status: str = "completed") -> dict:
        self.last_state = state
        write_json(self.run_dir / "state.json", state)
        actual = dict(self.config)
        actual.update(resolved_hyperparameters=state.get("resolved_hyperparameters"),
                      hyperparameter_provenance=state.get("hyperparameter_provenance"),
                      workflow_plan=state.get("workflow_plan"), compiled_workflow=state.get("compiled_workflow"))
        write_json(self.run_dir / "resolved_config.json", actual)
        groups = {
            "retrieval": ["knowledge_retrieval", "dimension_query", "dimension_audit", "dimension_audit_history", "interpretation_knowledge_retrieval"],
            "dimensions": ["dimension_ledger", "dimension_contract", "dimension_mat", "basis_pi_formulas", "basis_pi_matrices", "input_vars", "output_var", "output_resolution"],
            "workflow": ["workflow_result", "workflow_trace", "workflow_attempt_history", "run_provenance"],
        }
        for name, keys in groups.items():
            write_json(self.run_dir / "evidence" / f"{name}.json", {k: state[k] for k in keys if k in state})
        self._export_numeric_tables(state)
        if state.get("workflow_result") and state.get("input_vars"):
            from .plot_data import export_plot_data
            export_plot_data(state, self.run_dir)
        self.manifest.update(status=status, completed_at=utc_now(), node_events=self.event_count,
                             workflow_status=state.get("workflow_status"))
        write_json(self.run_dir / "manifest.json", self.manifest)
        self.export()
        return self.manifest

    def _export_numeric_tables(self, state: dict) -> None:
        rows = []
        def walk(value, path="workflow_result"):
            if isinstance(value, dict):
                scalar = {k: v for k, v in value.items() if v is None or isinstance(v, (str, int, float, bool))}
                if any(k.lower() in {"mi", "mutual_information", "normalized_score", "r2", "mean_r2", "n_clusters", "best_mi"} for k in scalar):
                    rows.append({"path": path, **scalar})
                for key, item in value.items():
                    walk(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")
        walk(portable(state.get("workflow_result", {})))
        _write_csv(self.run_dir / "tables" / "metrics.csv", rows)
        _write_csv(self.run_dir / "tables" / "node_timings.csv", portable((state.get("run_provenance") or {}).get("node_timings") or []))
        for key in ("knowledge_retrieval", "interpretation_knowledge_retrieval"):
            item = portable(state.get(key) or {})
            if isinstance(item, dict):
                for name, value in item.items():
                    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                        _write_csv(self.run_dir / "tables" / f"{key}_{name}.csv", value)

    def fail(self, exc: BaseException) -> None:
        self.record_event("__error__", {"error_type": type(exc).__name__})
        self.manifest.update(status="failed", completed_at=utc_now(), error_type=type(exc).__name__, node_events=self.event_count)
        write_json(self.run_dir / "manifest.json", self.manifest)
        if self.last_state:
            write_json(self.run_dir / "state.json", self.last_state)
        self.export()

    def export(self) -> None:
        state = portable(self.last_state)
        report = str(state.get("final_report") or "")
        intro = f"# AI Buckingham — {self.case_id}\n\nRun: `{self.run_dir.name}`  \nStatus: {self.manifest['status']}  \nMode: {self.mode}\n\n"
        (self.run_dir / "report.md").write_text(intro + report + "\n\nFull structured evidence is in state.json, events.jsonl and evidence/.\n", encoding="utf-8")
        cards = []
        for data in self.rich_outputs:
            if "text/html" in data:
                cards.append(data["text/html"])
            elif "image/png" in data:
                cards.append('<img style="max-width:100%" src="data:image/png;base64,' + data["image/png"] + '" alt="Recorded output">')
            elif "text/plain" in data:
                cards.append("<pre>" + html.escape(data["text/plain"]) + "</pre>")
        rendered = "\n".join(cards) or ("<pre>" + html.escape(report or "No rich display outputs were recorded for this run.") + "</pre>")
        rendered += saved_figures_html(self.run_dir)
        page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Buckingham — ' + html.escape(self.case_id) + '</title><style>body{font:16px/1.55 system-ui;margin:32px auto;max-width:1240px;padding:0 20px;color:#17202a}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}.identity{padding:14px;border:1px solid #ccd7de;background:#f0f6f9}</style></head><body><h1>AI Buckingham — ' + html.escape(self.case_id) + '</h1><p class="identity">' + html.escape(self.manifest["interpretation"]) + '<br>Run: ' + html.escape(self.run_dir.name) + ' · Status: ' + html.escape(self.manifest["status"]) + '</p>' + rendered + '</body></html>'
        (self.run_dir / "report.html").write_text(page, encoding="utf-8")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
