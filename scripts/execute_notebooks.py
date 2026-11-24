"""Execute case notebooks with real APIs, saving after success or failure.

Requires the notebook extra. Credentials must already exist in the environment.
Each case runs in a separate kernel, so model globals and random state are isolated.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]


def execute(case_id, kernel_name, rerun=False):
    import nbformat
    from nbclient import NotebookClient
    path = ROOT / "notebooks" / "cases" / f"{case_id}.ipynb"
    nb = nbformat.read(path, as_version=4)
    case_results = ROOT / "results" / "cases" / case_id
    if any(cell.get("execution_count") is not None for cell in nb.cells if cell.cell_type == "code"):
        if not rerun:
            raise FileExistsError(f"Already executed: {path.name}; use --rerun to preserve and repeat it.")
        previous_run = nb.metadata.get("ai_buckingham", {}).get("run_directory")
        previous_dir = ROOT / previous_run if previous_run else next(iter(sorted(case_results.glob("*"), reverse=True)), None)
        if previous_dir and previous_dir.is_dir() and not (previous_dir / "notebook.ipynb").exists():
            shutil.copy2(path, previous_dir / "notebook.ipynb")
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
    before_runs = {p.name for p in case_results.glob("*") if p.is_dir()}
    status, error = "completed", None
    print(json.dumps({"case": case_id, "event": "started"}), flush=True)
    try:
        NotebookClient(nb, kernel_name=kernel_name, timeout=None,
                       resources={"metadata": {"path": str(ROOT)}},
                       record_timing=True).execute()
    except Exception as exc:
        status, error = "failed", type(exc).__name__
    new_runs = [p for p in case_results.glob("*") if p.is_dir() and p.name not in before_runs]
    run_dir = sorted(new_runs)[-1] if new_runs else None
    nb.metadata["ai_buckingham"] = {"case_id": case_id, "execution": status,
              "executed_at": datetime.now(timezone.utc).isoformat(), "error_type": error,
              "run_directory": run_dir.relative_to(ROOT).as_posix() if run_dir else None}
    # Do not distribute a machine-specific kernel name.
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, path)
    if run_dir:
        shutil.copy2(path, run_dir / "notebook.ipynb")
    record = {"case": case_id, "event": "finished", "status": status, "error_type": error}
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Public case ID; repeat to select multiple cases")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rerun", action="store_true", help="Preserve an executed notebook with its run, then execute a new attempt")
    args = parser.parse_args()
    for key in ("OPENAI_API_KEY", "PARATERA_API_KEY"):
        if not os.environ.get(key):
            raise EnvironmentError(f"Set {key} before executing notebooks.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    case_ids = args.case or [p.stem for p in sorted((ROOT / "configs" / "cases").glob("*.json"))]
    with tempfile.TemporaryDirectory(prefix="ai_buckingham_kernels_") as folder:
        kernel_name = "ai-buckingham-build"
        kernel_dir = Path(folder) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "AI Buckingham build", "language": "python"}), encoding="utf-8")
        previous_path = os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_PATH"] = folder + (os.pathsep + previous_path if previous_path else "")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            results = [f.result() for f in as_completed([executor.submit(execute, case, kernel_name, args.rerun) for case in case_ids])]
    output = ROOT / "results" / "notebook_execution.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    output.write_text(json.dumps(history + results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "completed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
