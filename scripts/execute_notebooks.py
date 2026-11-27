"""Render saved case notebooks, or explicitly run new case analyses.

Executed notebook copies are written to a new output directory. The notebook
extra supplies the Jupyter kernel and client used by this command.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def execute(case_id, kernel_name, output_dir, run_analysis=False):
    import nbformat
    from nbclient import NotebookClient
    from ai_buckingham.artifacts import new_run_id
    from ai_buckingham.notebook import published_run

    source = ROOT / "notebooks/cases" / f"{case_id}.ipynb"
    destination = output_dir / source.name
    if destination.exists():
        raise FileExistsError(f"Notebook output already exists: {destination}")
    notebook = nbformat.read(source, as_version=4)
    run_dir = published_run(case_id, ROOT)
    if run_analysis:
        run_dir = ROOT / "generated/cases" / case_id / new_run_id()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            if run_analysis:
                cell.source = cell.source.replace("RUN_NEW_ANALYSIS = False", "RUN_NEW_ANALYSIS = True")
                cell.source = cell.source.replace("ROOT / 'generated' / 'cases' / CASE_ID / new_run_id()", f"Path({str(run_dir)!r})")
    status, error = "completed", None
    print(json.dumps({"case": case_id, "event": "started", "mode": "analysis" if run_analysis else "saved_results"}), flush=True)
    try:
        NotebookClient(notebook, kernel_name=kernel_name, timeout=None,
                       resources={"metadata": {"path": str(ROOT)}}, record_timing=True).execute()
    except Exception as exc:
        status, error = "failed", type(exc).__name__
    notebook.metadata["ai_buckingham"] = {
        "case_id": case_id, "execution": status, "execution_mode": "analysis" if run_analysis else "saved_results",
        "executed_at": datetime.now(timezone.utc).isoformat(), "error_type": error,
        "run_directory": run_dir.relative_to(ROOT).as_posix() if run_dir else None,
    }
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    output_dir.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    record = {"case": case_id, "event": "finished", "status": status, "error_type": error,
              "notebook": str(destination)}
    print(json.dumps(record), flush=True)
    return record


def main():
    from ai_buckingham.artifacts import new_run_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Case ID; repeat to select multiple cases")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-analysis", "--rerun", action="store_true", help="Enable full analysis with model services")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated/notebooks" / new_run_id())
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for source in [ROOT / "notebooks", ROOT / "results", ROOT / "data", ROOT / "configs", ROOT / "reproduction"]:
        if output.is_relative_to(source.resolve()):
            parser.error("Choose an output directory outside the published notebooks and case inputs/results")
    case_ids = list(dict.fromkeys(args.case or [path.stem for path in sorted((ROOT / "configs/cases").glob("*.json"))]))
    for case_id in case_ids:
        if not (ROOT / "configs/cases" / f"{case_id}.json").is_file():
            parser.error(f"Unknown case ID: {case_id}")
    if args.run_analysis:
        for key in ["OPENAI_API_KEY", "PARATERA_API_KEY"]:
            if not os.environ.get(key):
                raise EnvironmentError(f"Set {key} before running a new analysis")
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ.setdefault(name, "1")
    with tempfile.TemporaryDirectory(prefix="ai_buckingham_kernels_") as folder:
        kernel_name = "ai-buckingham-build"
        kernel_dir = Path(folder) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "AI Buckingham", "language": "python",
        }), encoding="utf-8")
        previous_path = os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_PATH"] = folder + (os.pathsep + previous_path if previous_path else "")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            jobs = [executor.submit(execute, case, kernel_name, output, args.run_analysis) for case in case_ids]
            results = [future.result() for future in as_completed(jobs)]
    output.mkdir(parents=True, exist_ok=True)
    (output / "execution.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "completed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
