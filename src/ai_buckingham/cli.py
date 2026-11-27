"""Explicit analysis entry points; saved-result browsing never initializes clients."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from .artifacts import RunRecorder, new_run_id, read_json, repository_root, write_json
from .bindings import frontend_bindings
from .execution import numerical_runtime


def load_case(case_id: str) -> dict:
    path = repository_root() / "configs" / "cases" / f"{case_id}.json"
    if not path.is_file():
        raise ValueError(f"Unknown case: {case_id}")
    return read_json(path)


def run_analysis(config_or_case_id, output_dir=None, *, initialize=True,
                 display_cards=True, on_event=None, model_names=None, agent_graph=None):
    """Execute the real graph and retain every node update and rich display.

    User configuration is passed to the existing numerical pipeline unchanged.
    Each execution has its own immutable directory. Model credentials are read
    from the environment and are never included in a result bundle.
    """
    from .graph import build_agent_graph
    from .models import initialize_models
    from .recording import record_run
    config = load_case(config_or_case_id) if isinstance(config_or_case_id, str) else dict(config_or_case_id)
    case_id = str(config.get("case_id") or config.get("id") or "custom")
    root = repository_root()
    dataset = Path(config["dataset"])
    if not dataset.is_absolute():
        dataset = root / dataset
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset is missing: {dataset}")
    run_dir = Path(output_dir).resolve() if output_dir else root / "generated" / "cases" / case_id / new_run_id()
    recorder = RunRecorder(run_dir, case_id, config, mode="live")
    state = {}
    captured = None
    try:
        # Capture the implementation's actual displays, including running and
        # completed cards. Replay them into the notebook after this call.
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        InteractiveShell.instance()
        with record_run(run_dir), numerical_runtime(config) as runtime, capture_output(stdout=True, stderr=True, display=True) as captured:
            recorder.manifest["numerical_runtime"] = runtime
            write_json(run_dir / "manifest.json", recorder.manifest)
            if initialize:
                initialize_models(**(model_names or {}))
            graph = agent_graph if agent_graph is not None else build_agent_graph()
            payload = {"messages": [], "prompt_problem": config["prompt"],
                       "csv_FilePath": str(dataset),
                       "manual_hyperparams": dict(config.get("manual_hyperparams") or {})}
            payload.update(frontend_bindings(config))
            for mode, item in graph.stream(payload, config={"recursion_limit": 100}, stream_mode=["updates", "values"]):
                if mode == "values":
                    state = item
                    recorder.last_state = state
                elif mode == "updates":
                    for node, update in item.items():
                        recorder.record_event(node, update)
                        if on_event:
                            on_event({"node": node, "sequence": recorder.event_count, "run_dir": str(run_dir)})
        recorder.save_rich_outputs(captured.outputs)
        (run_dir / "console.txt").write_text(captured.stdout + captured.stderr, encoding="utf-8")
        workflow_status = state.get("workflow_status")
        status = "completed" if workflow_status == "success" else "blocked" if workflow_status in {"terminal_unresolved", "basis_failed"} else "warning"
        recorder.finalize(state, status=status)
    except BaseException as exc:
        if captured is not None:
            recorder.save_rich_outputs(captured.outputs)
            (run_dir / "console.txt").write_text(captured.stdout + captured.stderr, encoding="utf-8")
        recorder.fail(exc)
        raise
    finally:
        if display_cards and captured is not None:
            captured.show()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an AI Buckingham case with explicit model initialization.")
    parser.add_argument("case", nargs="?", help="Case identifier")
    parser.add_argument("--list", action="store_true", help="List case identifiers without API access")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.list:
        for path in sorted((repository_root() / "configs" / "cases").glob("*.json")):
            print(path.stem)
        return 0
    if not args.case:
        parser.error("Supply a case identifier or --list")
    def progress(event):
        import sys
        print(json.dumps(event), file=sys.__stdout__, flush=True)
    state = run_analysis(args.case, args.output_dir, display_cards=False, on_event=progress)
    return 0 if state.get("workflow_status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
