"""Index actual case runs and figure sources without inventing missing evidence."""
from pathlib import Path
import json
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from ai_buckingham.artifacts import find_runs, write_json
    cases, summaries = [], []
    for config in sorted((ROOT / "configs" / "cases").glob("*.json")):
        case = json.loads(config.read_text(encoding="utf-8"))
        runs = find_runs(ROOT, case["case_id"])
        records = []
        for run in runs:
            folder = Path(run.pop("run_dir"))
            run.pop("_run_dir", None)
            trace = folder / "evidence" / "optimization_trace.jsonl"
            generations = 0
            if trace.is_file():
                generations = sum(json.loads(line).get("event") == "optimizer_generation" for line in trace.read_text(encoding="utf-8").splitlines() if line.strip())
            records.append({"run_id": run["run_id"], "status": run["status"], "mode": run.get("mode"),
                            "directory": folder.relative_to(ROOT).as_posix(),
                            "manifest": (folder / "manifest.json").relative_to(ROOT).as_posix(),
                            "optimizer_generations": generations,
                            "figures": [p.relative_to(ROOT).as_posix() for p in sorted((folder / "figures").glob("*"))],
                            "report": (folder / "report.html").relative_to(ROOT).as_posix()})
        successful = [item for item in records if item["status"] == "completed"]
        if successful:
            state = json.loads((ROOT / successful[-1]["directory"] / "state.json").read_text(encoding="utf-8"))
            numerical = state.get("workflow_result") or {}
            for local in numerical.get("local_workflows", []):
                if local.get("status") != "success":
                    continue
                final = local.get("final_solution") or {}
                summaries.append({"case_id": case["case_id"], "run_id": successful[-1]["run_id"],
                    "mechanism_decision": numerical.get("mechanism_decision"), "scope_id": local.get("scope_id"),
                    "local_samples": local.get("n_samples"), "similarity_type": local.get("similarity_type"),
                    "group_count": local.get("locked_count"), "mutual_information": final.get("mutual_information"),
                    "information_score": final.get("normalized_information_score"),
                    "conditional_test_r2": (final.get("gpr_validation") or {}).get("test_r2_mean"),
                    "input_formulas": json.dumps(final.get("input_formulas"), ensure_ascii=False),
                    "output_formula": final.get("output_dimensionless_formula"),
                    "recorded_attempts": len(records)})
        cases.append({"case_id": case["case_id"], "title": case["title"],
                      "dataset": case["dataset"], "notebook": f"notebooks/cases/{case['case_id']}.ipynb",
                      "selected_run": successful[-1]["run_id"] if successful else None,
                      "runs": records})
    study = {
        "mhd_discovery": ["mhd"],
        "four_type_i_examples": ["pressure_gradient", "rayleigh_benard", "laser_keyhole", "blasius"],
        "sandgrain_prediction": ["sandgrain"],
        "type_ii_examples": ["velocity_scaling", "goldenfeld"],
        "multiple_regimes": ["cess_channel", "rough_pipe"],
    }
    index = {"schema_version": "1.0", "cases": cases, "study_figure_groups": study,
             "historical_comparisons": {name: f"reproduction/{name}/README.md" for name in ["rag_ablation", "regime_comparison", "velocity_comparison"]},
             "fresh_landscapes": "results/mi_landscapes" if (ROOT / "results" / "mi_landscapes").exists() else None,
             "interpretation": "The included snapshot contains reviewed completed case executions. Historical comparison evidence is identified separately; reader-created runs belong under generated/."}
    write_json(ROOT / "results" / "index.json", index)
    if summaries:
        with (ROOT / "results" / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    rows = ["# Included case evidence", "", index["interpretation"], "", "| Case | Selected completed run | Recorded attempts | Notebook |", "|---|---|---:|---|"]
    for case in cases:
        selected = next((r for r in case["runs"] if r["run_id"] == case["selected_run"]), None)
        link = f"[HTML report]({Path(selected['report']).relative_to('results').as_posix()})" if selected else "No completed live run"
        rows.append(f"| {case['title']} | {link} | {len(case['runs'])} | [notebook](../{case['notebook']}) |")
    rows += ["", "Each run directory contains its actual state, node updates, resolved settings, recorded displays, optimizer trace, numerical tables and original figures. See `index.json` for machine-readable paths and all attempts.", ""]
    (ROOT / "results" / "README.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "completed_cases": sum(bool(c["selected_run"]) for c in cases), "attempts": sum(len(c["runs"]) for c in cases)}))


if __name__ == "__main__":
    main()
