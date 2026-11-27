"""Validate included scientific evidence without invoking models or optimizers."""
from pathlib import Path
from fractions import Fraction
import argparse
import base64
import json
import re
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "ai-buckingham-checks/result_validation.json")
    args = parser.parse_args()
    import numpy as np
    import nbformat
    from ai_buckingham.artifacts import read_json, run_figures, sha256_file, write_json
    index = read_json(ROOT / "results" / "index.json")
    cases, failures, generation_count = [], [], 0
    for case in index["cases"]:
        config = read_json(ROOT / "configs" / "cases" / f"{case['case_id']}.json")
        selected = next((r for r in case["runs"] if r["run_id"] == case["selected_run"]), None)
        if selected is None:
            raise ValueError(f"No completed run for {case['case_id']}")
        directory = ROOT / selected["directory"]
        manifest, state = read_json(directory / "manifest.json"), read_json(directory / "state.json")
        if manifest["dataset_sha256"] != sha256_file(ROOT / config["dataset"]):
            raise ValueError("Changed analysis input")
        expected = config["expected_dimensions"]["inputs"]
        actual = {v["csv_name"]:v["si_exponents"] for v in state["dimension_ledger"]["variables"]}
        if set(actual) != set(expected) or set(state["input_vars"]) != set(expected):
            raise ValueError("Live input set differs from the declared case")
        for name, axes in expected.items():
            for axis, exponent in axes.items():
                if Fraction(str(actual[name][axis])) != Fraction(str(exponent)):
                    raise ValueError(f"Dimension mismatch: {case['case_id']} {name} {axis}")
        dimension_order = config["expected_dimensions"]["dimension_order"]
        D = np.asarray([[float(Fraction(str(actual[name][axis]))) for name in state["input_vars"]] for axis in dimension_order])
        B = np.asarray(state["basis_pi_matrices"], dtype=float)
        residual = float(np.max(np.abs(D @ B.T)))
        if residual > 1e-10 or B.shape[0] != len(expected) - np.linalg.matrix_rank(D):
            raise ValueError("Invalid Buckingham-Pi null space")
        trace = directory / "evidence" / "optimization_trace.jsonl"
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        generations = [r for r in records if r.get("event") == "optimizer_generation"]
        if not generations or any(not isinstance(r["generation"], int) or r["generation"] < 1 for r in generations):
            raise ValueError("Missing actual CMA generations")
        generation_count += len(generations)
        nb = nbformat.read(ROOT / case["notebook"], as_version=4)
        nbformat.validate(nb)
        if nb.metadata["ai_buckingham"]["execution"] != "completed" or any(o.get("output_type") == "error" for c in nb.cells for o in c.get("outputs", [])):
            raise ValueError("Incomplete executed notebook")
        figures = [path for path in run_figures(directory) if path.suffix == ".png"]
        gallery = next((c for c in nb.cells if c.metadata.get("ai_buckingham_role") == "saved_figure_gallery"), None)
        if gallery is not None:
            actual_figures = [base64.b64decode(payload["image/png"]) for payload in gallery.attachments.values()]
        else:
            report_html = "".join(output.get("data", {}).get("text/html", "")
                                  for cell in nb.cells for output in cell.get("outputs", []))
            actual_figures = [base64.b64decode(encoded) for encoded in re.findall(
                r'<figure class="daa-report-figure"><img src="data:image/png;base64,([A-Za-z0-9+/=]+)"', report_html)]
        if sorted(actual_figures) != sorted(path.read_bytes() for path in figures):
            raise ValueError("Notebook figures differ from the selected run's PNGs")
        plot_manifest = read_json(directory / "evidence" / "plot_data_manifest.json")
        if plot_manifest["warnings"]:
            raise ValueError("Incomplete plot table export")
        for record in plot_manifest["files"]:
            if sha256_file(directory / record["file"]) != record["sha256"]:
                raise ValueError("Changed plot source table")
        numerical = state["workflow_result"]
        local = numerical["local_workflows"]
        if any(item.get("status") != "success" for item in local):
            raise ValueError("A local workflow did not succeed")
        cases.append({"case_id":case["case_id"], "run_id":manifest["run_id"], "dimensions_match_reference":True,
                      "nullspace_max_residual":residual, "basis_dimension":B.shape[0], "actual_cma_generations":len(generations),
                      "local_workflows":len(local), "group_counts":[v["locked_count"] for v in local],
                      "similarity_types":[v["similarity_type"] for v in local],
                      "candidate_partition_mode":"observed_during_execution" if (directory / "evidence" / "regime_candidates").exists() else "offline_recalculation_from_recorded_gradients",
                      "notebook_and_saved_figures_match":True})
        failures.extend({"case_id":case["case_id"], "run_id":r["run_id"], "status":r["status"]} for r in case["runs"] if r["status"] != "completed")
    comparison = read_json(ROOT / "results" / "sandgrain_prediction" / "manifest.json")
    if comparison.get("train_count") != 83 or comparison.get("test_count") != 21 or not comparison.get("learned_r2_matches_saved_run"):
        raise ValueError("Sandgrain comparison split or learned fit disagrees")
    report = {"status":"passed", "completed_cases":len(cases), "actual_cma_generations":generation_count,
              "cases":cases, "retained_unsuccessful_attempts":failures,
              "sandgrain_shared_split_and_learned_fit_agree":True,
              "method":"Read-only validation of real saved states, dimensions, null spaces, optimizer events, executed notebooks, embedded original figures and plot tables. No model call or numerical search."}
    write_json(args.output, report)
    print(json.dumps({"status":"passed", "completed_cases":len(cases), "actual_cma_generations":generation_count, "retained_unsuccessful_attempts":len(failures)}))


if __name__ == "__main__":
    main()
