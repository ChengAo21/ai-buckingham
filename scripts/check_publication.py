"""Check saved case evidence against the authors' reported numerical tables.

This command only reads saved results. The expected values are acceptance
criteria and are never used by the numerical tools or model prompts.
"""
from pathlib import Path
import argparse
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def follow(value, path):
    for part in path.split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def stage_metrics(stage):
    subsampling = stage.get("mi_score_subsampling_uq") or {}
    return {
        "mutual_information": stage.get("mutual_information"),
        "normalized_information_score": stage.get("normalized_information_score"),
        "conditional_test_r2": (stage.get("gpr_validation") or {}).get("test_r2_mean"),
        "sigma_mi_combined": math.hypot(
            stage.get("restart_mutual_information_std") or 0.0,
            subsampling.get("subsample_mi_std") or 0.0,
        ),
        "sigma_score_combined": math.hypot(
            stage.get("restart_information_score_std") or 0.0,
            subsampling.get("subsample_score_std") or 0.0,
        ),
    }


def check_cases(root=ROOT):
    root = Path(root)
    specification = read_json(root / "configs" / "publication_expectations.json")
    tolerance = specification["scalar_absolute_tolerance"]
    index = {case["case_id"]: case for case in read_json(root / "results" / "index.json")["cases"]}
    cases = []
    for expected in specification["cases"]:
        case_id = expected["case_id"]
        entry = index[case_id]
        run = next(record for record in entry["runs"] if record["run_id"] == entry["selected_run"])
        state = read_json(root / run["directory"] / "state.json")
        numerical = state["workflow_result"]
        checks = []

        def compare(name, actual, reference, scalar=False):
            passed = (
                isinstance(actual, (int, float))
                and math.isfinite(actual)
                and abs(actual - reference) <= tolerance
            ) if scalar else actual == reference
            checks.append({"quantity": name, "expected": reference, "actual": actual, "passed": bool(passed)})

        compare("included run status", run["status"], "completed")
        compare("included attempts", len(entry["runs"]), 1)
        regime = numerical["regime_detection"]
        compare("surrogate training samples", regime["surrogate"]["n_train_samples"], expected["regime_training_samples"])
        candidates = {str(item["k"]): item for item in regime["candidate_summary"]}
        for count, samples in expected["candidate_counts"].items():
            candidate = candidates.get(count, {})
            compare(f"K={count} partition sample counts", sorted(candidate.get("sample_counts", [])), samples)

        for local_expected in expected["local_workflows"]:
            samples = local_expected["samples"]
            locals_with_count = [local for local in numerical["local_workflows"] if local["n_samples"] == samples]
            if len(locals_with_count) != 1:
                compare(f"unique local scope with {samples} samples", len(locals_with_count), 1)
                continue
            local = locals_with_count[0]
            prefix = f"{samples} samples"
            compare(f"{prefix}: similarity type", local["similarity_type"], local_expected["similarity_type"])
            compare(f"{prefix}: retained group count", local["locked_count"], local_expected["group_count"])
            for stage_expected in local_expected["stages"]:
                values = stage_metrics(follow(local, stage_expected["path"]))
                for quantity, reference in stage_expected["expected"].items():
                    compare(f"{prefix}: {stage_expected['stage']}: {quantity}", values[quantity], reference, scalar=True)
        failures = [check for check in checks if not check["passed"]]
        cases.append({"case_id": case_id, "run_id": entry["selected_run"],
                      "supplementary_table": expected["supplementary_table"],
                      "status": "passed" if not failures else "failed", "checks": checks, "failures": failures})
    return {"status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
            "cases": cases, "checks": sum(len(case["checks"]) for case in cases),
            "method": "Offline comparison of actual saved numerical results with author table values at the reported three-decimal precision. Candidate labels may be permuted. No API, optimizer, fitting or rewriting of results.",
            "scope": "Reported MI, epsilon, combined uncertainty, R2, group counts, similarity types, candidate partition counts and surrogate training counts. This does not assert bitwise equality of every floating-point intermediate or identical LLM prose."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the acceptance report under docs/validation")
    args = parser.parse_args()
    report = check_cases()
    if args.write:
        target = ROOT / "docs" / "validation" / "publication_alignment.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "checks": report["checks"],
                      "failed_cases": [case["case_id"] for case in report["cases"] if case["status"] != "passed"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
