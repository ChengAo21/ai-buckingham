"""Recalculate missing candidate memberships from saved gradients, without a GPR refit.

The supplement is labeled as offline recalculation and is kept outside the
original run. Counts, objectives and any selected labels must match the record.
"""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    from ai_buckingham.regimes import _cluster_by_gradient_similarity
    from ai_buckingham.artifacts import read_json, write_json, sha256_file
    for manifest_path in sorted((ROOT / "results" / "cases").glob("*/*/manifest.json")):
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        run_dir = manifest_path.parent
        if (run_dir / "evidence" / "regime_candidates").exists():
            continue
        state = read_json(run_dir / "state.json")
        detection = (state.get("workflow_result") or {}).get("regime_detection") or {}
        candidates = detection.get("candidate_summary") or []
        if not candidates or not detection.get("surrogate_gradients"):
            continue
        destination = ROOT / "results" / "regime_candidates" / manifest["case_id"] / manifest["run_id"]
        if (destination / "manifest.json").exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        gradients = np.asarray(detection["surrogate_gradients"], dtype=float)
        resolved = state["compiled_workflow"]["resolved_gradient_regime_options"]["clustering"]
        allowed = set(inspect.signature(_cluster_by_gradient_similarity).parameters) - {"gradients", "n_clusters"}
        options = {k:v for k,v in resolved.items() if k in allowed}
        np.savez_compressed(destination / "coordinates.npz", **{k:np.asarray(detection[k]) for k in ["feature_coordinates", "surrogate_gradients", "schmidt_basis", "valid_indices"]})
        checks = []
        for candidate in candidates:
            k = int(candidate["k"])
            result = _cluster_by_gradient_similarity(gradients, k, **options)
            if not result or result.get("status") != "success":
                raise RuntimeError(f"Candidate K={k} did not recalculate successfully")
            labels = np.asarray(result["labels"], dtype=int)
            counts = [int(np.count_nonzero(labels == i)) for i in range(k)]
            objective_agrees = bool(np.isclose(result["objective"], candidate["clustering_objective"], rtol=1e-10, atol=1e-10))
            counts_agree = counts == candidate["sample_counts"]
            selected_labels_agree = None
            if detection.get("best_k") == k and detection.get("labels"):
                selected_labels_agree = bool(np.array_equal(labels, detection["labels"]))
            if not objective_agrees or not counts_agree or selected_labels_agree is False:
                raise ValueError("Recalculated memberships do not match the recorded diagnostics")
            similarities = np.asarray(result["similarities"], dtype=float)
            sorted_similarity = np.sort(similarities, axis=1)
            confidence = (sorted_similarity[:, -1] - sorted_similarity[:, -2]) / np.maximum(np.abs(sorted_similarity[:, -1]), 1e-15)
            np.savez_compressed(destination / f"k_{k}.npz", labels=labels, assignment_confidence=confidence)
            write_json(destination / f"k_{k}.json", candidate)
            checks.append({"k": k, "objective_matches": objective_agrees, "sample_counts_match": counts_agree,
                           "selected_labels_match": selected_labels_agree})
        write_json(destination / "manifest.json", {"mode": "offline_recalculation_from_recorded_gradients",
             "source_run": run_dir.relative_to(ROOT).as_posix(), "source_state_sha256": sha256_file(run_dir / "state.json"),
             "candidate_counts": [c["k"] for c in candidates], "validation": checks,
             "method": "Same clustering function and resolved options applied to recorded gradients. No surrogate refit, model call, or discovery optimization.",
             "files": {p.name:sha256_file(p) for p in sorted(destination.iterdir()) if p.is_file()}})
        print(json.dumps({"case":manifest["case_id"], "status":"validated_offline_recalculation", "candidates":len(checks)}))


if __name__ == "__main__":
    main()
