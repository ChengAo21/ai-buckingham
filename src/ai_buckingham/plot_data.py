"""Export the coordinates used by existing figures without refitting a model."""
from __future__ import annotations
from pathlib import Path
import re


def export_plot_data(state: dict, run_dir: Path) -> dict:
    import numpy as np
    import pandas as pd
    from .artifacts import repository_root, sha256_file, write_json
    from .optimization import global_coef_to_pi_mat
    from .plotting import _load_visualization_frame
    run_dir = Path(run_dir)
    manifest_path = run_dir / "evidence" / "plot_data_manifest.json"
    if manifest_path.exists():
        import json
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(state["csv_FilePath"])
    if not source.is_absolute():
        source = repository_root() / source
    result = state.get("workflow_result") or {}
    frame = _load_visualization_frame(source, result)
    X = frame[state["input_vars"]].to_numpy(dtype=float)
    Y = frame[state["output_var"]].to_numpy(dtype=float)
    written, warnings = [], []
    def save(name, data):
        path = run_dir / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        written.append({"file": f"tables/{name}", "sha256": sha256_file(path), "rows": len(data)})
    for local in result.get("local_workflows", []):
        if local.get("status") != "success":
            continue
        scope = re.sub(r"[^a-zA-Z0-9_]+", "_", str(local.get("scope_id", "global")))
        try:
            indices = np.asarray(local.get("original_csv_indices", local.get("global_sample_indices", [])), dtype=int)
            indices = indices[(indices >= 0) & (indices < len(X))]
            final = local.get("final_solution") or {}
            exponents = np.asarray(final.get("input_physical_exponents", []), dtype=float)
            coordinates = {f"pi_{i+1}": global_coef_to_pi_mat(row, X[indices]).ravel() for i, row in enumerate(exponents)}
            scale = np.ones(len(indices))
            if local.get("optimize_output") and final.get("output_physical_exponents") is not None:
                scale = global_coef_to_pi_mat(np.asarray(final["output_physical_exponents"], dtype=float), X[indices]).ravel()
            data = pd.DataFrame({"csv_row": indices, **coordinates, "response_original": Y[indices],
                                 "output_scale": scale, "response_transformed": Y[indices] * scale})
            if "case" in frame:
                data["source_case"] = frame.iloc[indices]["case"].to_numpy()
            save(f"{scope}_collapse.csv", data)
            surface = local.get("gpr_surface")
            if surface:
                xy = np.asarray(surface["coordinates"], dtype=float)
                save(f"{scope}_gpr_surface.csv", pd.DataFrame({"pi_1": xy[:, 0], "pi_2": xy[:, 1], "prediction": surface["predictions"]}))
                save(f"{scope}_gpr_triangles.csv", pd.DataFrame(surface["triangles"], columns=["vertex_1", "vertex_2", "vertex_3"]))
                for name in ("train_observations", "test_observations"):
                    save(f"{scope}_{name}.csv", pd.DataFrame(surface[name], columns=["pi_1", "pi_2", "response"]))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            warnings.append({"scope": scope, "error_type": type(exc).__name__})
    detection = result.get("regime_detection") or {}
    if detection.get("feature_coordinates"):
        xy, gradient = np.asarray(detection["feature_coordinates"]), np.asarray(detection["surrogate_gradients"])
        if xy.shape == gradient.shape and xy.ndim == 2:
            fields = {f"z_{i+1}": xy[:, i] for i in range(xy.shape[1])}
            fields.update({f"gradient_z_{i+1}": gradient[:, i] for i in range(gradient.shape[1])})
            rows = detection.get("original_csv_indices")
            if rows is not None and len(rows) == len(xy):
                fields["csv_row"] = rows
            else:
                fields["numerical_row"] = detection.get("valid_indices", list(range(len(xy))))
            if len(detection.get("labels", [])) == len(xy):
                fields["regime"] = detection["labels"]
            save("gradient_field.csv", pd.DataFrame(fields))
    manifest = {"mode": "derived_plot_coordinates_from_saved_state", "source_csv_sha256": sha256_file(source),
                "method": "Same stored exponents, row mappings, preprocessing snapshot and power-product evaluator as the original plotting routines; no model refit or optimizer search.",
                "files": written, "warnings": warnings}
    write_json(manifest_path, manifest)
    return manifest
