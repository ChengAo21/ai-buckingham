"""Compare seven original predictors with the learned groups on one saved split.

This is an explicit offline GPR refit using the existing validation routine.
It does not repeat discovery or change the selected representation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "sandgrain_prediction")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ai_buckingham.artifacts import read_json, sha256_file, write_json
    from ai_buckingham.optimization import global_coef_to_pi_mat
    from ai_buckingham.plotting import _load_visualization_frame
    from ai_buckingham.validation import evaluate_gpr_collapse, resolve_gpr_validation_options
    runs = sorted((ROOT / "results" / "cases" / "sandgrain").glob("*/manifest.json"), reverse=True)
    selected = args.run_dir
    if selected is None:
        selected = next((p.parent for p in runs if read_json(p).get("status") == "completed"), None)
    if selected is None:
        raise FileNotFoundError("A completed Sandgrain run is required.")
    selected = selected.resolve()
    state = read_json(selected / "state.json")
    local = next(local for local in state["workflow_result"]["local_workflows"] if local.get("status") == "success")
    final = local["final_solution"]
    fixed_split = final["gpr_validation"].get("fixed_split")
    if fixed_split is None:
        raise ValueError("The comparison requires the exact recorded source-stratified split.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Comparison already exists; use another output directory.")
    config = read_json(ROOT / "configs" / "cases" / "sandgrain.json")
    source = ROOT / config["dataset"]
    raw = pd.read_csv(source, float_precision="round_trip")
    transformed = _load_visualization_frame(source, state["workflow_result"])
    columns = state["input_vars"]
    raw_predictors = raw[columns].to_numpy(dtype=float)
    X = transformed[columns].to_numpy(dtype=float)
    learned = np.column_stack([global_coef_to_pi_mat(np.asarray(row), X).ravel() for row in final["input_physical_exponents"]])
    target = raw[state["output_var"]].to_numpy(dtype=float)
    options = resolve_gpr_validation_options(state["manual_hyperparams"])
    reports, models = {}, {}
    for label, coordinates in [("original_seven", raw_predictors), ("learned_groups", learned)]:
        captured = []
        report = evaluate_gpr_collapse(coordinates, target,
            max_samples=options["gpr_max_samples"], repeats=options["gpr_n_splits"],
            n_restarts_optimizer=options["gpr_n_restarts_optimizer"], log_features=options["gpr_log_features"],
            log_target=options["gpr_log_target"], constant_value_bounds=options["gpr_constant_value_bounds"],
            length_scale_bounds=options["gpr_length_scale_bounds"], noise_level_bounds=options["gpr_noise_level_bounds"],
            r2_threshold=options["gpr_r2_threshold"], test_size=options["gpr_test_size"], random_state=options["gpr_random_state"],
            fixed_split=fixed_split, on_first_fit=captured.append)
        if report["status"] != "success" or not captured:
            raise RuntimeError("GPR comparison fit did not complete.")
        model, train, test, seed, log_x, log_y = captured[0]
        predictions = model.predict(np.log(coordinates) if log_x else coordinates).ravel()
        if log_y:
            predictions = np.exp(predictions)
        reports[label] = report
        models[label] = predictions
    agreement = bool(np.isclose(reports["learned_groups"]["test_r2_mean"], final["gpr_validation"]["test_r2_mean"], rtol=1e-8, atol=1e-8))
    split = np.full(len(raw), "train", dtype=object)
    split[np.asarray(fixed_split["test_rows"], dtype=int)] = "test"
    predictions = pd.DataFrame({"csv_row": np.arange(len(raw)), "source": raw["case"], "split": split,
                                "reference": target, **models})
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        max_value = max(float(target.max()), *(float(v.max()) for v in models.values())) * 1.04
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
        for ax, (label, values), title in zip(axes, models.items(), ["Seven original surface statistics", f"{learned.shape[1]} learned dimensionless groups"]):
            for i, source_label in enumerate(pd.unique(raw["case"])):
                mask = raw["case"].to_numpy() == source_label
                for part, filled in [("train", False), ("test", True)]:
                    rows = mask & (split == part)
                    ax.scatter(target[rows], values[rows], marker=markers[i % len(markers)], s=42,
                               facecolors="#087d93" if filled else "none", edgecolors="#087d93" if filled else "#a0adb6",
                               linewidths=.9, alpha=.85, label=f"{source_label} · {part}")
            xx = np.linspace(0, max_value, 100)
            ax.plot(xx, xx, color="#1f3442", linewidth=1)
            ax.plot(xx, 1.35*xx, ":", color="#8a9aa4")
            ax.plot(xx, .65*xx, ":", color="#8a9aa4")
            ax.set(xlim=(0,max_value), ylim=(0,max_value), xlabel="Reference ks/Ra", ylabel="Predicted ks/Ra",
                   title=f"{title}\nHeld-out R² = {reports[label]['test_r2_mean']:.3f}")
            ax.grid(alpha=.16)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"sandgrain_prediction.{extension}", dpi=250, bbox_inches="tight")
        plt.close(fig)
    write_json(args.output_dir / "manifest.json", {"mode": "offline_validation_refit", "source_run": selected.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256_file(source), "original_features": columns, "learned_group_count": learned.shape[1],
        "train_count": int(np.count_nonzero(split == "train")), "test_count": int(np.count_nonzero(split == "test")),
        "fixed_split": fixed_split, "gpr_options": options, "validation": reports,
        "learned_r2_matches_saved_run": agreement,
        "method": "Both representations use the existing evaluate_gpr_collapse implementation and the identical recorded source-stratified train/test rows. Raw predictors use unshifted original Sk; learned groups use the declared preprocessing snapshot.",
        "files": {p.name: sha256_file(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({"status": "completed", "learned_r2_matches_saved_run": agreement,
                      "scores": {k:v["test_r2_mean"] for k,v in reports.items()}}))


if __name__ == "__main__":
    main()
