"""Export the MHD two-coefficient MI landscape and its actual recorded trajectory."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ai_buckingham.artifacts import read_json, write_json, sha256_file
    from ai_buckingham.optimization import global_calc_MI, global_coef_to_pi_mat
    from ai_buckingham.plotting import _load_visualization_frame
    runs = sorted((ROOT / "results" / "cases" / "mhd").glob("*/manifest.json"), reverse=True)
    run_dir = next(p.parent for p in runs if read_json(p).get("status") == "completed")
    output = ROOT / "results" / "mi_landscapes" / "mhd"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_metadata.json").exists():
        raise FileExistsError("MHD landscape already exists")
    state = read_json(run_dir / "state.json")
    records = [json.loads(line) for line in (run_dir / "evidence" / "optimization_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r.get("event") == "optimizer_generation" and r.get("phase") == "local_model_selection" and r.get("pi_count") == 1 and not r.get("optimize_output") and r.get("seed") == 42 and len(r.get("basis_matrix", [])) == 2]
    if not records:
        raise ValueError("An actual primary-seed, pre-sparsification MHD trace is required")
    search = records[0].get("search_id")
    records = [r for r in records if r.get("search_id") == search]
    basis = np.asarray(records[0]["basis_matrix"], dtype=float)
    if not np.allclose(records[0]["mean_coefficients"], records[0]["mean_input_basis_coefficients"][0], rtol=1e-12, atol=1e-12):
        raise ValueError("This exporter expects the recorded unnormalized two-coefficient representation")
    source = ROOT / read_json(ROOT / "configs" / "cases" / "mhd.json")["dataset"]
    frame = _load_visualization_frame(source, state["workflow_result"])
    X = frame[state["input_vars"]].to_numpy(dtype=float)
    Y = frame[[state["output_var"]]].to_numpy(dtype=float)
    hp = state["manual_hyperparams"]
    method, parameter = hp["mi_method_used"], int(hp["mi_param_used"])
    grid = np.linspace(-1, 1, 101)
    values = np.empty((len(grid), len(grid)))
    for iy, c2 in enumerate(grid):
        for ix, c1 in enumerate(grid):
            pi = global_coef_to_pi_mat(np.asarray([c1,c2]) @ basis, X)
            values[iy,ix] = global_calc_MI(pi, Y, method, parameter)
    xx, yy = np.meshgrid(grid, grid)
    pd.DataFrame({"coefficient_x":xx.ravel(),"coefficient_y":yy.ravel(),"mutual_information":values.ravel()}).to_csv(output / "response_surface.csv", index=False)
    np.savez_compressed(output / "response_surface.npz", coefficient_x=grid, coefficient_y=grid, mutual_information=values)
    history = pd.DataFrame({"generation":[r["generation"] for r in records],
        "global_best_mi":[-r["best_objective"] if r["best_objective"] < 1e4 else np.nan for r in records],
        "sigma":[r["sigma"] for r in records],
        **{f"mean_c_{j}":[r["mean_coefficients"][j] for r in records] for j in range(2)},
        **{f"best_c_{j}":[r["best_coefficients"][j] for r in records] for j in range(2)}})
    history.to_csv(output / "generation_history.csv", index=False)
    fig = plt.figure(figsize=(11,4.7), constrained_layout=True)
    ax = fig.add_subplot(121)
    contour = ax.contourf(xx, yy, values, levels=25, cmap="viridis")
    ax.plot(history.mean_c_0, history.mean_c_1, color="white", linewidth=1, alpha=.8)
    ax.scatter(history.mean_c_0.iloc[0], history.mean_c_1.iloc[0], color="white", edgecolors="#17202a", label="Recorded start")
    ax.scatter(history.best_c_0.iloc[-1], history.best_c_1.iloc[-1], color="#ffd166", edgecolors="#17202a", marker="*", s=130, label="Recorded best")
    ax.set(xlabel="c1",ylabel="c2",title="MHD pre-sparsification MI landscape")
    ax.legend(fontsize=8)
    fig.colorbar(contour, ax=ax, label="Estimated MI")
    surface = fig.add_subplot(122, projection="3d")
    surface.plot_surface(xx, yy, values, cmap="viridis", linewidth=0, alpha=.9)
    surface.set(xlabel="c1",ylabel="c2",zlabel="Estimated MI",title="Same numerical grid")
    for extension in ["png","pdf","svg"]:
        fig.savefig(output / f"mhd_landscape.{extension}",dpi=230,bbox_inches="tight")
    plt.close(fig)
    write_json(output / "run_metadata.json", {"mode":"offline_landscape_evaluation_with_recorded_trajectory", "case_id":"mhd", "status":"complete",
        "source_run":run_dir.relative_to(ROOT).as_posix(),"source_state_sha256":sha256_file(run_dir / "state.json"),
        "dataset_sha256":sha256_file(source),"input_variables":state["input_vars"],"basis_matrices":basis.tolist(),
        "mi_method":method,"mi_parameter":parameter,"response_surface_grid":101,"grid_bounds":[-1,1],
        "trajectory_search_id":search,"trajectory_generations":len(records),"trajectory_mode":"actual recorded primary-seed search, before sparsification",
        "method":"Fresh pointwise MI evaluation on the full two-coefficient input basis. The recorded optimizer trajectory is overlaid without running another search. Finite negative MI estimates are retained.",
        "files":{p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({"case":"mhd","grid_points":values.size,"actual_generations":len(records)}))


if __name__ == "__main__":
    main()
