"""Generate MI landscapes and optimization trajectories with the notebook functions."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CASE_NAMES = ("rayben", "presGrad", "keyhole", "blasius")
DATASET_CASE_IDS = {"rayben": "rayleigh_benard", "presGrad": "pressure_gradient", "keyhole": "laser_keyhole", "blasius": "blasius", "mhd": "mhd"}
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "datasets"
MHD_CONFIG = ROOT / "configs" / "cases" / "mhd.json"
MHD_BASIS = [[3, -2, 1, 0, 1, 0], [1, -0.5, 0, -0.5, 0, 1]]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def literal_assignment(source: str, name: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"Notebook does not define {name}.")


def read_notebook(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    source = "".join(notebook["cells"][0]["source"])
    cases = literal_assignment(source, "CASES")
    cells = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        if "ACTIVE_CASE =" in cell_source:
            case = literal_assignment(cell_source, "ACTIVE_CASE")
            cells[case] = (index, cell, literal_assignment(cell_source, "default_cma_config"))
    if set(cells) != set(CASE_NAMES) or set(cases) != set(CASE_NAMES):
        raise ValueError("Expected the four supplied notebook cases.")
    return notebook, source, cases, cells


def selected_cases(case: str):
    return CASE_NAMES if case == "all" else (next(name for name, public_id in DATASET_CASE_IDS.items() if public_id == case),)


def mhd_case():
    definition = json.loads(MHD_CONFIG.read_text(encoding="utf-8"))
    hyperparams = definition["manual_hyperparams"]
    return {
        "input_vars": definition["input_columns"],
        "output_var": definition["output_column"],
        "basis_matrices": MHD_BASIS,
        "cma_config": {
            "mi_method": hyperparams["mi_method_used"],
            "mi_param": hyperparams["mi_param_used"],
            "cma_init_position": hyperparams["cma_init_position"],
            "cma_init_stepsize": hyperparams["cma_init_stepsize"],
            "rational_max_denom": hyperparams["max_rational_denominator"],
            "basic_pi_names": ["Π_1", "Π_2"],
        },
        "plot_config": {"elev": 50, "azim": 50, "tick_pad": -2, "label_pad": -3},
        "basis_indices": (0, 1),
    }


def optimize(args, notebook_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd

    _, source, cases, cells = read_notebook(notebook_path)
    namespace = {"__name__": "mi_landscape_notebook_definitions"}
    exec(compile(source, str(notebook_path) + ":cell_0", "exec"), namespace)
    namespace["plt"].show = lambda: None
    original_optimizer = namespace["CMAEvolutionStrategy"]
    original_surface = namespace["_evaluate_conditional_response_surface"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in selected_cases(args.case):
        destination = output_dir / DATASET_CASE_IDS[name]
        destination.mkdir(parents=True, exist_ok=True)
        case = mhd_case() if name == "mhd" else cases[name]
        _, _, defaults = cells[CASE_NAMES[0] if name == "mhd" else name]
        config = {**defaults, **case["cma_config"]}
        if args.maxiter is not None:
            config["cma_maxiter"] = args.maxiter
        if args.popsize is not None:
            config["cma_popsize"] = args.popsize
        dataset_path = (DATA_DIR / f"{DATASET_CASE_IDS[name]}.csv").resolve()
        frame = pd.read_csv(dataset_path)
        clean = frame[case["input_vars"] + [case["output_var"]]].dropna()
        x_data = np.asarray(clean[case["input_vars"]], dtype=float)
        y_data = np.asarray(clean[[case["output_var"]]], dtype=float)
        metadata = {
            "mode": "new_notebook_optimization",
            "case_id": DATASET_CASE_IDS[name],
            "notebook_case_id": None if name == "mhd" else name,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "notebook_sha256": sha256_file(notebook_path),
            "dataset_name": dataset_path.name,
            "dataset_sha256": sha256_file(dataset_path),
            "rows_before_dropna": len(frame),
            "rows_after_dropna": len(clean),
            "input_variables": case["input_vars"],
            "output_variable": case["output_var"],
            "seed": args.seed,
            "notebook_source_seed": 42,
            "configuration": config,
            "basis_matrices": case["basis_matrices"],
            "plot_configuration": case["plot_config"],
            "response_surface_grid": args.grid_size,
            "budget_overrides": {key: value for key, value in {"maxiter": args.maxiter, "popsize": args.popsize}.items() if value is not None},
            "python_version": sys.version,
            "package_versions": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "cma", "matplotlib")},
        }
        if name == "mhd":
            metadata["case_configuration"] = MHD_CONFIG.relative_to(ROOT).as_posix()
            metadata["case_configuration_sha256"] = sha256_file(MHD_CONFIG)
            metadata["plot_basis_indices"] = list(case["basis_indices"])
            metadata["notebook_optimizer_tolfun"] = 1e-6
            metadata["trajectory_columns"] = ["bounded_mean_c_0", "bounded_mean_c_1"]
            metadata["trajectory_transform"] = "gp.pheno(mean, into_bounds=boundary_handler.repair)"
        metadata_path = destination / "run_metadata.json"
        write_json(metadata_path, metadata)
        optimizers = []
        bounded_mean_rows = []

        def optimizer_with_recorded_seed(initial_mean, step_size, options):
            options = {**options, "seed": args.seed}
            optimizer = original_optimizer(initial_mean, step_size, options)
            if name == "mhd":
                original_tell = optimizer.tell

                def tell_with_bounded_mean(*tell_args, **tell_kwargs):
                    result = original_tell(*tell_args, **tell_kwargs)
                    bounded_mean = optimizer.gp.pheno(
                        np.asarray(optimizer.mean, dtype=float).copy(),
                        into_bounds=optimizer.boundary_handler.repair,
                        copy=True,
                    )
                    bounded_mean_rows.append((int(optimizer.countiter), np.asarray(bounded_mean).copy()))
                    return result

                optimizer.tell = tell_with_bounded_mean
            optimizers.append(optimizer)
            return optimizer

        def export_surface(history, idx_x, idx_y, grid_size):
            grid_x, grid_y, grid_mi, fixed_c = original_surface(history, idx_x, idx_y, grid_size)
            axis_aliases = {"coefficient_x": grid_x[0, :], "coefficient_y": grid_y[:, 0]} if name == "mhd" else {}
            np.savez_compressed(
                destination / "response_surface.npz",
                grid_x=grid_x,
                grid_y=grid_y,
                mutual_information=grid_mi,
                fixed_coefficients=fixed_c,
                basis_indices=np.asarray([idx_x, idx_y]),
                **axis_aliases,
            )
            pd.DataFrame(
                {"coefficient_x": grid_x.ravel(), "coefficient_y": grid_y.ravel(), "mutual_information": grid_mi.ravel()}
            ).to_csv(destination / "response_surface.csv", index=False)
            return grid_x, grid_y, grid_mi, fixed_c

        namespace["CMAEvolutionStrategy"] = optimizer_with_recorded_seed
        namespace["_evaluate_conditional_response_surface"] = export_surface
        print(f"Optimizing {name}, seed {args.seed}, maxiter {config['cma_maxiter']}, population {config['cma_popsize']}", flush=True)
        try:
            with (destination / "console.txt").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                result = namespace["run_optimization"](
                    x_data, y_data, case["basis_matrices"], case["input_vars"], config
                )
                float_mi, _, rational_mi, _, strategy, raw, final, history = result
                plot_history = history
                if name == "mhd":
                    summary = history["summary"]
                    if summary["generation"].tolist() != [generation for generation, _ in bounded_mean_rows]:
                        raise ValueError("MHD trajectory generations do not match the optimizer history.")
                    bounded_means = np.asarray([mean for _, mean in bounded_mean_rows])
                    for index in range(2):
                        summary[f"bounded_mean_c_{index}"] = bounded_means[:, index]
                    plot_summary = summary.copy()
                    for index in range(2):
                        plot_summary[f"mean_c_{index}"] = summary[f"bounded_mean_c_{index}"]
                    plot_history = {**history, "summary": plot_summary}
                prefix = destination / f"single_ksg_k_{config['mi_param']}"
                namespace["save_history_tables"](history, str(prefix))
                _, diagnostics = namespace["_choose_two_significant_basis_coefficients"](
                    history["summary"], history["basic_pi_names"], history["basic_pi_formulas"]
                )
                diagnostics.to_csv(destination / "basis_selection.csv", index=False)
                plot_options = {"basis_indices": case["basis_indices"]} if name == "mhd" else {}
                plot_info = namespace["plot_optimization_history"](
                    plot_history,
                    case["input_vars"],
                    str(prefix),
                    plot_config=case["plot_config"],
                    response_surface_grid=args.grid_size,
                    **plot_options,
                )
                if name == "mhd":
                    figure = namespace["plt"].gcf()
                    for axis in figure.axes:
                        if axis.name == "3d" or axis.lines:
                            axis.set_xlim(config["cma_lower_bound"], config["cma_upper_bound"])
                            axis.set_ylim(config["cma_lower_bound"], config["cma_upper_bound"])
                        if axis.name == "3d":
                            ticks = [-1, -0.5, 0, 0.5, 1]
                            axis.set_xticks(ticks)
                            axis.set_yticks(ticks)
                            axis.set_yticklabels(["-1", "-0.5", "0", "0.5", ""])
                            axis.set_zlabel("MI", fontsize=12, labelpad=3)
                        if axis.name != "3d" and axis.lines:
                            axis.lines[0].set_label("CMA-ES trajectory")
                            axis.legend(loc="upper right", fontsize=11, facecolor="white", framealpha=1)
                    figure.axes[0].set_xlabel("MI", fontsize=11, labelpad=8)
                    figure.savefig(f"{prefix}_optimization_landscape.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
                final_mi = namespace["evaluate_c_mi"](
                    [float(value) for value in final], x_data, y_data,
                    history["basis_matrices"], history["mi_method"], history["mi_param"],
                )
            optimizer = optimizers[-1]
            write_json(
                destination / "optimization_result.json",
                {
                    "float_mi": float(float_mi),
                    "attempted_rational_mi": float(rational_mi),
                    "final_mi": float(final_mi),
                    "strategy": strategy,
                    "raw_coefficients": raw,
                    "final_coefficients": [str(value) for value in final],
                    "final_physical_exponents": namespace["calc_pi"]([float(value) for value in final], history["basis_matrices"]).tolist(),
                    "generations": int(optimizer.countiter),
                    "evaluations": int(optimizer.countevals),
                    "stop_reasons": {key: str(value) for key, value in optimizer.stop().items()},
                    "plot_info": plot_info,
                },
            )
            metadata["status"] = "complete"
            metadata["files"] = {
                path.name: sha256_file(path)
                for path in sorted(destination.iterdir())
                if path.is_file() and path != metadata_path
            }
        except Exception as error:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(metadata_path, metadata)
            namespace["plt"].close("all")
        print(f"Saved {name} optimization, history, numerical grid and plot to {destination}", flush=True)


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all",) + tuple(DATASET_CASE_IDS.values()), default="all")
    parser.add_argument("--notebook", type=Path, default=HERE / "MI_landscape_4cases.ipynb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=positive_integer, default=42)
    parser.add_argument("--maxiter", type=positive_integer, help="Explicitly override the notebook's optimization budget.")
    parser.add_argument("--popsize", type=positive_integer)
    parser.add_argument("--grid-size", type=positive_integer, default=101)
    args = parser.parse_args(argv)
    if args.seed > 2**32 - 1:
        parser.error("--seed must be between 1 and 4294967295.")
    if args.grid_size < 15:
        parser.error("--grid-size must be at least 15, matching the notebook's minimum.")
    notebook_path = args.notebook.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = HERE / "generated" / "optimize"
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(ROOT / "results"):
        for name in selected_cases(args.case):
            destination = output_dir / DATASET_CASE_IDS[name]
            if destination.exists() and any(destination.iterdir()):
                parser.error(f"Published results already exist at {destination}; choose another --output-dir.")
    optimize(args, notebook_path, output_dir)


if __name__ == "__main__":
    main()
