"""Regenerate MI landscapes and optimization trajectories from the notebook source."""

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
DATASET_CASE_IDS = {"rayben": "rayleigh_benard", "presGrad": "pressure_gradient", "keyhole": "laser_keyhole", "blasius": "blasius"}
DATA_DIR = HERE.parents[1] / "data" / "datasets"


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
        case = cases[name]
        _, _, defaults = cells[name]
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
            "notebook_case_id": name,
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
        metadata_path = destination / "run_metadata.json"
        write_json(metadata_path, metadata)
        optimizers = []

        def optimizer_with_recorded_seed(initial_mean, step_size, options):
            options = {**options, "seed": args.seed}
            optimizer = original_optimizer(initial_mean, step_size, options)
            optimizers.append(optimizer)
            return optimizer

        def export_surface(history, idx_x, idx_y, grid_size):
            grid_x, grid_y, grid_mi, fixed_c = original_surface(history, idx_x, idx_y, grid_size)
            np.savez_compressed(
                destination / "response_surface.npz",
                grid_x=grid_x,
                grid_y=grid_y,
                mutual_information=grid_mi,
                fixed_coefficients=fixed_c,
                basis_indices=np.asarray([idx_x, idx_y]),
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
                prefix = destination / f"single_ksg_k_{config['mi_param']}"
                namespace["save_history_tables"](history, str(prefix))
                _, diagnostics = namespace["_choose_two_significant_basis_coefficients"](
                    history["summary"], history["basic_pi_names"], history["basic_pi_formulas"]
                )
                diagnostics.to_csv(destination / "basis_selection.csv", index=False)
                plot_info = namespace["plot_optimization_history"](
                    history,
                    case["input_vars"],
                    str(prefix),
                    plot_config=case["plot_config"],
                    response_surface_grid=args.grid_size,
                )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all",) + tuple(DATASET_CASE_IDS.values()), default="all")
    parser.add_argument("--notebook", type=Path, default=HERE / "MI_landscape_4cases.ipynb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=positive_integer, default=42)
    parser.add_argument("--maxiter", type=positive_integer, help="Explicitly override the notebook's optimization budget.")
    parser.add_argument("--popsize", type=positive_integer)
    parser.add_argument("--grid-size", type=positive_integer, default=101)
    args = parser.parse_args()
    if args.seed > 2**32 - 1:
        parser.error("--seed must be between 1 and 4294967295.")
    if args.grid_size < 15:
        parser.error("--grid-size must be at least 15, matching the notebook's minimum.")
    notebook_path = args.notebook.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = HERE / "generated" / "optimize"
    optimize(args, notebook_path, output_dir.resolve())


if __name__ == "__main__":
    main()
