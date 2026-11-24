"""Recompute regional IT-pi diagnostics from the recorded global fit, without APIs."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CSV_FILES = ("regional_cell_scores.csv", "regional_sensitivity.csv")
FIT_FILES = ("global_itpi_result.npz", "global_itpi_config.json", "global_itpi_console.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protected = (HERE / "results", HERE / "data", HERE / "vendor")
    if output == HERE or any(output.is_relative_to(path) for path in protected):
        parser.error("Choose a separate generated-output directory")
    output.mkdir(parents=True, exist_ok=True)
    original_hashes = {path.relative_to(HERE): hashlib.sha256(path.read_bytes()).hexdigest()
                       for folder in (HERE / "results", HERE / "data", HERE / "vendor")
                       for path in folder.rglob("*") if path.is_file() and path.suffix != ".pyc"}
    for name in FIT_FILES:
        copyfile(HERE / "results" / name, output / name)

    notebook = HERE / "reproduce_cess.ipynb"
    def provenance_path(path):
        if path.is_relative_to(HERE):
            return path.relative_to(HERE).as_posix()
        return (Path("generated") / path.relative_to(output)).as_posix()

    namespace = {"__file__": str(notebook), "__name__": "regime_notebook",
                 "provenance_path": provenance_path}
    cells = json.loads(notebook.read_text(encoding="utf-8-sig"))["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        # Keep figure provenance portable when generated files go outside the package.
        source = source.replace("str(p.relative_to(HERE)):sha(p)", "provenance_path(p):sha(p)")
        exec(compile(source, f"{notebook.name}:cell{index}", "exec"), namespace)
        if index == 1:
            namespace["OUT"] = output
            namespace["DIAG_OUT"] = output / "regional_count_sweep"
            namespace["DIAG_OUT"].mkdir(exist_ok=True)
        if index == 2 and namespace["REFIT_GLOBAL"]:
            raise RuntimeError("The offline diagnostic requires the recorded global fit")
        if index == 9 and args.no_plots:
            namespace["render_main"] = lambda: None
            namespace["render_diagnostic"] = lambda: None

    checks = []
    for name in CSV_FILES:
        actual = pd.read_csv(output / name, float_precision="round_trip")
        expected = pd.read_csv(HERE / "results" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-9, atol=1e-10)
        checks.append({"file": name, "rows": len(actual), "matches_snapshot": True})
    membership_file = "regional_memberships.npz"
    with np.load(output / membership_file) as actual, np.load(HERE / "results" / membership_file) as expected:
        if set(actual.files) != set(expected.files):
            raise ValueError("Regional membership archive keys differ from the snapshot")
        for name in expected.files:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        membership_arrays = len(actual.files)
    for relative, digest in original_hashes.items():
        if hashlib.sha256((HERE / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Recorded input or snapshot changed: {relative}")
    report = {"tables": checks, "membership_arrays": membership_arrays,
              "membership_arrays_match_exactly": True, "snapshots_unchanged": True,
              "relative_tolerance": 1e-9, "absolute_tolerance": 1e-10,
              "global_optimization_repeated": False,
              "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest()}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} diagnostic tables and {membership_arrays} membership arrays")
    print(output)


if __name__ == "__main__":
    main()
