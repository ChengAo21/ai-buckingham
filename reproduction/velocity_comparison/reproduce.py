"""Recompute the velocity-transformation comparison without changing its snapshots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
TABLES = (
    "transformed_native_profiles.csv",
    "six_profile_pointwise_RMS.csv",
    "normalized_RMS_intervals.csv",
    "normalized_RMS_across_profiles.csv",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == HERE or output.is_relative_to(HERE / "results") or output.is_relative_to(HERE / "data"):
        parser.error("Choose a separate generated-output directory")
    output.mkdir(parents=True, exist_ok=True)
    original_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (HERE / "results").iterdir() if path.is_file()}

    notebook = HERE / "reproduce_tl.ipynb"
    namespace = {"__file__": str(notebook), "__name__": "velocity_notebook"}
    previous = Path.cwd()
    try:
        os.chdir(HERE)
        for index, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8-sig"))["cells"]):
            source = "".join(cell.get("source", []))
            if cell["cell_type"] == "code" and source.strip() != "main()":
                exec(compile(source, f"{notebook.name}:cell{index}", "exec"), namespace)
    finally:
        os.chdir(previous)

    native, reference, rms, normalized = namespace["calculate"](HERE / "data")
    means = normalized.groupby(["y_plus_min", "y_plus_max", "method"], as_index=False)[
        "normalized_RMS_percent"
    ].mean().rename(columns={
        "y_plus_min": "a", "y_plus_max": "b", "normalized_RMS_percent": "mean_normalized_RMS_percent",
    })
    tables = dict(zip(TABLES, (native, rms, normalized, means), strict=True))
    for name, table in tables.items():
        table.to_csv(output / name, index=False)

    ref = loadmat(HERE / "data" / "incom_ref.mat")
    pd.DataFrame({"y_plus": ref["y_plus"].ravel(), "U_plus": ref["U_utau"].ravel()}).to_csv(
        output / "incompressible_reference.csv", index=False,
    )
    if not args.no_plots:
        namespace["render_final_comparison"](native, reference, rms, normalized, output / "TL_comparison")

    checks = []
    for name in TABLES:
        actual = pd.read_csv(output / name, float_precision="round_trip")
        expected = pd.read_csv(HERE / "results" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-9, atol=1e-10)
        checks.append({"file": name, "rows": len(actual), "matches_snapshot": True})
    for name, digest in original_hashes.items():
        if hashlib.sha256((HERE / "results" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Recorded snapshot changed: {name}")
    report = {"tables": checks, "relative_tolerance": 1e-9, "absolute_tolerance": 1e-10,
              "snapshots_unchanged": True, "profile_count": int(native["case"].nunique()),
              "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest()}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} tables for {report['profile_count']} profiles against recorded snapshots")
    print(output)


if __name__ == "__main__":
    main()
