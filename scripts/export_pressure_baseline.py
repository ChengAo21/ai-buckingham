"""Export the Reynolds-number baseline from the supplied wall-friction data.

This is a direct coordinate calculation and plot, not a model fit or a fresh
discovery run. Run from any directory; it never modifies the source CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
CLASS_LABELS = {"zeroPG": "ZPG", "advPG": "APG", "favPG": "FPG"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, output: Path) -> dict:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, output = source.resolve(), output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("The baseline bundle already exists; choose another --output-dir.")
    frame = pd.read_csv(source)
    required = ["case", "U_inf", "delta", "nu", "U_tau/U_inf"]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    unknown = set(frame["case"].dropna().astype(str)) - set(CLASS_LABELS)
    if unknown or frame["case"].isna().any():
        raise ValueError(f"Unknown or missing pressure-gradient labels: {sorted(unknown)}")
    numeric = frame[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (numeric > 0).all():
        raise ValueError("All baseline coordinates must be finite and positive; no rows are silently removed.")
    points = pd.DataFrame({
        "csv_row": np.arange(len(frame), dtype=int),
        "source_case": frame["case"],
        "pressure_gradient_class": frame["case"].map(CLASS_LABELS),
        "U_inf": frame["U_inf"], "delta": frame["delta"], "nu": frame["nu"],
        "Re_delta": frame["U_inf"] * frame["delta"] / frame["nu"],
        "U_tau_over_U_inf": frame["U_tau/U_inf"],
    })
    output.mkdir(parents=True, exist_ok=True)
    points.to_csv(output / "baseline_points.csv", index=False)
    style = {"ZPG": ("o", "#245B78"), "APG": ("^", "#C17837"), "FPG": ("s", "#209B88")}
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 11,
                         "axes.labelsize": 14, "axes.linewidth": .9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.hashsalt": "ai-buckingham-pressure-baseline"}):
        fig, ax = plt.subplots(figsize=(6.4, 4.3), layout="constrained")
        for label in CLASS_LABELS.values():
            group = points[points.pressure_gradient_class == label]
            marker, color = style[label]
            ax.scatter(group.Re_delta, group.U_tau_over_U_inf, s=36, marker=marker,
                       facecolors="none", edgecolors=color, linewidths=1.15,
                       label=f"{label} (n={len(group)})", zorder=3)
        ax.set(xscale="log", xlabel=r"$Re_{\delta}=U_{\infty}\delta/\nu$",
               ylabel=r"$u_{\tau}/U_{\infty}$")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(which="major", color="#d9e1e9", linewidth=.7, alpha=.8)
        ax.set_axisbelow(True)
        ax.margins(x=.05, y=.09)
        ax.legend(loc="upper right", frameon=True, edgecolor="#d9e1e9", fontsize=10)
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(output / f"pressure_gradient_baseline.{extension}", dpi=300,
                        metadata={"CreationDate": None} if extension == "pdf" else None)
        plt.close(fig)
    def path_text(path):
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)
    counts = points.groupby("pressure_gradient_class").size().to_dict()
    manifest = {
        "schema_version": 1,
        "mode": "offline_direct_coordinate_evaluation",
        "status": "complete", "case_id": "pressure_gradient",
        "description": "Reynolds-number baseline for the pressure-gradient wall-friction observations, corresponding to Supplementary Figure 5.",
        "source_dataset": path_text(source), "source_dataset_sha256": sha256(source),
        "script": "scripts/export_pressure_baseline.py", "script_sha256": sha256(Path(__file__)),
        "coordinate_definitions": {"Re_delta": "U_inf * delta / nu", "response": "U_tau/U_inf"},
        "source_to_display_label": CLASS_LABELS,
        "label_policy": "Display labels translate the existing case values only; no new classification is inferred.",
        "rows": len(points), "excluded_rows": 0, "csv_row_index_base": 0,
        "class_counts": counts,
        "included_dataset_count_check": counts == {"ZPG": 84, "APG": 36, "FPG": 24},
        "calculation": "Direct coordinate calculation on every supplied row. No optimization, model fit, interpolation or API call.",
        "packages": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/datasets/pressure_gradient.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pressure_gradient_baseline")
    args = parser.parse_args()
    manifest = export(args.input, args.output_dir)
    print(json.dumps({"output": str(args.output_dir.resolve()), "rows": manifest["rows"],
                      "class_counts": manifest["class_counts"]}))


if __name__ == "__main__":
    main()
