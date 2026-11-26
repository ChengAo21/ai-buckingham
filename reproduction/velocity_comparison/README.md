# Relation to the Trettel–Larsson transformation

This package provides the six-profile velocity-transformation comparison in SI §5.3, using four compressible channel-flow profiles, two pipe-flow profiles and an incompressible reference.

Use Python 3.12 and run from this directory:

```bash
python -m pip install -r requirements.txt
python -B reproduce.py
```

Use `--no-plots` to recompute the tables, or `--output-dir PATH` to choose a separate output directory. The command writes to `generated/`, compares four recomputed CSV tables with the included results and writes `generated/verification.json`.

The notebook `reproduce_tl.ipynb` contains the transformation and error calculations. Its final cell writes to `results/`; use the CLI for a separate reproduction run.

## Inputs and outputs

| Path | Contents |
|---|---|
| `data/dataset_veloTrans.csv` | Six compressible velocity profiles |
| `data/incom_ref.mat` | Incompressible reference coordinates y⁺ and velocity U⁺ |
| `results/incompressible_reference.csv` | The reference arrays as `y_plus` and `U_plus` |
| `results/transformed_native_profiles.csv` | Profile coordinates, velocities, density/viscosity ratios and TL, IT–π and AI Buckingham transformations |
| `results/six_profile_pointwise_RMS.csv` | RMS velocity error across the six profiles at each common reference coordinate |
| `results/normalized_RMS_intervals.csv` | Per-profile normalized RMS errors for [0,30] and [30,200] |
| `results/normalized_RMS_across_profiles.csv` | Mean normalized errors across the six profiles |
| `results/TL_comparison.png`, `.pdf` | Comparison figure |

Profile IDs 1–6 correspond to bulk Reynolds numbers 9993, 15491, 23978, 9979, 8430 and 18224, and bulk Mach numbers 1.7, 1.7, 3.0, 4.0, 1.5 and 1.5. The viscosity ratio `M` is μ/μw.

The RMS definitions follow SI Eq. (82). Table verification uses relative tolerance 1e-9 and absolute tolerance 1e-10.
