# Velocity-transformation comparison

This directory reproduces Supplementary Section 5.3 from six compressible velocity profiles and the incompressible reference. The supplied notebook, input files, and recorded figures and tables are preserved.

Run from any directory with Python 3.12.

    python -m pip install -r requirements.txt
    python -B reproduce.py

Use --no-plots to recompute and verify only the tables. Use --output-dir PATH to choose a separate output directory. The command writes to generated/ by default, compares all four recomputed CSV tables against results/, and writes generated/verification.json. Recorded results are checked for changes before the command finishes.

The original notebook reproduce_tl.ipynb remains available. Its code computes the transformation and error definitions used here. Running its final cell writes to results/, so use the CLI for a separate verification run.

## Inputs and recorded results

- data/dataset_veloTrans.csv contains six native profiles.
- data/incom_ref.mat contains reference coordinates y_plus and reference velocity U_utau.
- results/incompressible_reference.csv exports those reference arrays as y_plus and U_plus for local visualization.
- results/transformed_native_profiles.csv contains native coordinates, velocities, density and viscosity ratios, and the TL, IT-pi, and AI Buckingham transformations.
- results/six_profile_pointwise_RMS.csv gives RMS velocity error over the six profiles at each common reference coordinate.
- results/normalized_RMS_intervals.csv gives each profile's spatial RMS error divided by the interval mean reference velocity for intervals [0,30] and [30,200].
- results/normalized_RMS_across_profiles.csv averages those normalized errors over the six profiles.
- results/TL_comparison.png and .pdf are the recorded comparison figure.

Profile identifiers 1 through 6 correspond to bulk Reynolds numbers 9993, 15491, 23978, 9979, 8430, and 18224 and bulk Mach numbers 1.7, 1.7, 3.0, 4.0, 1.5, and 1.5. The viscosity ratio named M in the data tables is mu/mu_w.

The CLI executes the supplied notebook's definitions and uses its original calculation and plotting functions. Table verification uses relative tolerance 1e-9 and absolute tolerance 1e-10. The tested environment regenerated all three figures across Sections 5.3 and 5.4 byte-identically as PNG files. Times New Roman is used when installed, with the notebook's built-in font fallback otherwise. The computation uses local files.
