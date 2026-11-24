# Regional comparison with IT-pi

This directory reproduces Supplementary Section 5.4 from the recorded global IT-pi fit and the recorded gradient-based partition of the Cess dataset. The author's vendor/IT_PI.py and license, the supplied notebook, data, and result snapshots are preserved.

Run from any directory with Python 3.12.

    python -m pip install -r requirements.txt
    python -B reproduce.py

Use --no-plots to recompute and verify only the diagnostic tables and membership arrays. Use --output-dir PATH to choose a separate output directory. The command copies the recorded global fit into generated/, runs the notebook's regional calculations, and produces both comparison figures there. It compares two CSV tables and all 100 membership arrays with results/, then writes generated/verification.json. Recorded inputs and results are checked for changes before the command finishes.

The supplied reproduce_cess.ipynb remains available. Its default REFIT_GLOBAL=False uses the recorded fit. Running its cells directly writes to results/, so use the CLI for a separate verification run. Repeating global optimization can be computationally expensive and changes the calculation being compared with the recorded regional diagnostics.

## Calculation and data

The cached global fit contains two input dimensionless groups from the author's IT-pi implementation. Regions are fitted with KMeans to StandardScaler(log1p(input_PI)). Regional mutual-information estimates are computed for both raw values and globally standardized logarithms of the inputs and response. Each condition is evaluated for region counts M=2,3,5,8,10 and seeds 42 through 46, giving 50 runs. The main figure uses standardized logarithms, M=3 and 5, and seed 42. Local classification uses the author's estimator with k=5 and threshold 0.9. Negative estimates and ratios outside [0,1] remain in the diagnostics.

- data/dataset_cess_channel.csv contains 1,000 observations.
- data/recorded_gradient_membership.csv gives the recorded gradient partition and physical plotting coordinates Re_tau, eta=y/h, and y_plus.
- results/global_itpi_result.npz contains the recorded global dimensionless groups and coefficients.
- results/regional_cell_scores.csv gives cell-level MI values, MI ratios, categories, counts, and overlap with the recorded intermediate group.
- results/regional_sensitivity.csv summarizes all 50 region-count, seed, and conditioning combinations.
- results/regional_memberships.npz stores pointwise region and category assignments. Keys follow conditioning__M<count>__seed<seed>__region or __category. Category codes are X1=0, X2=1, Both=2.
- results/figure_settings.json records figure controls and calculation settings.
- results/Cess_ITPI_comparison.png and .pdf are the main comparison.
- results/regional_count_sweep/ contains the region-count diagnostic figure.

The recorded gradient groups contain 779 near-wall samples, 114 intermediate samples, and 107 outer samples. The intermediate group's eta range is approximately 0.13005 to 0.35561.

CSV verification uses relative tolerance 1e-9 and absolute tolerance 1e-10. Membership arrays must agree exactly. The cached global fit is reused, so this entry point performs regional MI calculations and plotting without repeating the CMA-ES global fit. The computation uses local files.
