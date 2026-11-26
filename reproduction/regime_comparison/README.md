# Physical-regime detection in IT–π and AI Buckingham

This package provides the calculations and figures for SI §5.4. Inputs include a global IT–π fit, the gradient-based partition of the Cess dataset, the numerical notebook and `vendor/IT_PI.py` with its license.

Use Python 3.12 and run from this directory:

```bash
python -m pip install -r requirements.txt
python -B reproduce.py
```

Use `--no-plots` to recompute the diagnostic tables and membership arrays, or `--output-dir PATH` to choose a separate output directory. The command uses the included global fit, performs the regional calculations and writes figures and tables under `generated/`. It compares the two CSV tables and 100 membership arrays with the included results, then writes `generated/verification.json`.

The notebook `reproduce_cess.ipynb` uses the included global fit when `REFIT_GLOBAL=False`. Its cells write to `results/`; use the CLI for a separate run.

## Calculation settings

The global fit contains two input dimensionless groups from IT–π. The notebook partitions `StandardScaler(log1p(input_PI))` coordinates using KMeans and calculates regional MI-based relevance scores.

The paper's comparison figure uses logarithmic coordinates, M = 3 and M = 5, and seed 42. The package also includes diagnostic tables for raw and standardized-log coordinates, M = 2, 3, 5, 8 and 10, and seeds 42–46. These settings give 50 diagnostic calculations. The regional estimator uses k = 5 and a relevance threshold of 0.9.

## Files

| Path | Contents |
|---|---|
| `data/dataset_cess_channel.csv` | 1,000 Cess-model samples |
| `data/recorded_gradient_membership.csv` | Gradient-based partition and physical plotting coordinates: Reτ, η = y/h and y⁺ |
| `results/global_itpi_result.npz` | Global IT–π groups and coefficients |
| `results/regional_cell_scores.csv` | Regional MI, relevance scores, categories, counts and overlap with the intermediate region |
| `results/regional_sensitivity.csv` | The 50 diagnostic calculations |
| `results/regional_memberships.npz` | Pointwise region and category assignments |
| `results/figure_settings.json` | Figure and calculation settings |
| `results/Cess_ITPI_comparison.png`, `.pdf` | Comparison figure |
| `results/regional_count_sweep/` | Region-count diagnostic figure |

Membership-array keys use `conditioning__M<count>__seed<seed>__region` or `__category`. Category codes are X1 = 0, X2 = 1 and Both = 2.

The included gradient-based partition has 779 near-wall, 114 intermediate-transition and 107 outer samples. Its intermediate samples span η ≈ 0.13005–0.35561.

CSV verification uses relative tolerance 1e-9 and absolute tolerance 1e-10; membership arrays are compared exactly. The default command reuses the global fit and repeats the regional calculations.
