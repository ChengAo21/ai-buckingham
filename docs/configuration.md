# Numerical configuration

The configuration used by a result is the combination of library defaults,
the validated planner output, and explicit user settings, in that order of
precedence. Inspect `configs/cases/<case_id>.json`, the run's
`resolved_config.json`, and the `resolved_hyperparameters`,
`hyperparameter_provenance` and `compiled_workflow` fields of `state.json`.

## Native numerical execution

Each included case declares `numerical_threads` in its JSON configuration.
The rough-pipe reference uses **20 threads**; the other nine included runs use
one thread. `run_analysis` applies the declared count to the loaded NumPy,
SciPy and OpenMP libraries for that analysis, records their counts and
versions in `manifest.json` under `numerical_runtime`, and restores the caller's
settings afterwards. A custom analysis without this field inherits its runtime.

The rough-pipe affine-sigmoid GPR fit is sensitive to floating-point reduction
order near the prescribed branch discontinuity. The 20-thread execution is used
for the included gradients and candidate partitions. Use the declared thread
count when reproducing this case.

Different BLAS implementations can also affect floating-point intermediates.
The saved results include native-library metadata;
`python scripts/check_publication.py` checks the included numerical evidence
against the paper's tables. It checks reported
MI, epsilon, combined uncertainty, R², group counts, similarity types and
candidate partition counts.

The settings below describe the case configurations and numerical notebooks
included in this repository. Case overrides are applied to the shared defaults.

## Sand-grain roughness parameterization

[`configs/cases/sandgrain.json`](../configs/cases/sandgrain.json) configures
the following optimization and validation settings:

| Setting | Sand-grain roughness parameterization | Shared default |
| --- | --- | --- |
| CMA-ES population | 60 | 300 |
| Maximum CMA-ES generations | 240 | 30,000 |
| Collapse-validation splits | One fixed 83/21 train/test split | Three 80/20 splits |
| Validation GPR optimizer restarts | 1 | 3 |
| Validation feature coordinates | Original positive group values | Logarithms |
| Validation response coordinates | Original response values | Logarithms |
| Train/test stratification | Source category in the `case` column | Unstratified unless specified |

The 104 surfaces use the declared `Sk + 2` preprocessing for non-integer powers.
The seven-predictor and two-group prediction comparisons use the same 83
training rows and 21 test rows.
The saved validation record includes sample IDs and exact row indices. Its
`evaluation_space` and `resolved_options` identify the coordinates used to
calculate the reported scores.

## Rough-pipe random seeds

The rough-pipe case has distinct random controls for distinct calculations:

| Calculation | Setting | Value |
| --- | --- | --- |
| Regime GPR surrogate and gradient clustering | Resolved gradient-regime `random_state` | 42 |
| Primary dimensionless optimization | Primary random seed | 42 |
| Collapse-validation splitting and validation GPR | `gpr_random_state` | 123 |

The `gpr_random_state=123` override in
[`configs/cases/rough_pipe.json`](../configs/cases/rough_pipe.json) controls the
separate collapse-validation procedure. The surrogate and gradient clustering
use seed 42. Each optimizer restart and validation split has its own seed in
the run records.

The rough-pipe local MI calculations also use different estimators according to
the local sample counts: the larger 975-sample regime uses histogram MI with 20
bins, and the 225-sample regime uses KSG with five neighbors in the included
execution. Regime labels are identifiers, not a fixed physical ordering.

## Skin friction under pressure gradients: Reynolds-number comparison

The pressure-gradient baseline in
[`results/pressure_gradient_baseline/`](../results/pressure_gradient_baseline/)
calculates `Re_delta = U_inf * delta / nu` directly from the supplied 144 rows.
It compares `U_tau/U_inf` against this coordinate using the existing `zeroPG`,
`advPG`, and `favPG` labels, displayed as ZPG, APG and FPG. Its counts are 84, 36
and 24.

From the repository root, regenerate it into a new directory:

```bash
python scripts/export_pressure_baseline.py --output-dir generated/pressure_gradient_baseline
```

The command exports a CSV and figures in PNG, PDF and SVG formats.
