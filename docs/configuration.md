# Numerical configuration

The configuration used by a result is the combination of library defaults,
the validated planner output, and explicit user settings, in that order of
precedence. Inspect `configs/cases/<case_id>.json`, the run's
`resolved_config.json`, and the `resolved_hyperparameters`,
`hyperparameter_provenance` and `compiled_workflow` fields of `state.json`.
Configuration shown in a planner's prose can describe its proposal; the resolved
structured settings determine the executed calculation.

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
against the author tables without fitting another surrogate. It checks reported
MI, epsilon, combined uncertainty, R², group counts, similarity types and
candidate partition counts. The expected values are used only for validation,
never as optimization inputs.

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

## MI optimization landscape settings

The five landscape examples use the notebook's CMA-ES and MI calculations.
The full case workflows use the separate settings in `configs/cases/`.

| Case | Population | Maximum generations | Accepted rationalization MI loss |
| --- | --- | --- | --- |
| Magnetohydrodynamic power generator | 300 | 50,000 | 0.06 |
| Rayleigh–Bénard convection | 300 | 50,000 | 0.06 |
| Skin friction under pressure gradients | 300 | 50,000 | 0.06 |
| Laser-metal interaction in additive manufacturing | 300 | 50,000 | 0.10 |
| Blasius laminar boundary layer | 200 | 50,000 | 0.06 |

These searches use seed 42. CMA-ES convergence criteria can stop a search before
the maximum-generation budget. The main agent's
shared population and maximum-generation defaults are 300 and 30,000, with its
shared rationalization MI-loss threshold of 0.10. Full settings,
stopping reasons and numeric generation histories accompany each result
under [`results/mi_landscapes/`](../results/mi_landscapes/).

The trajectory follows the CMA-ES search center at each generation. The circle
and star mark the first and last centers on the displayed coefficient axes.

For MHD, the grid varies the exponents of the two basis groups in Figure 3b.
The search evaluates MI with the dimensionless velocity `u*` using the KSG
estimator with four neighbors. The MHD plot uses the CMA-ES center in the
bounded coefficient space. The accompanying data-collapse plot uses the
rationalized group in Figure 3e.

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

The CSV, full-resolution PNG, vector PDF/SVG and manifest preserve the row mapping,
coordinate definition, label mapping, source hash and generation script hash.
