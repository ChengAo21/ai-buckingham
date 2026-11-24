# Numerical configuration and case exceptions

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
SciPy and OpenMP libraries for that analysis, records their actual counts and
versions in `manifest.json` under `numerical_runtime`, and restores the caller's
settings afterwards. A custom analysis without this field inherits its runtime.

The rough-pipe affine-sigmoid GPR fit is sensitive to floating-point reduction
order near the prescribed branch discontinuity. The 20-thread execution
reproduces the reported gradients, candidate partitions and local numerical
evidence. No sample labels are manually assigned or changed. Thread count is
an execution parameter; the scientific objectives, clustering rules, random
seeds and data remain those of the supplied implementation.

Different BLAS implementations can also affect floating-point intermediates.
The saved results and native-library metadata preserve the verified execution;
`python scripts/check_publication.py` checks the included numerical evidence
against the author tables without fitting another surrogate. It checks reported
MI, epsilon, combined uncertainty, R², group counts, similarity types and
candidate partition counts. The expected values are used only for validation,
never as optimization inputs.

The study's shared settings describe defaults, not every case override. The
exceptions below originate in the supplied implementation and numerical
notebooks. They were retained when the source was split into modules.

## Sand-grain roughness

[`configs/cases/sandgrain.json`](../configs/cases/sandgrain.json) retains the
following explicit settings from the supplied source:

| Setting | Sandgrain case | Shared default |
| --- | --- | --- |
| CMA-ES population | 60 | 300 |
| Maximum CMA-ES generations | 240 | 30,000 |
| Collapse-validation splits | One fixed 83/21 train/test split | Three 80/20 splits |
| Validation GPR optimizer restarts | 1 | 3 |
| Validation feature coordinates | Original positive group values | Logarithms |
| Validation response coordinates | Original response values | Logarithms |
| Train/test stratification | Source category in the `case` column | Unstratified unless specified |

The 104 surfaces use the declared `Sk + 2` preprocessing for non-integer powers.
The original seven-predictor and learned-group prediction comparisons use the
same 83 training rows and 21 test rows; they do not independently select splits.
The saved validation record includes sample IDs and exact row indices. Its
`evaluation_space` and `resolved_options` identify the coordinates used to
calculate the reported scores.

These are case exceptions to the shared settings. Replacing them with the
generic defaults changes the supplied experiment. The supplementary case table
already reports the fixed 83/21 comparison; a concise table of case settings
should include the optimization and validation overrides above as well.

## Rough-pipe random seeds

The rough-pipe case has distinct random controls for distinct calculations:

| Calculation | Setting | Value |
| --- | --- | --- |
| Regime GPR surrogate and gradient clustering | Resolved gradient-regime `random_state` | 42 |
| Primary dimensionless optimization | Primary random seed | 42 |
| Collapse-validation splitting and validation GPR | `gpr_random_state` | 123 |

The `gpr_random_state=123` override in
[`configs/cases/rough_pipe.json`](../configs/cases/rough_pipe.json) controls the
separate collapse-validation procedure. It does not replace the surrogate's seed
with 123. The surrogate, clustering and validation records each retain their own
resolved seed. Additional optimizer restarts and repeated validation splits have
their recorded individual seeds.

The rough-pipe local MI calculations also use different estimators according to
the local sample counts: the larger 975-sample regime uses histogram MI with 20
bins, and the 225-sample regime uses KSG with five neighbors in the included
execution. Regime labels are identifiers, not a fixed physical ordering.

## Independent MI-landscape notebooks

The four supplied landscape examples are independent numerical-notebook
calculations. Their settings must not be substituted for the full agent's case
settings or described as the same optimization run.

| Independent landscape | Population | Maximum generations | Accepted rationalization MI loss |
| --- | --- | --- | --- |
| Rayleigh–Bénard | 300 | 50,000 | 0.06 |
| Pressure-gradient friction | 300 | 50,000 | 0.06 |
| Laser keyhole | 300 | 50,000 | 0.10 |
| Blasius | 200 | 50,000 | 0.06 |

These searches use the notebook's seed 42. The maximum is a budget; the original
CMA-ES convergence criteria can stop the actual search sooner. The main agent's
shared population and maximum-generation defaults are 300 and 30,000, with its
shared rationalization MI-loss threshold of 0.10. The independent notebook also
retains its own other convergence and rationalization settings. Full settings,
actual stopping reasons and numeric generation histories accompany each result
under [`results/mi_landscapes/`](../results/mi_landscapes/).

The MHD landscape has a different source mode:
`offline_landscape_evaluation_with_recorded_trajectory`. Its grid is evaluated
over the full two-coefficient input basis and overlaid with the actual recorded
primary-seed trajectory. No additional MHD CMA-ES search is claimed by that
export. A contour's fixed slice and a search trajectory projected onto two
coefficients are different objects.

## Direct Reynolds-number baseline

The pressure-gradient baseline in
[`results/pressure_gradient_baseline/`](../results/pressure_gradient_baseline/)
calculates `Re_delta = U_inf * delta / nu` directly from the supplied 144 rows.
It compares `U_tau/U_inf` against this coordinate using the existing `zeroPG`,
`advPG`, and `favPG` labels, displayed as ZPG, APG and FPG. Its counts are 84, 36
and 24. This is a reference coordinate plot, without any model fitting or
dimensional discovery.

From the repository root, regenerate it into a new directory:

```bash
python scripts/export_pressure_baseline.py --output-dir generated/pressure_gradient_baseline
```

The CSV, full-resolution PNG, vector PDF/SVG and manifest preserve the row mapping,
coordinate definition, label mapping, source hash and generation script hash.
