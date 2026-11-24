# Evidence, reproduction, and scientific interpretation

## Execution identities

Historical evidence, historical replay, fresh numerical recomputation, and a fresh complete agent execution answer different questions. Run manifests identify the execution mode and hashes. A new model response is not expected to reproduce the historical response byte-for-byte; fixed model names do not freeze a hosted service. The original archival evidence remains immutable.

The repository distributes the authors' supplied CSVs and reference evidence, not the unpublished manuscript. Figure and table descriptions below are navigation aids for the corresponding study; they do not assert that a new execution reproduces every historical number.

## Evidence by stage

| Stage | Inspect |
|---|---|
| Problem formulation | Exact prompt, input column order, dependent variable, semantic records and confidence |
| Physical consistency | Quantity-specific query, top-k results, lexical/semantic/hybrid scores, record citations, candidate and revised dimensions, gate and revision history |
| Dimensional basis | Ordered variables, dimensional matrix, rank/nullity, symbolic basis, orthonormal screening basis |
| Analysis protocol | Planner output, manual overrides, actual resolved settings and parameter provenance |
| Mechanism evidence | Eligible row mapping, surrogate settings, gradients, candidate cluster counts, eigenvalue ratios, sample fractions, directional evidence and acceptance reasons |
| Local selection | Similarity-type screening, formal local learning, group-count candidates, MI and epsilon, restart and subsampling uncertainty |
| Selected relations | Full, sparse and rationalized exponents; accepted/rejected simplifications and measured information loss |
| Validation and figures | Exact training/test rows, predictions and evaluation settings, per-regime collapse and final plots |

`state.json` is the full JSON export. Stage-oriented files under `evidence/` are navigation views of the same data; they do not run a second implementation. `events.jsonl` retains actual node updates and unsuccessful attempts. HTML cards are captured from the real IPython display output, not recreated to imply a successful historical execution.

## Optimizer generations

`evidence/optimization_trace.jsonl` is append-only. A generation record is written after the original `ask` / objective-evaluation / `tell` operations. Recording does not evaluate the objective again, consume scientific random draws, or change the selected candidate.

Records identify the local scope, phase, search, seed, group count and whether output scaling is optimized. They contain generation number, evaluations, sigma, best objective, population statistics and coefficient vectors. Scalar/direct paths record an `optimizer_end`; they do not have CMA generations. Missing historical trajectories are displayed as missing, rather than inferred from endpoints.

For feasible candidates the search objective is negative MI; original feasibility and collinearity penalties also contribute. A plot of `-objective` is therefore not universally an MI plot. A two-coordinate projection of a higher-dimensional trajectory is not an independently recomputed MI landscape. Fresh landscape grids are available through the separate reproduction package.

The original selection rule is retained: the primary-seed candidate is the reported result and other restarts quantify variability. The code does not replace this with the best result from all restarts.

## Figure coverage

| Study content | Corresponding evidence |
|---|---|
| MHD discovery | Dimensional matrix/basis, MI landscape, group count and simplification, collapse |
| Four Type I examples | Pressure-gradient, Rayleigh–Bénard, keyhole and Blasius landscape/selection/collapse |
| Sand-grain roughness | Raw surface statistics, two-group surface, training convex hull and seven-predictor/two-group comparison |
| Type II examples | Velocity and Goldenfeld input groups, output rescaling and before/after coordinates |
| Multiple regimes | Cess and rough-pipe gradients, partitions, selection criteria and local collapse |
| Dimensional-checking benchmark | 600 trials, 730 rounds, initial/revision/gate evidence and API statistics |
| Velocity comparison | Six profiles, transformations, matched-coordinate RMS and interval errors |
| Regime comparison | IT–Pi regional relevance with several region counts versus gradient partition and sensitivity |

## Limits that matter when reading results

- A held-out score calculated after selecting a representation evaluates conditional data collapse. It is not a strict out-of-sample estimate for the complete discovery procedure.
- Similarity-type screening coefficients are diagnostic. The formal selected representation comes from the subsequent learning stage.
- Invertible power transformations, group ordering, and cluster-label permutations may describe equivalent representations. Type II output-rescaling exponents must not be normalized like an input-direction display.
- `Ra^(1/3)` as an input coordinate does not establish `Nu proportional to Ra^(1/3)`.
- Sand-grain roughness includes the declared `Sk + 2` transformation and a source-stratified 83/21 split. Comparisons of raw and learned predictors must use the same split; extrapolation outside the training hull is marked.
- The rough-pipe CSV is generated from a prescribed piecewise relationship with a switch at Re = 3000. This is not an experimentally discovered transition threshold.
- Blasius has 200 original observations and 196 positive-response observations; velocity has 1,109 original and 1,103 positive-response observations. Surrogate training selection is separate from collapse-validation splitting.
- The dimensional-checking benchmark compares three parsing models, ten cases, two wording variants and ten repetitions, with the same Checker model and at most one revision. It is not a complete paired RAG-on/off experiment.
- Full semantic ranking can be recomputed for 60 archived Sandgrain rounds with recorded query vectors. Other archived rounds replay recorded top-k scores; missing query vectors are not silently reconstructed.
- Default Cess and velocity comparison reproduction uses the recorded fitted model or partition where documented. Replotting these comparisons does not claim to have retrained the complete agent pipeline.
