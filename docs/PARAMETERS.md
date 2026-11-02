# Parameter reference

**Authority: the supplied main manuscript and SI, especially SI Sections 1–4 and Tables 5–6.** The user explicitly specified that manuscript settings take precedence over inconsistent source defaults. Normal entry points use `settings.paper_settings(case_id)`; nine complete resolved configurations are committed in `configs/paper/`.

## Configuration precedence

Manuscript shared settings → manuscript case overrides → compiled, validated policy. A deliberately changed interactive diagnostic budget is recorded as a separate mode. Environment variables select credentials and service endpoints; no credentials are part of scientific configuration exports. The historical source dictionaries are audit material only.

## Semantic and retrieval layer

| Setting | Value / behavior |
|---|---|
| All LLM interactions | `gpt-5-mini`; requested temperature 0; SI Section 2 |
| Analyzer / planner environment names | `ANALYZER_MODEL`, `PLANNER_MODEL`; defaults both `gpt-5-mini` |
| Chat endpoint | `OPENAI_BASE_URL`, else `OPENAI_API_BASE`, default `https://api.chatanywhere.tech/v1` |
| Embedding endpoint | `PARATERA_BASE_URL`, default `https://llmapi.paratera.com/v1` |
| Embedding model / batch | `GLM-Embedding-3`; batches of 16 documents; l2-normalized vectors cached locally |
| Physical-quantity knowledge base | 247 entries; lexical weight 0.65, semantic weight 0.35; top 3 per variable |
| Dimensionless-relation knowledge base | 251 entries; lexical weight 0.20, semantic weight 0.80; top 5 for interpretation |
| Grounding threshold / row cap | hybrid score 0.22; at most 30 displayed evidence rows |
| Lexical fallback | disabled by default; no silent replacement of semantic grounding |
| Physical dimension convention | canonical `(M, L, T, I, Theta, N, J)`; case tables omit unused trailing components |
| API timeout | 500 seconds in the supplied adapter |
| Bounded macro replanning | at most 2 replans for recognized workflow failures |
| Tracing | `LANGCHAIN_TRACING_V2=false` by default; optional LangSmith endpoint and key |

Temperature zero is transmitted explicitly with `extra_body` for GPT-5 because the installed adapter otherwise drops it. A rejecting provider causes an explicit failure; no temperature adjustment is silently substituted. Changing a model through the environment is an intentional departure from the paper and is recorded in API-run metadata.

For benchmark API runs, the inferred dimensional subspace is checked against the SI subspace before being re-expressed in the SI basis convention. This preserves semantic analysis and tool auditing while ensuring that index-based fixed-power constraints refer to the correct physical groups.

## Dimensionless learning and optimization

| Setting / key | Manuscript default | Interpretation |
|---|---|---|
| `random_state` | 42 | Primary reproducibility seed |
| `cma_restarts`, `regime_cma_restarts` | 3 | Optimization runs per candidate |
| `cma_init_position` | 0.1 | Common starting coefficient |
| `cma_init_stepsize` | 0.5 | Initial CMA sigma |
| `cma_bounds_min/max` | −1 / 1 | CMA coefficient bounds |
| `cma_popsize` | 300 | Population |
| `cma_maxiter` | 30,000 | Maximum generations |
| CMA `tolx`, `tolfun` | 10⁻⁸ / 10⁻⁸ | Position / objective convergence |
| Zero free coefficients | direct evaluation | No unnecessary optimization |
| One free coefficient | bounded scalar search | Default [−2,2], xatol 10⁻⁸; avoids one-dimensional CMA instability |
| `normalize_input` | `none` | `norm2` in the Goldenfeld case; does not normalize Type II output magnitude |
| `log_transform_mi` | false | Actual finite-sample objective uses power products by default |
| Multi-input collinearity | enabled when more than one group | Maximum absolute cosine 0.70 by default |
| Penalty | 10⁴ + violation ×10⁵ | Rejects collinear candidates before MI evaluation |

| MI setting | Default |
|---|---|
| Automatic selection | KSG for local N <800; histogram for local N ≥800 |
| KSG neighbor count | 5, with case-specific overrides below |
| Histogram bins | 40, with velocity=50 and pipe local histogram=20 |
| Logarithm base | natural; MI measured in nats |
| Normalized information score | ε=exp[−max(MI,0)] |
| `abs_score_drop_threshold` | 0.10; continue group-count search only for sufficient absolute score decrease |
| Minimum local samples | 20 |
| Nominal samples per MI dimension | 20 |
| Minimum samples for k=2 comparison | 40 |
| Count policy | begin k=1; test at least two counts when allowed; never hard-code the final k from clustering |

## Type I / II screening and simplification

| Setting | Value |
|---|---|
| Screening candidate input count | k=1 for both Type I and Type II |
| Maximum relative MI loss without output rescaling | 0.15 |
| Negligible output physical-exponent threshold | 0.05 |
| Screening collinearity limit | 0.70 |
| First-kind fallback | used only when a prescribed/default policy requests it; paper cases use auto except pressure |
| Sparsification threshold | 0.05 on physical-variable exponents, across retained formulas |
| Sparsification acceptance | increase in ε no greater than 0.05; restrict null space and reoptimize |
| Rationalization maximum denominator | 5; Rayleigh=4, Blasius=8 |
| Maximum coefficient approximation error | 0.05 |
| Decimal fallback precision | 2 decimals |
| Rationalization acceptance | relative MI loss no greater than 0.10 |
| Fixed-power searches | sparsification disabled when the original workflow's basis-power constraints require it |

Screening removes final fixed-power constraints for the preliminary comparison and restores them for final inference. Velocity fixes the wall-unit input power to 1 and output power to 0; Goldenfeld fixes the output relative-roughness power to 0. These constraints refer to the SI basis order.

## Gradient regimes

| Setting | Value |
|---|---|
| Coordinates | log of Schmidt-orthonormalized Buckingham-Pi basis; no standardization/PCA substitution |
| Response transform | log of the dimensionless target |
| Covariance | isotropic squared-exponential, σ₀² exp(−σ₁²∥x−x′∥²), with diagonal noise |
| Default mean | constant; pipe uses affine–sigmoid transition |
| `max_train_samples` | all available by default; Cess=800, pipe=1,100 |
| GPR optimizer restarts | 5 |
| GPR optimizer maximum iterations | 500 |
| Bounds: log σ₀², log σ₁², log σu² | [−30,30] each; pipe overrides below |
| Diagonal jitter | 10⁻¹⁰ |
| Initial noise fraction | 10⁻⁶ |
| Function / gradient tolerances | 10⁻¹⁰ / 10⁻⁶ |
| Require GPR optimizer success | false; finite unconverged fits are explicitly reported |
| Candidate K | {2,3,4} |
| Minimum cluster samples | 60 |
| Minimum cluster fraction | 0.05 |
| Minimum λmax / λmin | 50 |
| Random clustering initializations | 30 |
| Maximum clustering iterations | 200 |
| Initialization | random eigenpairs; maximum 100 attempts |
| Require clustering convergence | true |
| Gradient norm cutoff | 10⁻¹²; directionless rows are recorded and excluded |
| Automatic directional support | K=2 leading-direction absolute cosine <0.90 |
| Cross-K selection | coarsest admissible partition, with the automatic support condition |
| Global leading-eigenvalue fraction E | diagnostic only |
| Cluster objective | compare starts within the same K; never a cross-K route score |
| Boundary-confidence threshold | 0.05; excluding low-confidence rows disabled by default |

## Conditional collapse validation and UQ

| Setting | Value |
|---|---|
| Validation GPR kernel | constant-amplitude anisotropic RBF + white noise; separate from the regime GPR |
| Constant-amplitude / length-scale bounds | [10⁻³,10³] / [10⁻³,10³] |
| White-noise bounds | [10⁻⁸,10] |
| Diagonal regularization | 10⁻¹⁰ |
| Optimizer restarts | 3 |
| Maximum evaluation samples | 800 |
| Held-out fraction / repeats | 0.20 / 3 |
| Log features / target | true by default in the supplied numerical evaluator |
| Mean held-out R² threshold | 0.90 |
| Subsampling trials / nominal fraction | 20 / 0.50 |
| Combined numerical SD | sqrt(σ_restart² + σ_subsample²), with both terms retained separately |

The validation split occurs after representation selection. The reported R² is not a nested evaluation of the complete discovery workflow.

## Per-case overrides

| Case | MI | Initial coefficient / sigma | CMA bounds | Qmax | Regime GPR changes | Similarity |
|---|---|---|---|---|---|---|
| MHD | KSG 4 | 0.1 / 0.5 | [−1,1] | 5 | all 483 | auto |
| Pressure | KSG 3 | 0.1 / 0.5 | [−1,1] | 5 | all 144 | prescribed Type I |
| Rayleigh | KSG 6 | 0.3 / 0.5 | [−1,1] | 4 | all 182 | auto |
| Keyhole | KSG 4 | 0.1 / 0.5 | [−1,1] | 5 | all 90 | auto |
| Blasius | KSG 5 | 0.6 / 0.3 | [−1,1] | 8 | 196 after near-zero filter | auto |
| Velocity | histogram 50 | 0.1 / 0.2 | [−1,1] | 5 | 1,103 after filter | auto |
| Goldenfeld | KSG 5 | 0.1 / 0.1 | [−0.1,0.5] | 5 | all 414 | auto |
| Cess | KSG 5 | 0.1 / 0.5 | [−1,1] | 5 | max 800 / 1,000 | auto |
| Pipe | automatic per local N; bins 20 / KSG 5 | 0.1 / 0.5 | [−1,1] | 5 | max 1,100 / 1,200; affine–sigmoid mean | auto |

Velocity uses a final MI collinearity limit 0.80 and compares only jointly free input/output coordinates (`penalty_opt_w_fixed=false`). Goldenfeld uses `normalize_input=norm2` and an output roughness power fixed to zero. Pipe bounds are log σ₀² ∈ [−40,40], log σ₁² ∈ [−30,30], log σu² ∈ [−30,5]. All cases perform automatic regime assessment.

The complete list of `hp.get(...)` controls, including notebook display switches, plot size/DPI, optional physical plot axes and diagnostic overrides, is extracted into `configs/parameter_inventory.csv` with source module, line, and fallback expression. These implementation controls are not all paper hyperparameters; principal scientific defaults are resolved by `configs/paper/*.json`.
