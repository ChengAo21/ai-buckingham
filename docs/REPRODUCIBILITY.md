# Reproducibility and manuscript precedence

The supplied manuscript and SI are authoritative. Historical numerical overrides in `configs/source_case_overrides.json` are retained for audit and are never selected by normal run entry points; that file also retains the nine physical problem prompts used at runtime.

| Item | Supplied script | Prepared release | Authority |
|---|---|---|---|
| Minimum cluster samples | 80 in schema/default/fallbacks | 60 throughout | SI Tables 4, 6; main methods |
| Rationalization relative MI loss | fallback 0.15 | 0.10 | SI Table 6 |
| Rational denominator fallback | 4 in the rationalizer | 5; Rayleigh 4, Blasius 8 | SI Tables 5–6 |
| Cess / rough-pipe route | prescribed multi-regime | automatic assessment | SI Table 5 |
| Cess similarity policy | implicit prescribed Type I | automatic screening | SI Table 5 |
| Pipe candidate K | case override [2,3] | [2,3,4] | SI Table 6; Table 25 includes K=4 |
| Cess / pipe GPR optimization | case override 3 restarts / 2,000 iterations | shared 5 restarts / 500 iterations | SI Table 6; Table 5 specifies no override for these fields |
| Pipe large local regime histogram | inherited 40 bins | 20 bins for local histogram mode | SI Table 5 |
| Analyzer model | gpt-4.1-mini | gpt-5-mini | SI Section 2: all interactions use gpt-5-mini |
| Temperature | zero requested in legacy constructor | zero explicitly preserved in request | SI Section 2 |
| Goldenfeld path | nested PRE_cases path, absent | existing data/dataset_goldenfeld.csv | Supplied files |
| Blasius output alias | prompt says u* | exact CSV u/U | SI Eq. 57 and supplied CSV |
| API setup | clients constructed during import | deferred until explicit API launch | Engineering correction |
| Paratera default host | api.paratera.com | llmapi.paratera.com | User-specified endpoint |

## What has and has not been reproduced

The reference JSON is transcribed from the SI, including uncertainties, missing validation entries and the failed rough-pipe result. All nine displayed collapse plots and landscape slices are recalculated on the supplied CSVs. These checks do not claim that all nine full 30,000-generation searches have been rerun.

Full-budget regime-only recomputation is available through `scripts/recompute_regimes.py`; reports in `results/computed` retain their own timestamps, numerical statuses, GPR diagnostics and settings. Reduced-budget previews in `results/previews` use 220 GPR training samples, zero optimizer restarts, at most 100 optimizer iterations and 8 clustering initializations. They may yield a different K and are labeled accordingly.

The CLI `--quick` numerical run uses population 24, at most 30 generations, one optimization run, no held-out GPR or subsampling UQ, and a reduced regime-surrogate budget. It is a smoke test. The separate interactive CMA slice records its chosen population/generations and fixes coefficients outside the displayed plane.

## Data handling and interpretation

The original executor removes a row if any selected input or output is nonfinite or has absolute value below 10⁻¹⁵. Power products and log-Pi gradients additionally require positive inputs. The supplied primary cases satisfy this after filtering. CSV indices are preserved; figures do not silently replace discarded values.

The report records seed 42 and effective settings. MI estimates depend on estimator, sample count and coordinate conventions at finite N; theoretical invariance does not make finite-sample estimates numerically identical. Multiple dominant groups are nonunique up to invertible mixing, so restart stability is interpreted at the subspace level when appropriate.

Reported validation R² evaluates an already-selected representation on three later 80/20 splits. It is conditional predictive-collapse validation, not a nested, held-out evaluation of variable selection or the whole agent. In particular, the rough-pipe Reynolds regime after rationalization has R² = −3.079 (SI Table 25), which is retained as a failed validation.

## API compatibility

No real API credentials are tracked. The provided services are configured through environment variables; LangSmith tracing is disabled until enabled explicitly. GPT-5 parameter handling varies by adapter/provider. The request preserves the SI temperature-zero setting with `extra_body` because the installed LangChain adapter otherwise removes it. Provider rejection is reported as a failure, never corrected by an unreported parameter change. The paid API route was not exercised for this local preparation.

The installed Windows/Python 3.13 dependency snapshot is in `requirements-lock.txt`. `pyproject.toml` lists portable dependency ranges. Dataset hashes, figure provenance, original symbol line mapping and machine-readable configurations are included so users can audit the implementation.
