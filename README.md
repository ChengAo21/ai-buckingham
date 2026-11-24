# AI Buckingham

**Agentic discovery of dimensionless scaling laws across physical regimes.**

AI Buckingham combines LLM-guided problem formulation and planning with fixed symbolic and numerical tools: retrieval-grounded dimensional checking, Buckingham–Pi bases, gradient-based regime identification, mutual-information optimization, simplification, and collapse validation.

The repository contains the ten supplied case datasets, small executable LangGraph notebooks, recorded run bundles, an interactive Streamlit evidence browser, and four independent reproduction packages. It does not contain unpublished manuscripts or copies of reference publications.

## Start here

Use **Python 3.12** in a fresh environment, then run these commands from the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[app,notebooks,dev]"
streamlit run streamlit_app.py
```

Browsing recorded evidence does not require API credentials. Numerical and historical reproduction have separate commands below. Complete new agent runs require the configured LLM and embedding services.

| Task | Entry point |
|---|---|
| Explore results and diagnostics | `streamlit run streamlit_app.py` |
| Read an executed case | [`notebooks/cases/`](notebooks/cases/) |
| Browse evidence without API access | [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) |
| Inspect machine-readable run evidence | [`results/cases/`](results/cases/) |
| List cases without model access | `ai-buckingham --list` |
| Check all supplied input/evidence hashes | `python reproduction/verify_integrity.py` |
| Understand package boundaries and changes | [`docs/refactoring.md`](docs/refactoring.md) |
| Understand evidence and scientific limits | [`docs/evidence.md`](docs/evidence.md) |
| Inspect case exceptions and native thread profiles | [`docs/configuration.md`](docs/configuration.md) |
| Check saved results against reported numerical tables | `python scripts/check_publication.py` |
| Use or deploy the web app | [`docs/app.md`](docs/app.md) |

## Ten cases

| Case | Class in the supplied study | Notebook |
|---|---|---|
| Magnetohydrodynamic duct flow | Type I | [mhd](notebooks/cases/mhd.ipynb) |
| Pressure-gradient wall friction | Type I | [pressure_gradient](notebooks/cases/pressure_gradient.ipynb) |
| Rayleigh–Bénard convection | Type I | [rayleigh_benard](notebooks/cases/rayleigh_benard.ipynb) |
| Laser keyhole depth | Type I | [laser_keyhole](notebooks/cases/laser_keyhole.ipynb) |
| Blasius boundary layer | Type I | [blasius](notebooks/cases/blasius.ipynb) |
| Equivalent sand-grain roughness | Type I, two informative groups | [sandgrain](notebooks/cases/sandgrain.ipynb) |
| Compressible velocity scaling | Type II | [velocity_scaling](notebooks/cases/velocity_scaling.ipynb) |
| Goldenfeld rough-pipe scaling | Type II | [goldenfeld](notebooks/cases/goldenfeld.ipynb) |
| Cess channel eddy viscosity | Multiple regimes | [cess_channel](notebooks/cases/cess_channel.ipynb) |
| Rough circular pipe | Multiple regimes | [rough_pipe](notebooks/cases/rough_pipe.ipynb) |

The ten CSVs contain 4,926 observations. The exact bytes and column orders are preserved from the supplied current data package. Some stages exclude zero responses; original row counts and eligible counts are recorded separately. Source descriptions, generation methods, preprocessing and licenses are documented in [`data/README.md`](data/README.md). Each case has its own [configuration](configs/cases/), rather than replacing its settings with uniform defaults.

## Run a complete agent analysis

Set credentials **in your process environment**, using the variable names in [`.env.example`](.env.example). The application does not automatically execute or load credential files. The LLM defaults are GPT-5 mini and the embedding model is GLM-Embedding-3. An OpenAI-compatible LLM endpoint can be selected with `OPENAI_BASE_URL`; the embedding endpoint uses `PARATERA_BASE_URL`.

```python
from ai_buckingham import initialize_models, run_case

initialize_models()
state = run_case("mhd")
```

Or initialize and execute explicitly through the CLI:

```bash
ai-buckingham mhd
```

New runs go to `generated/cases/<case_id>/<run_id>/` by default. A run directory is never overwritten. Package import, case listing, graph construction, and evidence browsing do not make API calls. Tracing to external telemetry is disabled unless explicitly enabled in the environment.

For custom data, pass a configuration with `case_id`, `dataset`, `prompt` and `manual_hyperparams` to `ai_buckingham.cli.run_analysis`, or use the web app's **Run new analysis** page. Expected dimensions in the supplied configurations are reference metadata; they do not replace the Analyzer/Checker workflow.

## Results are a shared evidence bundle

Each run stores:

```text
manifest.json                  # execution identity, status, hashes and environment
resolved_config.json           # case settings, actual plan and resolved parameters
state.json                     # complete serializable scientific state
events.jsonl                   # actual node updates, including revision attempts
evidence/                      # retrieval, dimensions and numerical diagnostics
evidence/optimization_trace.jsonl
tables/                        # numerical exports
figures/                       # original plots, with vector exports where available
report.html                    # actual recorded stage cards, readable without Jupyter
report.md
```

The CMA-ES viewer shows **actual generations**: best and population objective, step size, evaluations, coefficient vectors and search identity. The objective includes the original penalties and must not automatically be interpreted as mutual information. Scalar or direct searches have endpoints, not invented generations. See [`docs/evidence.md`](docs/evidence.md).

Included new executions are identified as new runs. They are not relabeled as the original manuscript executions. Every available historical comparison remains separately identified and hash-locked.

This prepared snapshot includes **10 verified completed live case runs and 9,853 actual CMA-ES generations**. It contains one reviewed execution per case. All 309 numerical-publication acceptance checks pass, covering reported stage MI, epsilon, combined uncertainty, R², group counts and candidate partition counts. The final input assignments match the reference SI dimensions, and the dimensional bases pass null-space checks. See [`docs/validation/publication_alignment.json`](docs/validation/publication_alignment.json), [`results/validation.json`](results/validation.json) and the per-mechanism [`results/summary.csv`](results/summary.csv).

Five MI landscapes are included in [`results/mi_landscapes/`](results/mi_landscapes/). Four are fresh searches with the supplied notebook budgets; the MHD grid overlays its actual recorded pre-sparsification trajectory. The [Sandgrain prediction comparison](results/sandgrain_prediction/) uses the same recorded 83/21 split for seven original predictors and the learned representation, with its offline refit identified separately.

The [pressure-gradient Reynolds-number baseline](results/pressure_gradient_baseline/) additionally preserves all 144 source observations and their original ZPG/APG/FPG classes.

Figure source tables contain the actual per-sample collapse coordinates and recorded gradient fields. Reusable offline exporters are provided as `scripts/export_plot_data.py`, `scripts/embed_saved_figures.py` and `scripts/index_results.py`; they read saved evidence without new discovery or API calls. Candidate memberships reconstructed from saved gradients are explicitly marked as offline recalculation in `results/regime_candidates/`.

## Offline reproduction

These commands preserve historical snapshots and write newly generated files under `generated/`:

```bash
python reproduction/verify_integrity.py
python reproduction/rag_ablation/reproduce.py
python reproduction/regime_comparison/reproduce.py
python reproduction/velocity_comparison/reproduce.py
```

The MI-landscape package performs fresh optimizer searches. See its README for case names and budgets:

```bash
python reproduction/mi_landscapes/reproduce.py --help
```

Use [`environments/`](environments/) when exact historical dependency versions are needed. The dimensional-checking archive was recorded with different NumPy/Pandas versions from the numerical comparisons; upgrading or merging those environments is not part of the refactor.

Reproduction coverage and the preparation-time checks are recorded in [`reproduction/publication_validation.json`](reproduction/publication_validation.json). In particular, the archive contains 600 scheduled dimensional-checking trials and 730 Checker rounds. It compares initial and revised outcomes; it is not a paired full-pipeline RAG-on/RAG-off causal experiment.

## Knowledge indexes

Two curated Markdown knowledge bases retain entry-level source citations. Their GLM-Embedding-3 NPZ indexes are provided under [`knowledge/indexes/`](knowledge/indexes/), with model, record count, source hash and embedding dimension metadata. Cached record vectors eliminate repeated document embedding; new semantic queries still require embedding API access.

To explicitly regenerate both distributable indexes:

```bash
python scripts/build_indexes.py
```

Runtime retrieval treats supplied inputs and indexes as read-only. Missing or stale caches rebuild under `generated/cache/`; the explicit index-authoring command validates and promotes them into `knowledge/indexes/`.

## Checks

```bash
python -m pytest -q
python reproduction/verify_integrity.py
python scripts/validate_results.py
python scripts/check_app.py
```

CI performs offline checks only. Complete new runs and fresh full-budget MI searches are explicit, potentially long-running operations and are not triggered by opening a page or running the tests.

The [preparation checks](docs/validation/) include the scientific result validation, package-install check, unit checks and API-blocked web interactions. These are local preparation results; the GitHub workflow itself has not been run on a remote repository.

## Citation and terms

Use [`CITATION.cff`](CITATION.cff) to cite the software and cite the original sources of datasets you use. Original project code is under the MIT license. Third-party code and datasets retain their respective terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No manuscript PDFs, publisher templates, reference PDFs, or local credential files are distributed here.
