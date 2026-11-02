# Architecture and source migration

The supplied notebook-style script has been separated by responsibility. The original authoring files remain outside the public repository. `data/manifest.json` records the original script SHA-256 and unchanged dataset hashes.

| Module | Responsibility |
|---|---|
| `engine/shared.py` | Shared imports, constants, deferred client setup and manuscript-aligned defaults |
| `engine/knowledge.py` | Markdown schemas, lexical/semantic retrieval, embedding cache |
| `engine/schemas.py` | Pydantic dimensional and workflow contracts; LangGraph state |
| `engine/basis.py` | Exact symbolic null-space tool and residual check |
| `engine/information.py` | KSG/histogram MI, power products, formulas, GPR collapse and uncertainty |
| `engine/regimes.py` | GPR maximum-likelihood fit, analytic gradients, eigenpair clustering, policy evidence |
| `engine/discovery.py` | CMA-ES, screening, adaptive counts, sparsification, rationalization and local workflows |
| `engine/agent.py` | Structured LLM interactions, workflow compilation, execution and bounded replanning |
| `engine/figures.py` | Original notebook plotting functions |
| `engine/notebook.py` | Optional notebook HTML and Markdown reporting |

Cross-module imports needed only at function execution are local imports. This preserves the interdependent research functions without import cycles. Importing these modules constructs no LLM clients and requires no API credentials. `build_adaptive_agent_graph()` initializes clients only when the API workflow is explicitly requested.

`catalog.py` defines the exact SI variable order, dimensions and displayed bases. `settings.py` resolves paper defaults and case-specific overrides. `experiments.py` is the boundary for runs and exported provenance. `showcase.py` supplies concise workflow and case descriptions for both interfaces. `app.py` is presentation and interaction; scientific inference remains in the engine. The static page is an export of supplied-data coordinates and explicitly labeled objective slices.

The numerical workflow uses verified SI inputs and bases, bypassing semantic inference. The full agent retains the Analyzer → physical consistency checker → basis tool → planner → numerical executor → bounded replanner → grounded report → visualizer graph. These two execution modes are distinguished in every exported report.

Instrumentation adds generation records to actual CMA-ES results and retains candidate-K label arrays for interactive inspection. It does not change objective evaluations, cluster acceptance or selection. The full-dimensional local traces are stored with their candidate results. The interactive optimizer is a separate, explicitly labeled two-coefficient experiment.

The two-coefficient previews also record the CMA distribution mean and its MI.
Those additional mean-MI evaluations are counted separately from candidate
evaluations. `build_research_assets.py` exports shared case data for GitHub Pages.
`build_discovery_map.py` composes the workflow from packaged original artwork crops.
The selected figures and their provenance live under `docs/assets/`. Rebuilding
the site does not require or copy private manuscript documents or editable slides.

## Interface and execution

`app.py` delegates to `ui/app.py`. The `overview`, `methods`, `explorer` and `run` modules
share `ui/theme.py` and the scientific chart functions. Historical settings are
retained for audit in the repository and are not displayed in the interface.

`execution.py` launches one `worker.py` subprocess per web run. Keys are passed
through the child environment, not command-line arguments or files. A temporary
CSV and result file are deleted when execution finishes. This prevents provider
clients or process environment changes from leaking between Streamlit sessions.
Numerical mode validates uploaded columns against the selected dimensional basis.
Custom definitions pass through `definitions.validate_definition`: finite matrices,
matching variable order, a complete independent null-space basis and D Bᵀ = 0.
The validated basis reaches the numerical executor, while custom agent runs infer
a new basis and omit benchmark-specific constraints.

`retrieval.py` provides a network-free lexical retrieval path over the original
knowledge files. `ui/knowledge.py` distinguishes this path from actual autonomous
run evidence. Shared constants load before the knowledge module; retriever
instances are owned by that module, avoiding cold-start import cycles.

`scopes.py` maps cluster IDs to physical A/B/C labels and records exact CSV-row subsets and fingerprints. `representations.py` defines reported formulas and per-axis display defaults. Both Streamlit and the static data export use these definitions. `ui/learning.py` separates numerical-slice searches from reported validation stages. `ui/eigenpairs.py` presents the calculated eigenpairs in one bottom inspector; reported eigenpair records remain in the reference archive.
