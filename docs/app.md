# Interactive evidence explorer

[Open the hosted Streamlit evidence explorer](https://ai-buckingham.streamlit.app/)
to browse the complete recorded workflow without installation or model credentials.
The app reads the evidence committed in this repository.

To run the same explorer locally, use Python 3.12 and run from the repository root:

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

## Six views

| View | Available interactions |
| --- | --- |
| Overview | Dataset coverage, case inventory, sample counts and saved-run coverage. |
| Case workflow | Input scatter, variable ledger, dimensions and basis, resolved configuration, mechanism evidence, selected relations, complete result cards and downloads. |
| Retrieval evidence | Select model, physical system, trial, checking round and variable. Inspect exact query, ranked records and score components, citations, dimensional changes and gate audit. |
| Numerical diagnostics | Inspect recorded gradients and all saved candidate K memberships in both log-Pi and physical coordinates. Select an exported conditional MI contour/3D surface and its genuine recorded generation history, or inspect the full live CMA-ES trace. |
| Comparisons | Filter 600 trial outcomes and 730 recorded checking rounds; inspect Cess regional relevance by M/seed/conditioning and velocity profiles and RMS intervals. |
| Run analysis | Start an explicit model-backed LangGraph calculation from a supplied case or uploaded CSV and download its independently recorded evidence bundle. |

The case pages read `configs/cases/*.json`, `data/datasets/`, and
`results/cases/<case_id>/<run_id>/`. New reader calculations are written under
`generated/cases/<case_id>/<run_id>/`; the explorer includes these alongside immutable included runs. They are local runtime artifacts and
should not be committed as historical paper evidence.

## Evidence and missing records

A historical record, offline recalculation and new execution are different
sources. The app shows source/status information and reports missing records
instead of marking an unrecorded stage complete. A final figure does not imply
that historical per-generation optimizer records exist. CMA-ES generations are
read from `evidence/optimization_trace.jsonl`, using actual `optimizer_generation`
events; a final optimizer event is not treated as a generation.

The optimizer view labels the penalized objective explicitly. It does not label
the objective as mutual information or fabricate intermediate generations. Its
two-coordinate path is a projection of an evolving high-dimensional search,
not a conditional MI surface. A conditional surface is shown only when an
exported numeric grid is available.

The reported held-out collapse score is conditional on an already selected
representation. It does not measure generalization of the complete discovery
workflow. Input-group power reparameterizations can be equivalent; output
rescaling exponents are not normalized in the same way.

## Live runs and credentials

No live calculation starts while a page is being loaded or a parameter is being
edited. A reader must submit the analysis form. Each job starts a separate Python
interpreter, which initializes the actual model clients and runs the actual
LangGraph through `ai_buckingham.cli.run_analysis`. This prevents the numerical
implementation's global clients from being shared between concurrent sessions.

Credentials are passed to the child through stdin rather than command-line
arguments, result JSON or shared caches. Supplied keys remain in the current
Streamlit session and its execution process; they are not written by the app to
the repository. The live runner disables implicit remote tracing. The uploaded
CSV is intentionally saved inside its run bundle so the execution is inspectable.

The graph uses the LLM and embedding providers configured in the form. Launching
a calculation sends the problem description, relevant dataset previews and
retrieval queries to those providers. Configured host environment credentials
can be used when the corresponding form fields are empty.

Run progress refreshes in a small Streamlit fragment. Completed node events,
the current manifest and generation-trace availability are visible while a job
runs. Cancellation retains partial evidence with a cancelled status. Reduced
budgets are labelled exploratory and do not reproduce the manuscript budget.

## Hosted application

The public application is hosted at [ai-buckingham.streamlit.app](https://ai-buckingham.streamlit.app/),
separately from the [GitHub Pages research page](https://chengao21.github.io/ai-buckingham/).
GitHub Pages serves static results; Streamlit provides the interactive Python interface.

Saved evidence can also be browsed locally without API credentials. New analyses
require the configured model services and enough compute for their selected budgets.
Long CMA-ES searches consume actual CPU and memory; use a local installation when
more compute or durable run storage is needed. Runtime artifacts require durable
storage if they must survive hosted-app restarts.

Keep deployment credentials in the platform's secret settings, outside the repository;
do not commit `.streamlit/secrets.toml`.

## Additional recorded numerical evidence

`results/mi_landscapes/<case_id>/` supplies the exported 101-by-101 MI grids,
generation histories, run metadata and original figures for the included
landscape calculations. Their source modes distinguish a fresh notebook
optimization from offline grid evaluation over a previously recorded search.
The generation slider only reveals the stored history.

Candidate membership arrays are read from a run's
`evidence/regime_candidates/<record_id>/` or the separately validated
`results/regime_candidates/<case_id>/<run_id>/` supplement. The supplement is
explicitly labelled `offline_recalculation_from_recorded_gradients`; the viewer
never implies that these later arrays were observed in the original run.
Source hashes and agreement checks remain downloadable.

Case relations include pointwise `tables/*_collapse.csv` and can be explored
interactively. The Sandgrain page additionally reads the separate offline
validation refit, with source/split filters and comparison of the original seven
statistics against the learned groups using the identical 83/21 split.

A toggle in the case downloads view displays the complete saved HTML result
card in a scrollable frame. It is disabled by default and does not execute a
calculation. All data downloads and inline figures are loaded from local files.
