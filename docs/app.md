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
| Overview | Research results, dataset coverage, case inventory and sample counts. |
| Case workflow | Input scatter, variable ledger, dimensions and basis, resolved configuration, mechanism evidence, selected relations, complete result cards and downloads. |
| Retrieval evidence | Select model, physical system, trial, checking round and variable. Inspect exact query, ranked records and score components, citations, dimensional changes and gate audit. |
| Numerical diagnostics | Inspect gradients and candidate K memberships in both log-Pi and physical coordinates. Explore conditional MI contours/3D surfaces and saved CMA-ES generation histories. |
| Comparisons | Filter 600 trial outcomes and 730 recorded checking rounds; inspect Cess regional relevance by M/seed/conditioning and velocity profiles and RMS intervals. |
| Run analysis | Inspect saved-run status, start an analysis from a supplied case or uploaded CSV, and download its result bundle. |

The case pages read `configs/cases/*.json`, `data/datasets/`, and
`results/cases/<case_id>/<run_id>/`. New reader calculations are written under
`generated/cases/<case_id>/<run_id>/`; the explorer labels these separately from
the paper's included runs.

## Evidence and missing records

A saved result, offline recalculation and new analysis have different sources.
The app shows source and status labels, including when records are unavailable.
CMA-ES generation histories are stored in `evidence/optimization_trace.jsonl`.

The optimizer view shows the penalized objective, which can differ from mutual
information. Its two-coordinate path projects a higher-dimensional search;
conditional MI surfaces are available for exported numeric grids.

The reported held-out collapse score is conditional on an already selected
representation. It does not measure generalization of the complete discovery
workflow. Input-group power reparameterizations can be equivalent; output
rescaling exponents are not normalized in the same way.

## Live runs and credentials

Submit the analysis form to start a calculation. Each analysis runs in an
isolated process through `ai_buckingham.cli.run_analysis`.

Supplied keys remain in the current Streamlit session and its analysis process;
they are not saved in the result configuration or repository. The uploaded CSV
is included in the downloadable run bundle.

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
Long CMA-ES searches consume CPU and memory; use a local installation when
more compute or durable run storage is needed. Runtime artifacts require durable
storage if they must survive hosted-app restarts.

Keep deployment credentials in the platform's secret settings, outside the repository;
do not commit `.streamlit/secrets.toml`.

## Additional numerical evidence

`results/mi_landscapes/<case_id>/` supplies the exported 101-by-101 MI grids,
generation histories, run metadata and original figures for the included
landscape calculations. Their source modes distinguish a fresh notebook
optimization from offline grid evaluation over a previously recorded search.
The generation slider only reveals the stored history.

Candidate membership arrays are read from a run's
`evidence/regime_candidates/<record_id>/` or the separately validated
`results/regime_candidates/<case_id>/<run_id>/` supplement. The supplement is
labelled `offline_recalculation_from_recorded_gradients`.
Source hashes and agreement checks remain downloadable.

Case relations include pointwise `tables/*_collapse.csv` and can be explored
interactively. The Sandgrain page additionally reads the separate offline
validation refit, with source/split filters and comparison of the original seven
statistics against the learned groups using the identical 83/21 split.

A toggle in the case downloads view displays the complete saved HTML result
card in a scrollable frame. It is disabled by default.
