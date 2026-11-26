# Interactive application

[Open the Streamlit application](https://ai-buckingham.streamlit.app/) to browse the included case results without installation or model credentials.

To run it locally, use Python 3.12 and run from the repository root:

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

## Views

| View | Contents |
|---|---|
| Overview | The ten physical systems and their results |
| Case workflow | Inputs, dimensional records, Buckingham–Π bases, settings, selected dimensionless groups, data collapse and downloads |
| Retrieval evidence | Queries, ranked knowledge-base records and dimensional-checking decisions |
| Numerical diagnostics | GPR gradients, candidate partitions, MI optimization landscapes and CMA-ES histories |
| Comparisons | Dimensional-checking trials, Cess physical-regime comparison and velocity transformations |
| Run analysis | Case selection, uploaded data, model configuration and analysis progress |

The case pages read `configs/cases/*.json`, `data/datasets/` and `results/cases/<case_id>/<run_id>/`. New analyses are written under `generated/cases/<case_id>/<run_id>/`.

CSV tables, figures and result records can be downloaded from the case views. The [result-file guide](evidence.md) describes the numerical records and validation settings.

## Running an analysis

Choose a supplied case or upload a CSV with a problem description, configure the model services and submit the analysis form. Each analysis runs through `ai_buckingham.cli.run_analysis`.

The form uses the configured LLM and embedding providers. Starting an analysis sends the problem description, dataset previews and retrieval queries to those services. Host environment credentials can be used when the corresponding form fields are empty.

Supplied keys remain in the current Streamlit session and analysis process; they are not saved in the result configuration or repository. The uploaded CSV is included in the downloadable run bundle. See [`.env.example`](../.env.example) for environment-variable names.

Run progress includes completed workflow nodes and analysis status. Cancellation leaves a partial run with a cancelled status. Reduced computation budgets are labelled in the run configuration.

## Hosted application

[GitHub Pages](https://chengao21.github.io/ai-buckingham/) presents the case results. Streamlit provides the interactive workflow and analysis interface.

New analyses require configured model services and sufficient CPU time and memory. Use a local installation for longer calculations or persistent output storage. Hosted runtime files require durable storage to survive application restarts.

Keep deployment credentials in the platform's secret settings; do not commit `.streamlit/secrets.toml`.

## Numerical downloads

MI landscape files under `results/mi_landscapes/<case_id>/` include grids, optimizer histories, settings and figures. The generation slider selects a point in the saved history.

Candidate partitions are stored under a run's `evidence/regime_candidates/<record_id>/` or the `results/regime_candidates/<case_id>/<run_id>/` supplement. Manifests identify the array sources and sample mappings.

The sand-grain roughness prediction table compares the seven surface statistics and two rationalized groups using the same 83/21 train/test split. Source and split filters select the corresponding samples.
