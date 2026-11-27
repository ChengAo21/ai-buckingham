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
| Overview | The ten physical systems in two columns, with the paper's similarity settings and workflow figures available to expand |
| Case results | System HTML cards, selected relations, data collapse, dimensional checking, Buckingham–Π bases, physical-regime identification, MI landscapes and stage comparisons for one selected run |
| Retrieval evidence | Dimensional-checking results, queries and retrieved records for the 600-trial benchmark |
| Comparisons | LLM ablation and API usage in dimensional checking, interactive Cess partitions and velocity transformations with the paper's comparison figures |
| Run analysis | Open saved results or configure and run a new analysis |

The case pages read `configs/cases/*.json`, `data/datasets/` and `results/cases/<case_id>/<run_id>/`. New analyses are written under `generated/cases/<case_id>/<run_id>/`.

**Case results** displays the selected run's final relations and figures first. Open a stage to read its completed cards and use the related numerical controls. Physical-regime identification includes the gradient field and candidate partitions. Dimensionless learning shows the paper's MI landscapes in 3D and labeled MI and ε bars with ±1 SD uncertainty for each stage. Interactive data collapse uses the paper's default axis scales for each case and region.

The dimensional-basis stage also contains the selected run's retrieval queries, ranked records and dimensional-checking decisions.

The **Downloads** section provides the HTML report, figures and run files. The [result-file guide](evidence.md) describes the numerical records and validation settings.

## Running an analysis

Choose **View saved results**, select a physical system and result, then click **Open case results** to read its cards and figures.

Choose **Run new analysis** to use a supplied case or upload a CSV with a problem description. **Numerical settings** provides controls for similarity, MI estimation, rationalization and the optimization budget. Additional parameters can be edited as JSON. Configure the model services and submit the form to run through `ai_buckingham.cli.run_analysis`.

Completed runs include `result.html` with their analysis cards and figures. The [case notebooks](../notebooks/cases/) can also generate these reports from the paper's saved results offline, or run a new analysis from the input data when `RUN_NEW_ANALYSIS` is enabled.

New analyses use the configured LLM and embedding providers. Leave API-key fields empty to use host environment credentials. The uploaded CSV is included in the downloadable run bundle. See [`.env.example`](../.env.example) for environment-variable names.

Run progress includes completed workflow nodes and analysis status. Cancellation leaves a partial run with a cancelled status. Reduced computation budgets are labelled in the run configuration.

## Hosted application

[GitHub Pages](https://chengao21.github.io/ai-buckingham/) presents the case results. Streamlit provides the interactive workflow and analysis interface.

New analyses require configured model services and sufficient CPU time and memory. Use a local installation for longer calculations or persistent output storage. Hosted runtime files require durable storage to survive application restarts.

Keep deployment credentials in the platform's secret settings; do not commit `.streamlit/secrets.toml`.

## Numerical downloads

MI landscape files under `results/mi_landscapes/<case_id>/` include grids, optimizer histories, settings and figures. The 3D view displays the landscape and its corresponding trajectory.

Candidate partitions are stored under a run's `evidence/regime_candidates/<record_id>/` or the `results/regime_candidates/<case_id>/<run_id>/` supplement. Manifests identify the array sources and sample mappings.

The sand-grain roughness prediction table compares the seven surface statistics and two rationalized groups using the same 83/21 train/test split. Source and split filters select the corresponding samples.

In the Cess comparison, select the number of IT–π regions, M, to compare its partition with AI Buckingham's K = 3 result. The velocity comparison includes interactive profiles and RMS-error bars as well as the paper's figure.
