# Reproduction

Each case's **Reproduce** link on [the results page](https://chengao21.github.io/ai-buckingham/) gives its input CSV, configuration, notebook and analysis command. The [case notebooks](../notebooks/cases/) generate the system's HTML cards from saved results locally. The workflow diagram uses LangGraph's native Mermaid renderer and requires Internet access when rerun. Set `SAVE_HTML = True` to save the report under `generated/reports/`.

To run the complete analysis from the case input, configure the model and embedding services described in [the README](../README.md#run-locally), then enable `RUN_NEW_ANALYSIS` in its notebook or run a command such as `ai-buckingham mhd` from the repository root. Open the new run's `result.html` under `generated/cases/<case_id>/<run_id>/`.

For the comparison calculations and input checks below, install the corresponding [dependency set](../environments/README.md) and run commands from the repository root. These commands use included files and do not require model-service credentials.

| Task | Command | Calculation |
|---|---|---|
| Input integrity | `python reproduction/verify_integrity.py` | Checksums for case data, knowledge bases and comparison inputs |
| LLM ablation and API usage in dimensional checking (SI §5.2) | `python -B reproduction/rag_ablation/reproduce.py` | Dimensional-checking results and API usage for 600 trials |
| Relation to the Trettel–Larsson transformation (SI §5.3) | `python -B reproduction/velocity_comparison/reproduce.py` | Six-profile transformations, RMS-error tables and figure |
| Physical-regime detection in IT–π and AI Buckingham (SI §5.4) | `python -B reproduction/regime_comparison/reproduce.py` | Regional MI calculations and figures using the included global IT–π fit and Cess partition |
| MI optimization landscapes | `python reproduction/mi_landscapes/reproduce.py` | The four Type I cases in Figure 4. Use `--case mhd` for the MHD landscape. |

Use `--output-dir` to choose an output directory. The dimensional-checking and comparison wrappers write to their package-local `generated/` directories by default. The MI-landscape command writes to `reproduction/mi_landscapes/generated/optimize/`.

## Package inputs

- [Dimensional checking](rag_ablation/) includes LLM responses, retrieved records, checking decisions and API usage.
- [Velocity transformations](velocity_comparison/): six compressible profiles and an incompressible reference. The package-local velocity CSV has a different column order from the central dataset; use the inputs named by this package.
- [Physical-regime comparison](regime_comparison/): the IT–π implementation and license, a fitted global two-group representation, regional calculations and the Cess gradient-based partition.
- [MI optimization landscapes](mi_landscapes/) includes the numerical notebook and the settings for the five CMA-ES searches.

The wrapper commands write separate outputs for comparison with the included results. Directly running the source notebooks or the original dimensional-checking scripts can write into the package's `results/` directory; use the wrappers for a separate reproduction run.
