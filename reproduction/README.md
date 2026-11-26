# Reproduction

Run commands from the repository root after installing the corresponding [dependency set](../environments/README.md). These commands use included files and do not require model-service credentials.

| Task | Command | Calculation |
|---|---|---|
| Input integrity | `python reproduction/verify_integrity.py` | Checksums for case data, knowledge bases and comparison inputs |
| LLM ablation and API usage in dimensional checking (SI §5.2) | `python -B reproduction/rag_ablation/reproduce.py` | Dimensional-checking results and API usage for 600 trials |
| Relation to the Trettel–Larsson transformation (SI §5.3) | `python -B reproduction/velocity_comparison/reproduce.py` | Six-profile transformations, RMS-error tables and figure |
| Physical-regime detection in IT–π and AI Buckingham (SI §5.4) | `python -B reproduction/regime_comparison/reproduce.py` | Regional MI calculations and figures using the included global IT–π fit and Cess partition |
| MI optimization landscapes | `python reproduction/mi_landscapes/reproduce.py` | The four Type I cases in Figure 4. Use `--case mhd` for the MHD landscape. |

Use `--output-dir` to choose an output directory. The dimensional-checking and comparison wrappers write to their package-local `generated/` directories by default. The MI-landscape command writes to `reproduction/mi_landscapes/generated/optimize/`.

## Package inputs

- [Dimensional checking](rag_ablation/) includes Analyzer responses, retrieved records, Checker decisions and API usage.
- [Velocity transformations](velocity_comparison/): six compressible profiles and an incompressible reference. The package-local velocity CSV has a different column order from the central dataset; use the inputs named by this package.
- [Physical-regime comparison](regime_comparison/): the IT–π implementation and license, a fitted global two-group representation, regional calculations and the Cess gradient-based partition.
- [MI optimization landscapes](mi_landscapes/) includes the numerical notebook and the settings for the five CMA-ES searches.

The wrapper commands write separate outputs for comparison with the included results. Directly running the source notebooks or the original dimensional-checking scripts can write into the package's `results/` directory; use the wrappers for a separate reproduction run.
