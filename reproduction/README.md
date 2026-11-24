# Reproduction and recorded evidence

Run commands from the repository root. None of the commands below calls an API. Generated outputs are separate from the published snapshots.

| Task | Command | What is repeated |
|---|---|---|
| Input/archive integrity | `python reproduction/verify_integrity.py` | SHA-256 checks for case data, knowledge bases and historical evidence |
| RAG benchmark | `python -B reproduction/rag_ablation/reproduce.py` | Retrieval/checker replay and summary statistics from 600 recorded trials |
| Velocity comparison | `python -B reproduction/velocity_comparison/reproduce.py` | Six profile transformations, error tables and comparison figure |
| Regional comparison | `python -B reproduction/regime_comparison/reproduce.py` | Regional MI and membership diagnostics using the recorded global IT-pi fit and recorded gradient partition |
| MI landscapes | `python reproduction/mi_landscapes/reproduce.py` | A new CMA-ES search for four cases, optimization histories, grids and figures |

Use the package-specific `--output-dir` argument to select a separate output directory. RAG and the comparisons write to package-local `generated/` directories by default. The MI command writes to `reproduction/mi_landscapes/generated/optimize/`.

Install the corresponding dependency set in [environments](../environments/README.md). Original package requirements and verification reports are retained as historical records; those reports do not certify a fresh execution on another machine.

## Scope and provenance

- `rag_ablation/` retains all 600 trial identities, 730 recorded Checker rounds, source identity audits, request/response evidence and `SHA256SUMS`. Sandgrain's 60 rounds support full semantic ranking replay using archived query vectors. The other 670 rounds replay recorded top-k scores; complete historical query vectors are unavailable. These are retrieval-grounded checking and revision outcomes, not a fresh live-model experiment.
- `regime_comparison/` retains third-party `vendor/IT_PI.py` and its license. Its default entry point does not repeat global IT-pi optimization or infer a fresh gradient partition.
- `velocity_comparison/` retains a historical input CSV whose column order differs from the central velocity dataset. All rows match by column name. Both byte identities are preserved because historical manifests identify the local copy.
- `mi_landscapes/MI_landscape_4cases.ipynb` is the original unexecuted numerical source. The CLI resolves its four datasets to current `data/datasets/` names, records input hashes and exports fresh numerical results. A fresh optimizer trajectory is not claimed to be the historical trajectory used for a published figure.

Internal historical file names, absolute provenance paths and embedded identifiers are unchanged. New public case identifiers are mapped in `configs/cases/`. Use the wrapper entry points: running the original RAG scripts or comparison notebooks directly can overwrite local `results/` snapshots.
