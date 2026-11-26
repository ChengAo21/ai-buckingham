# Reproduction dependency sets

Use Python 3.12 for numerical reproduction. The files below pin the direct dependencies for each task.

| File | Purpose |
|---|---|
| `data-generation.txt` | Four numerical input generators |
| `mi-landscapes.txt` | Four fresh CMA-ES landscape searches |
| `comparisons.txt` | Velocity and regional IT-pi comparisons |
| `rag-replay.txt` | Dimensional-checking results and API usage tables |

For example, create an isolated environment and run `python -m pip install -r environments/comparisons.txt`. Regional comparison also uses included third-party IT-pi code and license.

`rag-replay.txt` and `reproduction/rag_ablation/requirements.txt` specify NumPy 2.3.1, pandas 2.3.0 and pydantic 2.10.3; that package's guide uses Python 3.13. Use a separate environment for these dependencies and for the numerical-comparison dependencies.

These reproduction commands run without API keys.
