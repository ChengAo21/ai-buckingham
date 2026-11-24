# Reproduction dependency sets

Use Python 3.12 for numerical reproduction; the supplied verification environment was Python 3.12.12 on Windows. The files below pin direct scientific dependencies, not every transitive dependency or every operating system.

| File | Purpose |
|---|---|
| `data-generation.txt` | Four numerical input generators |
| `mi-landscapes.txt` | Four fresh CMA-ES landscape searches |
| `comparisons.txt` | Velocity and regional IT-pi comparisons |
| `rag-replay.txt` | Historical RAG replay with original numerical/schema versions |

For example, create an isolated environment and run `python -m pip install -r environments/comparisons.txt`. Regional comparison also uses included third-party IT-pi code and license.

The RAG archive originally specified Python 3.13 and NumPy 2.3.1/pandas 2.3.0/pydantic 2.10.3. Its preserving wrapper was also verified in supplied author records on Python 3.12.12. Original `requirements.txt` remains unchanged. Use a separate environment to retain these historical pins rather than silently replacing them with comparison versions.

These commands need no API keys. Live agent execution and generation of new embedding indexes are separate from replaying recorded evidence.
