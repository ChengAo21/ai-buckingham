# Numerical validation

| Record | Contents |
|---|---|
| [`publication_alignment.json`](publication_alignment.json) | Comparisons of included case results with the paper's numerical tables |
| [`numerical_runtime.json`](numerical_runtime.json) | Numerical-library versions, thread configuration and rough-pipe partition statistics |

Run the current checks from the repository root:

```bash
python -m pytest -q
python scripts/check_publication.py
python scripts/validate_results.py
python scripts/check_app.py
python scripts/check_web.py
python reproduction/verify_integrity.py
```

These commands inspect saved results and application behavior without starting a model-backed analysis. See [Numerical configuration](../configuration.md) for the settings needed to rerun the cases.
