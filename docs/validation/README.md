# Preparation validation

These reports describe verification of the scientific repository and its presentation assets. GitHub Pages publication uses the separate workflow described in [pages.md](../pages.md); the Streamlit application remains a separately runnable Python service.

| Record | Scope |
|---|---|
| [`unit_tests.json`](unit_tests.json) | Complete 76-test offline suite |
| [`web_smoke.json`](web_smoke.json) | Current API-blocked checks across all six Streamlit views |
| [`final_web.json`](final_web.json) | Final checks of actual gradients, verified rough-pipe metrics, zero failed included runs and the pressure-gradient baseline |
| [`publication_alignment.json`](publication_alignment.json) | 309 acceptance checks against the author numerical tables, including all ten cases |
| [`numerical_runtime.json`](numerical_runtime.json) | Verified rough-pipe native thread profile, actual partition statistics and numerical-source provenance |
| [`input_integrity.json`](input_integrity.json) | 4,679 original input/evidence file hashes verified |
| [`presentation.json`](presentation.json) | README asset/link checks, unchanged scientific files, and static Pages validation |
| [`preparation.json`](preparation.json) | Package, notebook and publication checks |
| [`../../results/validation.json`](../../results/validation.json) | Ten actual case results, dimensions, null spaces, 9,853 actual CMA generations, notebooks and original figures |
| [`../../reproduction/publication_validation.json`](../../reproduction/publication_validation.json) | Four reproduction packages, 4,679 hashes, 600 trials/730 rounds, data generators and numerical comparisons |

Run `python -m pytest -q`, `python scripts/check_app.py`, `python scripts/check_web.py`, `python scripts/validate_results.py`, `python scripts/check_publication.py` and `python scripts/check_release.py` from the repository root. App checks block HTTP client sends and do not use API credentials. The overall checksum manifest excludes generated caches and user runs. Its explicit update command is `python scripts/check_release.py --write`.

The preparation environment inherited a non-fatal Requests dependency warning from the author's existing environment. Tests passed with zero failures or skips. Some earlier Windows sandbox attempts needed a writable temporary directory or local socket permissions; these environment issues did not require changes to the numerical algorithms.
