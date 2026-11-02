# Contributing

Open an issue describing the physical system, data schema and reproduction steps.
Install development dependencies with `python -m pip install -e '.[app,dev]'`.
Run `python -m pytest -q` and `python scripts/validate_repository.py` before a PR.

Changes to numerical criteria must cite the corresponding manuscript/SI section
and include the resolved settings in the exported result. Preserve the distinction
between manuscript-reported values, fresh computations and reduced-budget previews.
Do not change reported values to match a new run. Do not commit credentials or raw
provider response headers. Rights for contributions follow the repository's
current rights notice; no open-source grant is implied until authors select one.
