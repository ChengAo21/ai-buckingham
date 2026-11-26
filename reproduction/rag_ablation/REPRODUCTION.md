# Reproduce the dimensional-checking results

With Python 3.13, run the following commands from this directory.

```bash
python -m pip install -r requirements.txt
python -B reproduce.py
```

The command checks the retrieved records and Checker decisions, regenerates the summary tables in a temporary copy and compares them with `results/`. It runs without API credentials.

Regenerated files and `verification.json` are written to `generated/`. To select another directory, use `python -B reproduce.py --output-dir PATH`.
