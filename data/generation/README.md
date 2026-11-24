# Dataset generation

Install `environments/data-generation.txt` in Python 3.12. From the repository root:

```bash
python data/generation/generate.py --output-dir generated/case_data
```

Append any of `blasius goldenfeld cess_channel rough_pipe` to generate a subset. Existing output files are rejected. The command writes `<case_id>.csv` in the selected directory without changing archived `data/datasets/` inputs.

| Case | Rows | Archived construction |
|---|---:|---|
| `blasius` | 200 | Integrate on 100 streamwise and 100 wall-normal coordinates; retain 2% with pandas seed 5 |
| `goldenfeld` | 414 | Convert `nikuradse_data.csv` with NumPy `RandomState(42)` |
| `rough_pipe` | 1,200 | Sample with `default_rng(42)`; `64/Re` for `Re <= 3000`, Colebrook above |
| `cess_channel` | 1,000 | Sample with `default_rng(30)` in archived wall-distance bands; evaluate Cess relation |

Scientific calculations and column orders are unchanged from the supplied script; only public case/file names changed. Original generation was verified with Python 3.12.12, NumPy 2.4.0, pandas 2.3.3 and SciPy 1.16.3 on Windows. Other platforms or solver versions can change final floating-point digits. Central input hashes are in `data/manifest.json`.
