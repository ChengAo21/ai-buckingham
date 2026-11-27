# Magnetohydrodynamic power generator

[View results](https://chengao21.github.io/ai-buckingham/cards/mhd.html) · [Reproduce](https://chengao21.github.io/ai-buckingham/cards/mhd.html#reproduce) · [Notebook](../../../../notebooks/cases/mhd.ipynb)

From the repository root, configure the model services in [`.env.example`](../../../../.env.example) and run:

```bash
ai-buckingham mhd
```

Open `result.html` in the new `generated/cases/mhd/<run_id>/` directory. The notebook also generates cards from the included result using local files.

| Contents | Files |
|---|---|
| Figures in PNG, PDF and SVG | `plots/` |
| Coordinates and numerical tables | `tables/` |
| Input data and numerical settings | `resolved_config.json` |
| Complete state | `state.json` |
| Stage records and optimizer history | `evidence/` |
| Original system output | `report.html` |
