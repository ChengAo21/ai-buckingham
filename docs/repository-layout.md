# Repository layout

| Path | Contents |
|---|---|
| `src/ai_buckingham/` | Scientific implementation, CLI and Streamlit views |
| `configs/cases/` | Case prompts, input/output columns and numerical settings |
| `data/datasets/` | Case input CSVs |
| `data/generation/` | Dataset generators and source information |
| `knowledge/` | Physical-quantity and dimensionless-relation knowledge bases and embedding indexes |
| `notebooks/cases/` | Case reports rebuilt from saved results, workflow diagrams and an optional new analysis from the inputs |
| `results/cases/<case_id>/<run_id>/` | HTML reports, system cards, settings, numerical tables, plots and execution records |
| `reproduction/` | Dimensional-checking statistics, comparison calculations and MI-landscape examples |
| `docs/` | Usage, configuration and file-format guides |
| `site/` | Static results explorer |
| `scripts/` | Execution, export and validation commands |
| `tests/` | Automated checks |

Case IDs identify the corresponding configuration, dataset, notebook and run directory. Reference dimensions in case metadata are used for validation. Runtime settings are resolved from library defaults, planner output and explicit case settings; see [Numerical configuration](configuration.md).

New analyses, caches and reproduction outputs use `generated/`, which is excluded from Git. Each analysis has a separate run directory. Create a new run when changing settings.

Within a run, `result.html` presents the analysis cards and result figures. `plots/` is the figure directory shared by the state, reports and viewer. PNG, PDF and SVG files provide raster and vector versions of the figures.

The velocity-comparison package uses its own input CSV and column order. Follow that package's [input and reproduction instructions](../reproduction/velocity_comparison/README.md).

See [Architecture and execution](refactoring.md) for module responsibilities and entry points.
