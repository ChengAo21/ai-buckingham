# Repository layout

| Path | Contents |
|---|---|
| `src/ai_buckingham/` | Scientific implementation, CLI and Streamlit views |
| `configs/cases/` | Case prompts, input/output columns and numerical settings |
| `data/datasets/` | Case input CSVs |
| `data/generation/` | Dataset generators and source information |
| `knowledge/` | Physical-quantity and dimensionless-relation knowledge bases and embedding indexes |
| `notebooks/cases/` | Executed case notebooks |
| `results/cases/<case_id>/<run_id>/` | Included case results, settings, tables, figures and execution records |
| `reproduction/` | Dimensional-checking statistics, comparison calculations and MI-landscape examples |
| `docs/` | Usage, configuration and file-format guides |
| `site/` | Static results explorer |
| `scripts/` | Execution, export and validation commands |
| `tests/` | Automated checks |

Case IDs identify the corresponding configuration, dataset, notebook and run directory. Reference dimensions in case metadata are used for validation. Runtime settings are resolved from library defaults, planner output and explicit case settings; see [Numerical configuration](configuration.md).

New analyses, caches and reproduction outputs use `generated/`, which is excluded from Git. Each analysis has a separate run directory. Create a new run when changing settings.

Within a run, `plots/` contains figures referenced by the state, and `figures/` contains the copies used by the viewer and reports. PDF files in result directories contain generated figures.

The velocity-comparison package uses its own input CSV and column order. Follow that package's [input and reproduction instructions](../reproduction/velocity_comparison/README.md).

See [Architecture and execution](refactoring.md) for module responsibilities and entry points.
