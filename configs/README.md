# Case configuration contract

`cases.json` lists the ten public case IDs. Each `cases/<case_id>.json` contains:

- `dataset`: repository-relative path to the author-supplied CSV in `data/datasets/`.
- `prompt`: exact problem prompt extracted from supplied current agent definitions.
- `manual_hyperparams`: exact declared overrides, including scientific constraints, preprocessing, numerical settings and plotting controls. JSON arrays represent source Python tuples without changing numeric values.
- `input_columns` and `output_column`: scientific inputs and response, using unchanged CSV spellings. Other columns remain available and are listed as auxiliary columns.
- `expected_dimensions`: reference seven-axis SI exponents from the recorded dimensional-checking benchmark. These validate outcomes without replacing the Analyzer/Checker workflow.
- `data_metadata`: row count, literal column order, positive-response row count and SHA-256.
- `provenance`: source identity and original source symbols.

The seven SI axes are mass, length, time, electric current, thermodynamic temperature, amount of substance and luminous intensity: `M, L, T, I, Theta, N, J`.

Case settings override shared scientific defaults through the existing agent interface. No missing override has been invented. Formula constraints stated in the velocity/Goldenfeld prompts and the `Sk + 2` Sandgrain preprocessing are retained.
