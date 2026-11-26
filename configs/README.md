# Case configuration contract

`cases.json` lists the ten public case IDs. Each `cases/<case_id>.json` contains:

- `dataset`: repository-relative path to the CSV in `data/datasets/`.
- `prompt`: problem description passed to the agent.
- `manual_hyperparams`: case overrides for constraints, preprocessing, numerical settings and plotting controls.
- `input_columns` and `output_column`: input and response column names. Other columns are listed as auxiliary columns.
- `expected_dimensions`: reference seven-axis SI exponents used for validation.
- `data_metadata`: row count, literal column order, positive-response row count and SHA-256.
- `provenance`: source identifiers.

The seven SI axes are mass, length, time, electric current, thermodynamic temperature, amount of substance and luminous intensity: `M, L, T, I, Theta, N, J`.

Case settings override shared numerical defaults. The velocity-scaling and Goldenfeld prompts specify formula constraints; the sand-grain roughness case specifies the `Sk + 2` preprocessing.
