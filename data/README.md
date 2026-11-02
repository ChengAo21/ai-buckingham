# Supplied data

The CSV and knowledge-base files in this directory are unchanged copies of the
user-supplied `Data_csv` directory. Hashes are recorded in `manifest.json`.

See [dataset provenance](../docs/DATA_SOURCES.md) for sample counts, sources,
preparation, auxiliary data and rights. Metadata/source/case columns are not
silently treated as physical input variables. The exact per-case input and
dependent-column assignments are in `src/ai_buckingham/catalog.py`.

The source executor removes nonfinite and near-zero rows (absolute values below
10^-15) for the selected inputs and output. This leaves 196 Blasius and 1,103
compressible-velocity rows. Raw observations are preserved and downloadable.
