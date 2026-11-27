# AI Buckingham dimensional-checking benchmark

This package reproduces the LLM ablation and API usage in dimensional checking for ten physical systems, two prompt variants per system, ten trials per variant, and three LLMs for semantic parsing. GPT-5 mini performs dimensional checking and output-column verification. Each trial permits one revision followed by one further dimensional check.

## Reproduce the results

Use Python 3.13 and run these commands from this directory. Offline verification and statistics use the packaged files and require no API credentials.

    python -m pip install -r requirements.txt
    python -B reproduce.py

The wrapper verifies the packaged hashes, regenerates the CSV and JSON tables, and compares them with `results/`. For output options, see [REPRODUCTION.md](REPRODUCTION.md).

## Inputs

| Path | Contents |
|---|---|
| trials/ | 600 trials, including initial responses, dimensional reviews, revisions, basis outcomes, output-column checks, and API usage. |
| reference/schedule.json | The 200 case, prompt-variant, and repetition identifiers used for each LLM, with full prompts and dataset hashes. |
| reference/Data_csv/ | Ten datasets and the physical-quantity knowledge base. |
| reference/physical_quantity_index.npz | The 247 knowledge-base document embeddings, with 2,048 coordinates each. |
| reference/expected_dimensions.json | The reference input dimensions and requested output columns. |
| rag.py | Physical-quantity retrieval, evidence formatting and dimensional checking. |
| raw_evidence.py | Checks of HTTP request records, response metadata, model replies and API usage. |

The original and challenge prompt variants share a dataset, variable symbols, requested output, and explicit formula constraints. The reworded variant describes one input through its defining relation or physical meaning.

The Sandgrain reference inputs Ra, kt, and k_rms have length dimension. Sk, Ku, ES, and Po are dimensionless, as is the requested output ks/Ra.

## Scoring

A successful trial passes dimensional checking, reaches basis construction, contains exactly the reference input set with the correct seven SI exponents for every input, and uses the intended response column in both the checked ledger and the resulting basis. The SI bases are mass, length, time, electric current, thermodynamic temperature, amount of substance, and luminous intensity.

The dimensional records submitted for checking are scored against the reference dimensions and requested output column. First-pass success meets these criteria on the first check. A recovery meets the criteria after one revision and a second check. Every case and prompt-variant cell contains ten scheduled trials, including failures.

## Generated results

| File in results/ | Interpretation |
|---|---|
| table1_success.csv | First-pass successes + recoveries = total successes / trials. |
| table2_api_usage.csv | Stage token totals pooled over 20 trials per case and LLM, with summed API time and its per-trial mean. |
| table30_display.csv | The supplementary table display, with Init, Emb1, Chk1, Rev, Emb2, Chk2, Sum, and API. Tokens are thousands per scheduled trial, rounded half up to two decimals. API time is seconds per scheduled trial, rounded half up to one decimal. |
| api_usage_full_precision.csv | Pooled totals and unrounded means, with separate first- and second-round embedding use. |
| api_usage_by_variant.csv | Detailed tokens, durations, attempts, and missing usage by case, model, and prompt variant, including output-column verification. |
| model_aggregates.json | Per-model success rates, improvement, revision-token fraction, and average API time. |
| trial_audit.csv, coverage.json | Per-trial reference scores, archive-identity checks, and aggregate API accounting. |
| recorded_pipeline_outcomes.csv | Recorded gate and basis outcomes before reference scoring. |
| rag_replay_rounds.csv, rag_replay_summary.json | Verification of retrieved records and dimensional-checking messages and decisions for all 730 rounds. |
| raw_evidence_audit.json | Checks of HTTP records and API usage. |

Sum includes the initial response, first dimensional check, revision and second dimensional check. Emb1 and Emb2 report embedding tokens separately. Token totals sum reported input and output usage. API time is the summed request duration divided by the scheduled trial count.
