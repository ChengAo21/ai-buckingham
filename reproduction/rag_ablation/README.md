# AI Buckingham dimensional-checking benchmark

This package reproduces the retrieval-grounded dimensional-checking statistics for ten physical systems, two prompt variants per system, ten trials per variant, and three Analyzer models. GPT-5 mini performs dimensional checking and output-column verification for every Analyzer. Each trial permits one Analyzer revision followed by one further dimensional check.

## Reproduce the results

Use Python 3.13 and run these commands from this directory. The offline replay and statistics use the packaged files and require no API credentials.

    python -m pip install -r requirements.txt
    python -B validate_rag.py
    python -B summarize.py

The scripts regenerate the CSV and JSON files in results/. SHA256SUMS records the supplied file hashes. On systems with sha256sum, run "sha256sum -c SHA256SUMS" to verify the package. Git attributes preserve the archive bytes through checkout. Historical absolute paths in the JSON identify the original runs. The replay resolves its inputs within this directory.

## Inputs

| Path | Contents |
|---|---|
| trials/ | 600 trials, including initial Analyzer replies, dimensional reviews, revisions, basis outcomes, output-column checks, and API provenance. |
| reference/schedule.json | The 200 case, prompt-variant, and repetition identifiers used for each Analyzer, with full prompts and dataset hashes. |
| reference/Data_csv/ | Ten datasets and the physical-quantity knowledge base. |
| reference/physical_quantity_index.npz | The 247 knowledge-base document embeddings, with 2,048 coordinates each. |
| reference/expected_dimensions.json | The reference input dimensions and requested output columns. |
| reference/source_identity.json | Source hashes, scientific-definition comparisons, and the adaptations used for offline replay. |
| reference/*source_equivalence.json | Original Sandgrain source-identity audit records. |
| rag.py | Physical-quantity retrieval, evidence formatting, Checker message construction, and audit decision logic. |
| raw_evidence.py | Verification of the archived HTTP requests, responses, usage, and retry receipts. |

The original and challenge prompt variants share a dataset, variable symbols, requested output, and explicit formula constraints. The reworded variant describes one input through its defining relation or physical meaning.

Sandgrain includes 60 initial.json, binding.json, and on.json triples. Its requests/<attempt_id>/ folders preserve 600 successful HTTP requests and responses, including 420 query embedding vectors. The 22 audit/retry_accounting_only.json files account for 46 transport failures. Each retry retains its identifier, stage, request hash, and duration. All identifiers and source records are preserved.

The Sandgrain reference inputs Ra, kt, and k_rms have length dimension. Sk, Ku, ES, and Po are dimensionless, as is the requested output ks/Ra.

## Scoring

A successful trial passes dimensional checking, reaches basis construction, contains exactly the reference input set with the correct seven SI exponents for every input, and uses the intended response column in both the checked ledger and the resulting basis. The SI bases are mass, length, time, electric current, thermodynamic temperature, amount of substance, and luminous intensity.

First-pass success meets these criteria on the first check before any Analyzer revision. It uses the ledger submitted for checking after the recorded deterministic preparation, including removal of a duplicate output entry or provisional filling of null coordinates with zeros that require verification. A recovery meets the criteria after one revision and a second check. Every case and prompt-variant cell contains ten scheduled trials, including failures.

For GPT-4.1-mini/MHD_challenge_02, the second dimensional check passed and a basis was built, but the revised output changed from u* to u. The reference output criterion excludes this trial. recorded_pipeline_outcomes.csv preserves counts based on the recorded workflow before reference scoring.

## Generated results

| File in results/ | Interpretation |
|---|---|
| table1_success.csv | First-pass successes + recoveries = total successes / trials. |
| table2_api_usage.csv | Stage token totals pooled over 20 trials per case and Analyzer, with summed API time and its per-trial mean. |
| table30_display.csv | The supplementary table display, with Init, Emb1, Chk1, Rev, Emb2, Chk2, Sum, and API. Tokens are thousands per scheduled trial, rounded half up to two decimals. API time is seconds per scheduled trial, rounded half up to one decimal. |
| api_usage_full_precision.csv | Pooled totals and unrounded means, with separate first- and second-round embedding use. |
| api_usage_by_variant.csv | Detailed tokens, durations, attempts, and missing usage by case, model, and prompt variant, including output-column verification. |
| model_aggregates.json | Per-model success rates, improvement, revision-token fraction, and average API time. |
| trial_audit.csv, coverage.json | Per-trial reference scores, archive-identity checks, and aggregate API accounting. |
| recorded_pipeline_outcomes.csv | Recorded gate and basis outcomes before reference scoring. |
| rag_replay_rounds.csv, rag_replay_summary.json | Offline retrieval, message, and audit replay for every recorded Checker round. |
| raw_evidence_audit.json | Integrity checks for the 600 successful raw HTTP bundles and 46 transport-failure receipts. |

Sum includes the initial Analyzer, first dimensional Checker, Analyzer revision, and second dimensional Checker. Emb1 and Emb2 report embedding tokens separately. The detailed outputs also retain output-column verification tokens. API duration includes all recorded requests and retries, including output-column verification. It is summed request time divided by the scheduled trial count.

API accounting deduplicates cumulative provenance by attempt_id. The package contains 6,584 distinct attempts and removes 202 repeated provenance entries. Token totals sum reported input and output usage. All attempts have durations. Token usage is absent for 319 transport failures, which remain included in request counts and duration sums.

## Replay and source identity

The replay verifies query text, literal knowledge-base records, lexical and hybrid scores, evidence, Checker messages and responses, and deterministic gate decisions for 730 Checker rounds. The original nine cases contribute 670 rounds. Sandgrain contributes 60 rounds, with 54 first-pass successes and six terminal evidence-contract failures.

For Sandgrain, the archived query embeddings and packaged document index allow recomputation of the full semantic ranking across all 247 knowledge-base records. A tolerance of 1e-12 applies only to semantic and hybrid scores because numerical libraries can differ in the last floating-point bit. Record identities, rank order, evidence, messages, hashes, and gate decisions must match exactly.

For the other cases, replay uses recorded top-k semantic scores and gives unreturned records a zero lower bound. This checks the returned ordering and whether an omitted record must outrank it even at zero semantic similarity. The original full semantic ranking requires historical query vectors.

The supplied latest agent source and the source recorded for Sandgrain have different whole-file hashes. Their dimensional schemas, physical-quantity retrieval, evidence construction, and checking functions are equal as Python syntax trees. The numerical workflow changes occur after this benchmark's endpoint. The package contains the applicable RAG code and source-comparison metadata.

Initial Analyzer replies and API attempt records exist for all 600 trials. Initial request text is available for 360 trials. The other 240 retain the problem hash and response. Two GPT-5 mini trials reuse archived initial replies with their original usage records.
