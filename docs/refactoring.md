# Architecture and execution

The implementation lives in `src/ai_buckingham/`. Notebooks, the CLI and the Streamlit interface call the same scientific modules. Case prompts and settings are defined in `configs/cases/`.

## Module boundaries

| Module | Responsibility |
|---|---|
| `config.py` | Numerical defaults, canonical dataset and knowledge paths |
| `schemas.py` | Typed dimensional ledgers, audit results, plans and workflow state |
| `bindings.py` | Explicit case input/output contracts and name-based order binding |
| `models.py` | Credentials and model-client initialization |
| `provenance.py` | Node timings, HTTP attempts, token usage and runtime identity |
| `knowledge.py` | Knowledge-table parsing, lexical/semantic retrieval and embedding caches |
| `dimensions.py` | Dimensional contracts, physical-consistency checks and Buckingham-Pi bases |
| `regimes.py` | GPR gradient surrogate, gradient clustering and regime selection |
| `optimization.py` | Mutual information, CMA-ES, similarity screening, group-count selection, sparsification and rationalization |
| `validation.py` | Held-out GPR validation and subsampling uncertainty |
| `workflow.py` | Planning, compilation, numerical execution, replanning and interpretation |
| `graph.py` | LangGraph construction and routing |
| `presentation.py` | Notebook cards and human-readable reports |
| `plotting.py` | Regime partitions, model evidence, collapse plots and GPR surfaces |

`paths.py` and `recording.py` handle run isolation and observation. `artifacts.py`, `cli.py`, `plot_data.py` and the web modules provide report export, execution and browsing interfaces.

## Entry points and compatibility

`initialize_models()` initializes clients and applies the Python and NumPy seeds once per process. `build_agent_graph()` constructs the graph without credentials or network requests.

Public names include `build_agent_graph`, `save_run_report`, `estimate_mutual_information`, `combine_pi_basis`, `evaluate_power_product` and `format_power_product`. Compatibility aliases support earlier callable names. LangGraph node IDs and state keys, including `csv_FilePath`, support the saved result records.

## Input order and execution profiles

For a configured case, `input_columns` and `output_column` enter the graph as `frontend_expected_inputs` and `frontend_expected_output`. The Analyzer receives these contracts. The boundary adapter matches variable records by their exact `csv_name` and applies the configured order before dimensional compilation. This adapter binds column order rather than assigning SI exponents, filling missing variables or replacing a computed basis. Extra, missing and duplicate inputs are errors. A custom analysis without a configured input order uses the Analyzer's order.

The graph state stores `analyzer_raw_ledger` before output-role repair or input-order binding. The `input_order_binding` record contains the observed input order, observed variable-record order, bound order, authority and whether reordering occurred.

The rough-pipe affine-sigmoid GPR fit is sensitive to the native linear algebra thread profile. The `rough_pipe` case declares 20 numerical threads in its configuration. Use the case's declared execution profile when reproducing its gradients and partitions. See [Numerical configuration and case exceptions](configuration.md) for thread settings, runtime metadata and case overrides.

## Selection and run isolation

Local learning proceeds through group-count search, sparsification and rationalization. The reported CMA result is the **primary-seed candidate**; additional successful restarts provide diagnostic uncertainty and stability statistics. A failed primary restart is reported as `primary_restart_failed`, even if a diagnostic restart succeeds.

The output path is resolved through a `ContextVar`. Within `record_run(run_dir)`, plotting and report code writes beneath that run directory. Leaving the context restores the previous path, including after an exception. Published datasets and knowledge files are read-only inputs. Missing or stale embedding indexes are rebuilt under `generated/cache/` or an explicitly configured private cache.

## Optimization traces and candidate partitions

`evidence/optimization_trace.jsonl` stores CMA generations after `es.tell(solutions, function_values)`. Each record describes the evaluated population and optimizer state: local scope, phase, search identity, seed, candidate input-group count, generation number, objective statistics, sigma, cumulative evaluations, internal mean coefficients and evaluated best coefficients. Reconstructed mean basis coefficients and the basis matrix support interpretation. The internal mean is an optimizer state, rather than an additional evaluated candidate.

Objective values include infeasibility and collinearity penalties and therefore are not raw MI values. Direct evaluation and bounded scalar optimization produce endpoint records with `generation: null`. Recording warnings are written to `evidence/optimization_trace_warnings.json`.

A regime assessment can also write `evidence/regime_candidates/`, containing `coordinates.npz`, per-candidate `k_<K>.npz` labels and confidence, JSON diagnostics and a manifest. Follow the manifest's row-mapping information and the workflow's CSV index mappings when aligning candidate partitions with source samples.

## Checks

From the repository root, run the offline checks for scientific execution contracts and input-order binding:

```bash
python -m pytest tests/test_scientific_contract.py tests/test_input_bindings.py
```

See [Evidence, reproduction, and scientific interpretation](evidence.md) for result records and their interpretation, and [Numerical configuration and case exceptions](configuration.md) for reproducibility settings.
