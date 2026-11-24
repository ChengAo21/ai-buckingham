# Refactoring and scientific behavior

The engine was extracted from the author-supplied implementation into ordinary Python modules. The original source is not required to install, test or run this repository. Its SHA256 and the original location and AST fingerprint of each definition are retained in `src/ai_buckingham/_source_map.json`.

The source contained 10,529 lines, 234 top-level functions and 18 classes. The extraction used Python's symbol table to distinguish local variables from global references. References to definitions in another module received explicit module qualifiers; function-local names, scientific expressions, signatures, thresholds and default numerical parameters were retained. Modules use normal imports and do not execute source fragments in a shared namespace.

## Module boundaries

| Module | Responsibility |
|---|---|
| `config.py` | Numerical defaults, canonical dataset and knowledge paths |
| `schemas.py` | Typed dimensional ledgers, audit results, plans and workflow state |
| `bindings.py` | Explicit case input/output contracts and name-based order binding |
| `models.py` | Explicit credentials and model-client initialization |
| `provenance.py` | Node timings, native HTTP attempts, token usage and runtime identity |
| `knowledge.py` | Knowledge-table parsing, lexical/semantic retrieval and embedding caches |
| `dimensions.py` | Dimensional contracts, physical-consistency checks and Buckingham-Pi bases |
| `regimes.py` | GPR gradient surrogate, gradient clustering and regime selection |
| `optimization.py` | Mutual information, CMA-ES, similarity screening, group-count selection, sparsification and rationalization |
| `validation.py` | Held-out GPR validation and subsampling uncertainty |
| `workflow.py` | Planning, compilation, numerical execution, replanning and interpretation |
| `graph.py` | LangGraph construction and routing |
| `presentation.py` | Notebook cards and human-readable reports |
| `plotting.py` | Regime partitions, model evidence, collapse plots and GPR surfaces |

The handwritten `bindings.py`, `paths.py` and `recording.py` support input contracts, run isolation and observation. The repository's `artifacts.py`, `cli.py`, `plot_data.py` and web modules add publication, execution and browsing interfaces. These new definitions are outside the original 252-definition audit.

## Audit of extracted definitions

The comparison parsed both implementations with the same Python AST parser, replaced only cross-module qualified references with their original global names, and compared ASTs without source-position attributes. The initial structural extraction retained 240 identical definitions. After adding the explicit input-order boundary and its typed evidence fields, **238 of the 252 original definitions are identical under that normalization**, including 17 of the 18 original classes. The other 14 definitions have the explicit adaptations below; the numerical dimensional compiler and basis calculator remain unchanged.

| Definition | Explicit adaptation |
|---|---|
| `_metadata` | Read the package identity through its owning module |
| `_publish` | Resolve the provenance renderer through its owning module |
| `observe_workflow_node` | Resolve the basis routing function through its owning module |
| `AdaptiveWorkflowState` | Declare raw Analyzer-ledger and input-binding evidence fields |
| `_assert_private_cache_write` | Protect published knowledge files as well as published datasets |
| `_private_rebuild_cache_path` | Route missing or stale published indexes into a private writable cache |
| `save_State_2md` | Resolve the report's displayed package version through its owning module |
| `detect_gradient_regimes` | Export existing candidate arrays before non-final labels are omitted from the returned summary |
| `_run_local_cma_once` | Observe completed optimizer generations and endpoints |
| `_plot_generic_regime_partition` | Export the same figure as PDF and SVG after the original PNG |
| `adaptive_visualizer_node` | Export the same evidence and surface figures as PDF and SVG |
| `initialize_models` | Apply the original random seeds once at explicit startup |
| `analyze_problem_node` | Bind explicitly configured input order by exact names and retain the raw reply |
| `display_dimagent_graph` | Build a graph explicitly when none is supplied |

Changes to module-level configuration are separate from those counts. Case prompts and manually chosen parameters moved into `configs/cases/`; dated case globals and implementation-version filenames were removed. Dataset CSV columns and values were retained. Model clients and retriever instances remain live references to their owning modules, so initialization does not leave another module holding an outdated `None`. The tool registry resolves its original callable objects lazily to avoid circular-import ordering problems.

The original module performed graph construction, graph display, client initialization and global random seeding during import. These actions now occur through explicit entry points. `initialize_models()` initializes clients and applies the original Python and NumPy seeds once per process. `build_agent_graph()` constructs the graph without credentials or network requests. Package versioning is centralized at `1.0.0`; it does not appear in scientific module or dataset filenames.

Conventional public names include `build_agent_graph`, `save_run_report`, `estimate_mutual_information`, `combine_pi_basis`, `evaluate_power_product` and `format_power_product`. Their original names remain available as aliases where compatibility matters. Existing LangGraph node IDs and state keys, including `csv_FilePath`, remain compatible with recorded evidence.

## Explicit input order and execution profiles

For a configured case, `input_columns` and `output_column` enter the graph as `frontend_expected_inputs` and `frontend_expected_output`. The Analyzer receives those explicit contracts. If its input-name list or variable-record list arrives in a different order, the boundary adapter matches records by their exact `csv_name` and applies the configured order before calling the unchanged dimensional compiler. It does not assign dimensions by position, change SI exponents, fill missing variables or replace a computed basis. Extra, missing and duplicate inputs remain errors. A custom analysis without a configured input order retains the original Analyzer-order behavior.

The typed graph state retains `analyzer_raw_ledger` before output-role repair or input-order binding. A separate `input_order_binding` record stores the observed input order, observed variable-record order, bound order, authority and whether reordering occurred. This preserves the original answer while making the numerical input order stable and reviewable.

The rough-pipe investigation also established that the native linear algebra thread profile matters for its ill-conditioned affine-sigmoid GPR fit: the original 20-thread profile reproduced the manuscript spectrum, while a wrapper forcing one thread produced different fitted gradients. This is an execution-profile difference, distinct from the input-order safeguard. Case execution and provenance must retain the declared numerical thread profile; the GPR formulas, optimization objective, stopping criteria and default scientific parameters are unchanged.

## Scientific rules retained

The refactor preserves the MI estimators, sample-dependent estimator selection, GPR options, gradient-coordinate transformations, clustering initialization and acceptance rules, score thresholds, optimization bounds, stopping criteria and explicit case constraints. It preserves the sequence of group-count search, sparsification and rationalization.

CMA restarts retain the **primary-seed candidate**. Additional successful restarts provide diagnostic uncertainty and stability statistics, even when one has higher MI. A failed primary restart is reported as `primary_restart_failed`; a successful diagnostic restart does not silently replace it. The tests exercise both rules.

The output path is resolved through a `ContextVar`. Within `record_run(run_dir)`, existing plotting and report code writes beneath that run directory. Leaving the context restores the previous path, including after an exception. Published datasets and knowledge files are read-only inputs to runtime cache handling. Missing or stale supplied embedding indexes are rebuilt under `generated/cache/` or an explicitly configured private cache.

## Observing optimization and candidate regimes

`evidence/optimization_trace.jsonl` contains observations of actual optimizer work. Each CMA generation is recorded immediately after the original `es.tell(solutions, function_values)`. The observer reads that evaluated population and the optimizer's existing state. It does not call `ask`, `tell`, the objective or an MI estimator again, and does not sample a scientific random generator.

Generation records include scope, phase, search identity, seed, candidate input-group count, generation number, objective statistics, sigma, cumulative evaluations, internal mean coefficients and evaluated best coefficients. Reconstructed mean basis coefficients and the basis matrix support interpretation. The internal mean is an optimizer state, not an additional evaluated candidate. Objective values include the original infeasibility and collinearity penalties; the displayed objective must not be presented as raw MI. Direct evaluation and bounded scalar optimization have endpoint records with `generation: null`, without invented CMA generations.

An observation failure is recorded in `evidence/optimization_trace_warnings.json`; it does not substitute scientific values or select a different optimizer result.

Each regime assessment can also write a directory under `evidence/regime_candidates/`. It contains `coordinates.npz`, per-candidate `k_<K>.npz` labels and confidence, per-candidate JSON diagnostics and a manifest. These are the existing candidate partitions captured before the original summary removes non-final labels. No new partition is computed. The original returned workflow summary is unchanged. Follow the manifest's row-mapping information and the workflow's original CSV index mappings when aligning these arrays with source samples.

## Verification performed

During extraction, the author source was parsed into an isolated validation namespace after removing startup invocation and display calls. No model requests were made. Original and extracted numerical functions were then run in the same environment with identical synthetic data, parameters and random states.

| Comparison | Observed result |
|---|---|
| Two-parameter real CMA search, three generations | Identical MI and physical exponents; three real generation records plus an endpoint |
| Direct evaluation with all coefficients constrained | Identical successful result and physical exponents |
| One-parameter scalar search with a deliberately tiny iteration budget | Both implementations returned `optimization_failed` |
| Complete local workflow on 50 positive synthetic samples, two restarts and a small search budget | Both succeeded, locked one input group, and returned identical complete `final_solution` dictionaries |
| Observation of that complete local workflow | 16 completed CMA generations, correct regime scope, two distinct search identities, no observation warnings |
| Module-global symbol audit | All referenced module globals resolve after extraction |

These small-budget checks validate the refactor and observation contract; they are not publication benchmark runs. Published runs retain their configured scientific budgets and are separately recorded with their real status and provenance.

The independent repository test `tests/test_scientific_contract.py` does not load the original author source. It verifies graph construction without credentials or network, real CMA equality with recording on and off, unchanged scientific RNG states and MI-evaluation counts, the primary-seed policy, published-input cache protection, unchanged candidate arrays during export, and nested output-path restoration. Run it with:

```bash
python -m pytest tests/test_scientific_contract.py
```

`tests/test_input_bindings.py` additionally checks that independently permuting the Analyzer's input names and variable records gives identical dimensional matrices, computed bases and Schmidt coordinates under an explicit contract. It tests raw-reply retention through the real Analyzer node and typed graph state, CLI contract propagation, custom-order preservation, duplicate/missing-input rejection, unchanged provisional-coordinate handling, and separation of validation and regime random seeds. These tests use offline model doubles and make no API requests.

## Maintainer extraction baseline

`scripts/refactor_source.py` requires an explicitly supplied `--source` path and validates that file's SHA256 against the audited source fingerprint in `src/ai_buckingham/_source_map.json`. Its line-based extraction rules apply only to that exact source version. A different source is rejected before any output directory is created. The supplied Python file is parsed, never imported or executed.

By default the script stages the original 252 definitions in 13 modules under `generated/extraction/ai_buckingham/`. An optional `--output-dir` selects another new staging directory. Existing output directories and the maintained `src/ai_buckingham` tree are refused, so this tool cannot overwrite subsequent source changes.

```bash
python scripts/refactor_source.py --source /path/to/audited_author_source.py
python scripts/refactor_source.py --source /path/to/audited_author_source.py --output-dir generated/inspection/ai_buckingham
```

The staged files are an inspection baseline, not an installable reconstruction of the maintained package. Later input bindings, typed evidence fields, execution profiles and handwritten support/interface modules are not automatically rebuilt. Compare staged definitions with the maintained source and apply intentional changes through normal review; do not copy the staging tree over the package. Ordinary users and the test suite do not need the author source or this tool. Temporary extraction output can be removed after inspection and is excluded from the published payload.
