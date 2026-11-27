# Result files and numerical validation

## Run records

The included case results are stored under `results/cases/<case_id>/<run_id>/`. By default, new analyses are written under `generated/cases/<case_id>/<run_id>/`. Each run's manifest identifies its configuration and data sources.

For a complete presentation of a case, use **View results** on [GitHub Pages](https://chengao21.github.io/ai-buckingham/) or **Case results** in [Streamlit](https://ai-buckingham.streamlit.app/). These views use the saved system cards, selected relations and figures. The case notebook rebuilds the same HTML report offline; its final section runs a new analysis from the case data and configuration.

`result.html` is the standalone case report. `report.html` retains the original run output, and `evidence/display_bundles.json` stores the rich displays used to build the case report. `plots/` contains the result figures in PNG, PDF and SVG formats. The report and numerical views read the selected run's files.

| Workflow stage | Files and fields to inspect |
|---|---|
| Problem formulation | Prompt, input column order, dependent variable and dimensional records |
| Dimensional checking | Retrieval queries, ranked knowledge-base records, dimensional assignments and checking decisions |
| Buckingham–Π basis construction | Ordered variables, dimensional matrix, rank, nullity and basis |
| Workflow planning | Planner output, manual overrides and resolved settings |
| Physical-regime identification | Sample mapping, GPR settings, gradients, candidate partitions and partition-selection criteria |
| Dimensionless learning | Similarity-type screening, candidate group counts, MI and normalized information score ε |
| Sparsification and rationalization | Physical exponents, proposed simplifications and changes in MI and ε |
| Validation | Training/test rows, predictions, GPR settings and data-collapse plots |

`state.json` contains the full run state. Files under `evidence/` provide stage-specific views. `events.jsonl` contains node events and run status updates.

## Optimization records

`evidence/optimization_trace.jsonl` contains optimizer generations and endpoints. Records include the local scope, search identifier, random seed, group count, generation number, evaluation count, objective value and coefficient vectors. Direct evaluation and scalar optimization have endpoint records with no CMA-ES generation number.

The optimization objective includes infeasibility and collinearity penalties.

The primary-seed candidate supplies the reported result. Additional restarts provide optimization-variability estimates, as described in SI §3.5.

## Collapse validation

Test-set R² evaluates predictive collapse conditional on the selected representation, as described in [SI §3.5 of the accompanying paper](https://arxiv.org/html/2611.33961v1). The validation records include the evaluation coordinates, train/test indices and GPR settings.

For sand-grain roughness parameterization, the inputs include the `Sk + 2` transformation. Predictions from the seven surface statistics and the two rationalized groups use the same 83 training and 21 test samples (Figure 5 and SI Table 19).

Samples with non-finite or non-positive values are excluded where logarithmic coordinates are used. The Blasius and compressible-velocity cases contain 200 and 1,109 observations, with 196 and 1,103 used for their regime-identification GPR fits, respectively (SI Table 7).

During sparsification, candidate removal uses normalized input-group exponents. For Type II similarity, the corresponding absolute output-rescaling exponent is checked against the 0.05 threshold without normalization (SI §3.4).

## Case and comparison files

The rough circular pipe dataset uses the prescribed laminar/Colebrook branch switch at Re = 3000 (main text §2.3.2). Its input samples, candidate partitions and regime-wise results are available in the case directory.

Comparison packages document their inputs and commands separately:

- [LLM ablation and API usage in dimensional checking](../reproduction/rag_ablation/) includes trial records, dimensional-checking decisions and summary tables.
- [Relation to the Trettel–Larsson transformation](../reproduction/velocity_comparison/): six velocity profiles, transformations and RMS errors.
- [Physical-regime detection in IT–π and AI Buckingham](../reproduction/regime_comparison/): global IT–π groups, regional relevance scores and Cess partitions.
- [MI optimization landscapes](../reproduction/mi_landscapes/) includes coefficient grids and optimizer histories for MHD and the four Type I cases in Figure 4.
