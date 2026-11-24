"""Repository coverage, dataset inventory, and the reader's starting points."""
import pandas as pd
import plotly.express as px
import streamlit as st

from .components import chart, masthead, table
from .data import case_catalog, dataset_path, read_csv, read_json, reproduction, runs


def render():
    masthead("Discover the scaling. Inspect the evidence.",
             "Explore dimensional reasoning, retrieval grounding, physical regimes, and dimensionless learning across ten physical systems.")
    catalog = case_catalog()
    coverage = read_json(reproduction("rag_ablation") / "results" / "coverage.json")
    replay = read_json(reproduction("rag_ablation") / "results" / "rag_replay_summary.json")
    available_runs = runs()
    statuses = pd.Series([read_json(path / "manifest.json").get("status", "unknown") for path in available_runs], dtype="object")
    columns = st.columns(5)
    columns[0].metric("Physical systems", len(catalog))
    columns[1].metric("Completed runs", int(statuses.isin(["completed", "complete", "success"]).sum()))
    columns[2].metric("Running runs", int(statuses.eq("running").sum()))
    columns[3].metric("Failed / stopped", int(statuses.isin(["failed", "blocked", "cancelled", "warning"]).sum()))
    columns[4].metric("Archived trials", coverage.get("archived_trials", "Unavailable"))
    st.caption(f"{len(available_runs)} saved case records · {replay.get('checker_rounds', 'Unavailable')} recorded historical checking rounds. Running and failed calculations are retained separately from completed evidence.")
    st.markdown("### Follow a result from input to evidence")
    st.markdown("Use **Case workflow** for a complete case, **Retrieval evidence** for variable-level checks, "
                "**Numerical diagnostics** for decisions and optimization, and **Comparisons** for the recorded evaluation. "
                "Browsing these pages does not start a model request. New calculations start only from **Run analysis**.")
    records = []
    for case_id, item in catalog.items():
        frame = read_csv(dataset_path(case_id))
        available = runs(case_id)
        case_statuses = [read_json(path / "manifest.json").get("status", "unknown") for path in available]
        records.append({"case_id": case_id, "Physical system": item["title"], "Similarity setting": item["kind"],
                        "Dataset rows": len(frame) if not frame.empty else None, "Dataset columns": len(frame.columns),
                        "Completed runs": sum(s in {"completed", "complete", "success"} for s in case_statuses),
                        "Running runs": case_statuses.count("running"),
                        "Failed / stopped runs": sum(s in {"failed", "blocked", "cancelled", "warning"} for s in case_statuses),
                        "Data available": not frame.empty})
    frame = pd.DataFrame(records)
    st.markdown("### Benchmark collection")
    table(frame, "case_inventory.csv")
    if frame["Dataset rows"].notna().any():
        chart(px.bar(frame, x="Dataset rows", y="Physical system", color="Similarity setting", orientation="h",
                     height=470, labels={"Dataset rows": "Observations"}), "overview_samples")
    st.markdown("### Reading the evidence")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Historical record**\n\nAn archived reply, numerical result, or figure from the supplied study. Its source and completeness are shown beside the result.")
    with b:
        st.markdown("**Offline recalculation**\n\nA calculation from recorded inputs, coefficients, or embeddings. This does not imply a fresh language-model response or a repeated global search.")
    with c:
        st.markdown("**New analysis**\n\nA user-triggered execution with its own configuration, trace and result bundle. It remains separate from the paper evidence.")
    st.caption("The collapse-validation score evaluates prediction conditional on a selected representation. "
               "It is not an out-of-sample evaluation of the complete discovery workflow.")
