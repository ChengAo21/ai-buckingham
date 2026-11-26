"""Repository coverage, dataset inventory, and the reader's starting points."""
import pandas as pd
import plotly.express as px
import streamlit as st

from .components import chart, masthead, table
from .data import ROOT, case_catalog, dataset_path, read_csv, read_json, reproduction


def render():
    masthead("Discover the scaling. Inspect the evidence.",
             "Explore dimensional reasoning, retrieval grounding, physical regimes, and dimensionless learning across ten physical systems.")
    catalog = case_catalog()
    coverage = read_json(reproduction("rag_ablation") / "results" / "coverage.json")
    publication = read_json(ROOT / "results" / "index.json")
    included_cases = {
        case["case_id"] for case in publication.get("cases", [])
        if any(run.get("run_id") == case.get("selected_run") and run.get("status") == "completed"
               for run in case.get("runs", []))
    }
    records = []
    for case_id, item in catalog.items():
        data = read_csv(dataset_path(case_id))
        records.append({"Physical system": item["title"], "Similarity setting": item["kind"],
                        "Observations": len(data) if not data.empty else None})
    frame = pd.DataFrame(records)
    columns = st.columns(4)
    columns[0].metric("Physical systems", len(catalog))
    columns[1].metric("Observations", f"{int(frame['Observations'].sum()):,}")
    columns[2].metric("Paper case coverage", f"{len(included_cases.intersection(catalog))}/{len(catalog)}")
    columns[3].metric("Dimensional-checking trials", coverage.get("archived_trials", "Unavailable"))
    st.markdown("### Selected results")
    results = [
        ("Local physical regimes", "cess_regimes.png",
         "Gradient-based partition and local scaling in the Cess channel-flow model."),
        ("Equivalent sand-grain roughness", "sandgrain_predictions.png",
         "Offline prediction comparison using surface statistics and learned dimensionless groups on the same train–test split."),
    ]
    for column, (title, filename, caption) in zip(st.columns(2), results):
        with column:
            st.markdown("#### " + title)
            image = ROOT / "docs" / "assets" / filename
            if image.is_file():
                st.image(str(image), caption=caption, width="stretch")
    st.caption("The collapse-validation score evaluates prediction conditional on a selected representation. "
               "It is not an out-of-sample evaluation of the complete discovery workflow.")
    st.markdown("### Follow a result from input to evidence")
    st.markdown("Use **Case workflow** for a complete case, **Retrieval evidence** for variable-level checks, "
                "**Numerical diagnostics** for decisions and optimization, and **Comparisons** for evaluation. "
                "Use **Run analysis** to start a calculation or browse saved runs.")
    st.markdown("### Benchmark collection")
    table(frame, "case_inventory.csv")
    if frame["Observations"].notna().any():
        chart(px.bar(frame, x="Observations", y="Physical system", color="Similarity setting", orientation="h",
                     height=470), "overview_samples")
    with st.expander("About result sources", expanded=False):
        st.markdown("**Paper results** are the included case executions and benchmark records. "
                    "**Offline recalculations** evaluate saved inputs, coefficients or embeddings. "
                    "**New analyses** are user-initiated calculations with their own settings and result bundles.")
