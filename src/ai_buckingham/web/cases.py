"""The complete case workflow, without treating absent stages as complete."""
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from .components import chart, figures, interpretation_commentary, json_panel, masthead, selected_run, table
from .data import ROOT, archive_bytes, case_catalog, dataset_path, first, read_csv, read_json


def _values(title, value):
    if value is None or value == {} or value == []:
        st.info(f"{title} was not recorded in this evidence source.")
    elif isinstance(value, str):
        st.markdown(value)
    else:
        json_panel(title, value, key="workflow_" + title)


def render():
    masthead("Case workflow", "Read the input, inspect the decisions, and download the complete result.", "Case study")
    catalog = case_catalog()
    case_id = st.selectbox("Physical system", list(catalog), format_func=lambda key: catalog[key]["title"], key="workflow_case")
    item = catalog[case_id]
    run = selected_run(case_id, key="workflow_run")
    state = run["state"] if run else {}
    config = run["config"] if run and run["config"] else item["config"]
    tabs = st.tabs(["Input & formulation", "Dimensions & basis", "Analysis protocol", "Mechanism evidence", "Selected relations", "Figures & downloads"])
    with tabs[0]:
        prompt = state.get("prompt_problem") or config.get("prompt_problem") or config.get("prompt") or config.get("problem_description")
        _values("Problem description", prompt)
        frame = read_csv(dataset_path(case_id))
        if not frame.empty:
            st.markdown("#### Observations")
            table(frame, f"{case_id}_data.csv", "case_data_download")
            numeric = frame.select_dtypes(include="number").columns.tolist()
            if len(numeric) >= 2:
                xcol, ycol = st.columns(2)
                x = xcol.selectbox("Input coordinate", numeric, key="case_scatter_x")
                y = ycol.selectbox("Response coordinate", numeric, index=len(numeric)-1, key="case_scatter_y")
                chart(px.scatter(frame, x=x, y=y, hover_data=list(frame.columns)[:12], opacity=.75), "case_input_scatter")
        _values("Variable ledger", state.get("dimension_ledger"))
        if case_id == "sandgrain":
            st.caption("The study translates skewness by Sk + 2 before non-integer powers. See the saved preprocessing record for the executed transform.")
    with tabs[1]:
        _values("Dimensional matrix", state.get("dimension_mat", first(state, "dimensional_matrix")))
        basis = state.get("basis_pi_formulas")
        if isinstance(basis, dict) and basis:
            table(pd.DataFrame([{"Group": k, "Expression": v} for k, v in basis.items()]), "basis_formulas.csv")
        else:
            _values("Buckingham–Pi basis", basis)
        _values("Full-precision basis vectors", state.get("basis_pi_matrices"))
        _values("Basis construction evidence", state.get("basis_tool_history"))
    with tabs[2]:
        _values("Resolved configuration", config)
        _values("Planner proposal", state.get("workflow_plan"))
        _values("Configuration provenance", state.get("hyperparameter_provenance"))
        _values("Compiled workflow", state.get("compiled_workflow"))
        st.caption("Explicit user settings take precedence over planner suggestions and defaults. A prescribed policy and an automatic decision are distinct records.")
    with tabs[3]:
        detection = first(state, "regime_detection")
        _values("Mechanism assessment", detection)
        if run and run["events"]:
            table(pd.json_normalize(run["events"]), "workflow_events.csv")
        if not detection:
            st.markdown("The archived Cess partition and regional comparison are available under **Numerical diagnostics** and **Comparisons**.")
    with tabs[4]:
        locals_ = first(state, "local_workflows", default=[])
        for i, local in enumerate(locals_ if isinstance(locals_, list) else []):
            st.markdown("#### " + str(local.get("scope_id", f"Regime {i+1}")).replace("_", " ").title())
            final = local.get("final_solution", {})
            values = st.columns(3)
            values[0].metric("Selected groups", local.get("locked_count", "Not recorded"))
            values[1].metric("Mutual information", str(final.get("mutual_information", "Not recorded")))
            r2 = (final.get("gpr_validation") or {}).get("test_r2_mean", final.get("gpr_test_r2", "Not recorded"))
            values[2].metric("Conditional held-out R²", str(r2))
            inputs = final.get("input_formulas", [])
            if inputs:
                coordinates = ", ".join(f"Π{i+1}" for i in range(len(inputs)))
                st.caption(f"Selected response: Π_out ≈ F({coordinates}). The input coordinates and output rescaling are defined below for this scope.")
            for formula in final.get("input_formulas", []):
                st.code(formula, language=None)
            if final.get("output_dimensionless_formula"):
                st.markdown("**Output rescaling**")
                st.code(final["output_dimensionless_formula"], language=None)
            json_panel("Local evidence " + str(i+1), local, key=f"local_evidence_{i}")
        if not locals_:
            _values("Selected relations", first(state, "selected_relations", "final_solution"))
        interpretation_commentary(state.get("final_report"))
        _values("Interpretation retrieval", state.get("interpretation_knowledge_retrieval"))
        if run:
            st.markdown("#### Interactive data collapse")
            collapse_tables = sorted((run["path"] / "tables").glob("*collapse.csv"))
            if collapse_tables:
                path = st.selectbox("Scope", collapse_tables, format_func=lambda p: p.stem.replace("_collapse", ""), key=f"collapse_scope_{case_id}_{run['path'].name}")
                frame = read_csv(path)
                pi_columns = [c for c in frame if c.startswith("pi_")]
                if pi_columns and "response_transformed" in frame:
                    x = st.selectbox("Dimensionless coordinate", pi_columns, key="collapse_coordinate")
                    logarithmic = st.checkbox("Logarithmic collapse axes", value=True, key="collapse_log")
                    valid = frame[(frame[x]>0)&(frame.response_transformed>0)] if logarithmic else frame
                    chart(px.scatter(valid, x=x, y="response_transformed", log_x=logarithmic, log_y=logarithmic,
                                     hover_data=[c for c in ["csv_row","response_original","output_scale"] if c in valid], opacity=.75), "case_collapse")
                    table(frame, path.name, "case_collapse_download")
            else:
                st.info("Pointwise collapse coordinates are not present in this run.")
        if case_id == "sandgrain":
            prediction_view()
        if case_id == "pressure_gradient":
            pressure_baseline_view()
    with tabs[5]:
        if run:
            figures(run["path"] / "figures", "case_figures")
            for path in sorted((run["path"] / "tables").glob("*.csv")):
                with st.expander(path.name):
                    table(read_csv(path), path.name, "case_table_" + path.stem)
            report = run["path"] / "report.html"
            if report.is_file():
                if st.toggle("Display the complete result card", value=False, key="show_complete_card"):
                    st.caption("Complete result card, including model commentary and figures. See the structured results for the selected relations.")
                    components.html(report.read_text(encoding="utf-8"), height=950, scrolling=True)
                st.download_button("Download complete result card · HTML", report.read_bytes(), file_name=f"{case_id}_report.html", mime="text/html")
            st.download_button("Download run bundle · ZIP", archive_bytes(run["path"]), file_name=f"{case_id}_{run['path'].name}.zip", mime="application/zip")
            json_panel("Run manifest", run["manifest"])
        else:
            st.info("No saved run bundle is available for this case.")


def prediction_view():
    folder = ROOT / "results" / "sandgrain_prediction"
    frame = read_csv(folder / "predictions.csv")
    if frame.empty:
        return
    st.markdown("#### Sand-grain prediction comparison")
    json_panel("Prediction comparison provenance", read_json(folder / "manifest.json"), key="prediction_manifest")
    # Column names are retained from the exported validation refit.
    filtered = frame.copy()
    for field in ["source", "split"]:
        if field in filtered:
            values = sorted(filtered[field].dropna().astype(str).unique())
            selected = st.multiselect(field.title(), values, default=values, key="prediction_" + field)
            filtered = filtered[filtered[field].astype(str).isin(selected)]
    numeric = filtered.select_dtypes(include="number").columns.tolist()
    reference_candidates = [c for c in numeric if "reference" in c.lower() or c in {"y_true", "ks_Ra", "target"}]
    if reference_candidates:
        reference = reference_candidates[0]
        prediction_columns = [c for c in numeric if "pred" in c.lower() or c in {"original_seven", "learned_groups"}]
        for i, prediction in enumerate(prediction_columns):
            figure = px.scatter(filtered, x=reference, y=prediction, color="split" if "split" in filtered else None,
                                symbol="source" if "source" in filtered else None,
                                title=prediction.replace("_", " ").title())
            if not filtered.empty:
                low, high = filtered[reference].min(), filtered[reference].max()
                figure.add_shape(type="line",x0=low,x1=high,y0=low,y1=high,line={"color":"#243b50","dash":"dash"})
                for factor in [.65, 1.35]:
                    figure.add_shape(type="line",x0=low,x1=high,y0=factor*low,y1=factor*high,line={"color":"#8b9baa","dash":"dot"})
            chart(figure, f"sandgrain_prediction_{i}")
    table(filtered, "sandgrain_predictions.csv", "prediction_download")
    figures(folder, "sandgrain_prediction_figures")
    st.caption("Offline validation refit comparing both representations on the same 83 training and 21 test samples.")


def pressure_baseline_view():
    folder = ROOT / "results" / "pressure_gradient_baseline"
    frame = read_csv(folder / "baseline_points.csv")
    if frame.empty:
        return
    st.markdown("#### Reynolds-number baseline")
    manifest = read_json(folder / "manifest.json")
    st.caption("Direct coordinate evaluation from all 144 supplied observations. Source labels are retained: zeroPG → ZPG (84), advPG → APG (36), favPG → FPG (24). No model is fitted.")
    classes = st.multiselect("Pressure-gradient classes", ["ZPG", "APG", "FPG"], default=["ZPG", "APG", "FPG"], key="pressure_baseline_classes")
    selected = frame[frame.pressure_gradient_class.isin(classes)]
    figure = px.scatter(selected, x="Re_delta", y="U_tau_over_U_inf", color="pressure_gradient_class",
                        symbol="pressure_gradient_class", log_x=True,
                        color_discrete_map={"ZPG":"#245B78", "APG":"#C17837", "FPG":"#209B88"},
                        symbol_map={"ZPG":"circle-open", "APG":"triangle-up-open", "FPG":"square-open"},
                        category_orders={"pressure_gradient_class":["ZPG", "APG", "FPG"]},
                        hover_data=["csv_row", "source_case", "U_inf", "delta", "nu"],
                        labels={"Re_delta":"Reδ = U∞δ/ν", "U_tau_over_U_inf":"uτ/U∞", "pressure_gradient_class":"Pressure-gradient class"},
                        height=500)
    figure.update_traces(marker={"size":8,"line":{"width":1.3}})
    chart(figure, "pressure_baseline_scatter")
    st.caption("This plot examines the conventional Reynolds-number coordinate; the discovered dimensionless relation and its collapse are shown above.")
    table(selected, "pressure_gradient_baseline_points.csv", "pressure_baseline_csv")
    with st.expander("Original baseline figure and vector downloads"):
        figures(folder, "pressure_baseline_original")
    json_panel("Baseline coordinate provenance", manifest, key="pressure_baseline_manifest")
