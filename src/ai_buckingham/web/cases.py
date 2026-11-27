"""Case reports and their numerical views, bound to one selected run."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ..artifacts import run_figures
from ..reports import render_report, report_figures, report_sections, report_styles
from . import diagnostics, rag
from .components import chart, collapse_axis_controls, figures, json_download, masthead, selected_run, table
from .data import ROOT, archive_bytes, case_catalog, dataset_path, first, read_csv, read_json
from .publication import paper_cases


def _stage(section):
    if section:
        st.html(section["html"])


def render():
    masthead("Case results")
    catalog = case_catalog()
    case_id = st.selectbox("Physical system", list(catalog),
                           format_func=lambda key: catalog[key]["title"], key="workflow_case")
    run = selected_run(case_id, key="workflow_run")
    if not run:
        return
    state = run["state"]
    published = diagnostics.is_publication_run(case_id, run)
    sections = {section["id"]: section for section in report_sections(run["path"])}
    st.html(report_styles())

    _stage(sections.get("results"))
    paper_case = paper_cases().get(case_id)
    if published and paper_case and (ROOT / paper_case["image"]).is_file():
        st.image(str(ROOT / paper_case["image"]), width="stretch")
    else:
        for image in report_figures(run["path"]):
            st.image(str(image), width="stretch")

    with st.expander("Interactive data collapse", expanded=False):
        collapse_view(run, case_id)
        if published and case_id == "sandgrain":
            prediction_view()
        if published and case_id == "pressure_gradient":
            pressure_baseline_view()

    for stage_id in ("formulation", "dimensions", "planning"):
        section = sections.get(stage_id)
        if section or stage_id == "dimensions":
            title = section["title"] if section else "Physical consistency and dimensional basis"
            with st.expander(title, expanded=False):
                if stage_id == "dimensions":
                    cards, retrieval = st.tabs(["Dimensions and basis", "Retrieved records"])
                    with cards:
                        _stage(section)
                    with retrieval:
                        rag.render_case(run)
                else:
                    _stage(section)
                if stage_id == "formulation":
                    dataset = Path(run["config"].get("dataset", ""))
                    if not dataset.is_absolute():
                        dataset = ROOT / dataset
                    if not dataset.is_file() and published:
                        dataset = dataset_path(case_id)
                    if dataset.is_file():
                        st.download_button("Download input dataset · CSV", dataset.read_bytes(),
                                           file_name=dataset.name, mime="text/csv", key="case_dataset_download")

    detection = first(state, "regime_detection", default={}) or {}
    with st.expander("Physical-regime identification", expanded=False):
        _stage(sections.get("regimes"))
        views = []
        if diagnostics.gradient_available(detection, case_id, allow_publication_fallback=published):
            views.append("Gradient field")
        if diagnostics.candidate_sources(run, case_id):
            views.append("Candidate partitions")
        if views:
            for tab, title in zip(st.tabs(views), views):
                with tab:
                    if title == "Gradient field":
                        diagnostics.gradient_view(detection, case_id, allow_publication_fallback=published)
                    else:
                        diagnostics.candidate_memberships(run, case_id)

    with st.expander("Dimensionless learning and validation", expanded=False):
        _stage(sections.get("learning"))
        diagnostics.mi_view(state, run, case_id)

    with st.expander("Downloads", expanded=False):
        st.download_button("Download case report · HTML", render_report(run["path"]).encode("utf-8"),
                           file_name=f"{case_id}_report.html", mime="text/html", key="case_report_download")
        st.download_button("Download run files · ZIP", archive_bytes(run["path"]),
                           file_name=f"{case_id}_{run['path'].name}.zip", mime="application/zip", key="case_bundle_download")
        for path in run_figures(run["path"]):
            st.download_button("Download " + path.name, path.read_bytes(), file_name=path.name,
                               key="case_figure_" + path.name)
        json_download("run settings", run["config"], key="case_config")
        json_download("run manifest", run["manifest"], key="case_manifest")


def collapse_view(run, case_id):
    collapse_tables = sorted((run["path"] / "tables").glob("*collapse.csv"))
    if not collapse_tables:
        st.info("Data-collapse coordinates are unavailable for this run.")
        return
    path = st.selectbox("Scope", collapse_tables, format_func=lambda p: p.stem.removesuffix("_collapse"),
                        key=f"collapse_scope_{case_id}_{run['path'].name}")
    frame = read_csv(path)
    pi_columns = [column for column in frame if column.startswith("pi_")]
    if not pi_columns or "response_transformed" not in frame:
        return
    panel = path.stem.removesuffix("_collapse")
    control_key = f"collapse_{case_id}_{run['path'].name}_{panel}"
    x = st.selectbox("Dimensionless coordinate", pi_columns, key=f"{control_key}_coordinate")
    log_x, log_y = collapse_axis_controls(case_id, panel, key=control_key)
    valid = frame
    if log_x:
        valid = valid[valid[x] > 0]
    if log_y:
        valid = valid[valid.response_transformed > 0]
    figure = px.scatter(valid, x=x, y="response_transformed", log_x=log_x, log_y=log_y,
                        hover_data=[column for column in ["csv_row", "response_original", "output_scale"] if column in valid],
                        opacity=.75)
    figure.update_traces(marker={"size":8})
    figure.update_layout(xaxis_type="log" if log_x else "linear", yaxis_type="log" if log_y else "linear")
    chart(figure, f"case_{control_key}")
    table(frame, path.name, "case_collapse_download")


def prediction_view():
    folder = ROOT / "results" / "sandgrain_prediction"
    frame = read_csv(folder / "predictions.csv")
    if frame.empty:
        return
    st.markdown("#### Sand-grain prediction comparison")
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
    figures(folder, "sandgrain_prediction_figures", display_images=False)
    json_download("prediction comparison settings", read_json(folder / "manifest.json"), key="prediction_manifest")


def pressure_baseline_view():
    folder = ROOT / "results" / "pressure_gradient_baseline"
    frame = read_csv(folder / "baseline_points.csv")
    if frame.empty:
        return
    st.markdown("#### Reynolds-number baseline")
    manifest = read_json(folder / "manifest.json")
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
    table(selected, "pressure_gradient_baseline_points.csv", "pressure_baseline_csv")
    figures(folder, "pressure_baseline_original", display_images=False)
    json_download("baseline coordinate settings", manifest, key="pressure_baseline_manifest")
