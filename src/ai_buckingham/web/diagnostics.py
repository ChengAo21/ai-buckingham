"""Numerical decisions and the actual generation trace of each CMA-ES search."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .components import chart, json_panel, masthead, selected_run, table
from .data import ROOT, case_catalog, dataset_path, first, json_lines, read_csv, read_json, reproduction


def mi_contour(pivot: pd.DataFrame) -> go.Contour:
    """Render the saved MI grid with 80 color bands; retain every input value."""
    finite = pivot.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    contours = {"coloring":"fill", "showlines":False}
    settings = {}
    if finite.size and finite.max() > finite.min():
        low, high = float(finite.min()), float(finite.max())
        contours.update(start=low, end=high, size=(high-low)/80)
        settings["autocontour"] = False
    return go.Contour(x=pivot.columns, y=pivot.index, z=pivot.values,
                      colorscale="Blues", ncontours=80, contours=contours,
                      colorbar={"title":{"text":"MI", "side":"top"}, "nticks":8,
                                "tickformat":".3g", "thickness":16, "len":.85},
                      name="Conditional MI", **settings)


def gradient_view(detection: dict, case_id: str):
    coordinates = detection.get("feature_coordinates", [])
    gradients = detection.get("surrogate_gradients", [])
    if coordinates and gradients:
        x = np.asarray(coordinates, dtype=float)
        g = np.asarray(gradients, dtype=float)
        if x.ndim == 2 and x.shape[1] >= 2 and x.shape == g.shape:
            options = list(range(x.shape[1]))
            a, b, c = st.columns(3)
            ix = a.selectbox("Horizontal log-Pi coordinate", options, format_func=lambda i: f"z{i+1}", key="gradient_x")
            iy = b.selectbox("Vertical log-Pi coordinate", options, index=1, format_func=lambda i: f"z{i+1}", key="gradient_y")
            stride = c.slider("Arrow sampling interval", 1, max(1, len(x)//10), max(1, len(x)//70), key="gradient_stride")
            labels = detection.get("labels", [])
            frame = pd.DataFrame({"x": x[:, ix], "y": x[:, iy], "Row": detection.get("valid_indices", list(range(len(x))))})
            frame["Regime"] = [str(v) for v in labels] if len(labels) == len(x) else "Not partitioned"
            fig = px.scatter(frame, x="x", y="y", color="Regime", hover_data=["Row"], opacity=.6)
            norms = np.linalg.norm(g, axis=1)
            unit = np.divide(g, norms[:, None], out=np.zeros_like(g), where=norms[:, None] > 0)
            scales = np.maximum(np.ptp(x, axis=0), 1e-12) * .035
            for i in range(0, len(x), stride):
                fig.add_annotation(x=x[i,ix] + unit[i,ix]*scales[ix], y=x[i,iy] + unit[i,iy]*scales[iy],
                                   ax=x[i,ix], ay=x[i,iy], xref="x", yref="y", axref="x", ayref="y", text="",
                                   showarrow=True, arrowhead=2, arrowsize=.8, arrowwidth=1, arrowcolor="#243b50")
            fig.update_layout(xaxis_title=f"z{ix+1} · orthonormal log-Pi", yaxis_title=f"z{iy+1} · orthonormal log-Pi", height=570)
            chart(fig, "gradient_field")
            st.caption("Arrows show projections of normalized, recorded posterior-mean gradients. The scatter uses the recorded coordinate system and valid sample indices.")
            table(pd.concat([frame, pd.DataFrame(g, columns=[f"gradient_z{i+1}" for i in range(g.shape[1])])], axis=1), "gradient_field.csv")
            return
    if case_id in {"cess_channel", "cess"}:
        frame = read_csv(reproduction("regime_comparison") / "data" / "recorded_gradient_membership.csv")
        if not frame.empty:
            chart(px.scatter(frame, x="Re_tau", y="eta", color="original_regime", log_x=True, log_y=True,
                             hover_data=["row_id", "y_plus"], height=570), "recorded_cess_partition")
            st.caption("Historical pointwise partition. The supplied comparison archive does not contain its underlying gradient vectors; arrows cannot be reconstructed from memberships alone.")
            table(frame, "recorded_gradient_membership.csv")
            return
    st.info("Gradient coordinates and vectors were not recorded for this source. Run an analysis to export the full numerical state.")


def candidate_memberships(run, case_id):
    if not run:
        return
    direct = sorted((run["path"] / "evidence" / "regime_candidates").glob("*/manifest.json"))
    supplementary = ROOT / "results" / "regime_candidates" / case_id / run["path"].name / "manifest.json"
    sources = direct + ([supplementary] if supplementary.is_file() else [])
    if not sources:
        st.info("Pointwise labels for unselected K candidates were not recorded in this run. Summary statistics alone cannot reconstruct those assignments.")
        return
    source = st.selectbox("Pointwise partition evidence", sources,
                          format_func=lambda p: read_json(p).get("mode", "recorded candidate partitions") + " · " + p.parent.name,
                          key=f"candidate_membership_source_{case_id}_{run['path'].name}")
    manifest = read_json(source)
    mode = manifest.get("mode", "actual execution record")
    st.caption("Source mode: " + mode + ". " + str(manifest.get("method", "Recorded from the actual numerical execution.")))
    available = sorted(source.parent.glob("k_*.npz"))
    if not available:
        st.info("No pointwise membership arrays are available in this evidence source.")
        return
    chosen = st.selectbox("Candidate partition", available, format_func=lambda p: "K = " + p.stem.split("_")[-1], key=f"candidate_membership_k_{case_id}_{run['path'].name}_{source.parent.name}")
    with np.load(source.parent / "coordinates.npz", allow_pickle=False) as arrays:
        coordinates = arrays["feature_coordinates"].copy()
        valid_indices = arrays["valid_indices"].copy() if "valid_indices" in arrays else np.arange(len(coordinates))
    with np.load(chosen, allow_pickle=False) as arrays:
        labels = arrays["labels"].copy()
        confidence = arrays["assignment_confidence"].copy() if "assignment_confidence" in arrays else None
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or len(labels) != len(coordinates):
        st.info("The selected pointwise arrays cannot be plotted together; inspect their source files.")
        return
    a, b = st.columns(2)
    ix = a.selectbox("Partition coordinate x", range(coordinates.shape[1]), format_func=lambda i:f"z{i+1}", key="candidate_coordinate_x")
    iy = b.selectbox("Partition coordinate y", range(coordinates.shape[1]), index=1, format_func=lambda i:f"z{i+1}", key="candidate_coordinate_y")
    frame = pd.DataFrame({"z_x":coordinates[:,ix],"z_y":coordinates[:,iy],"Regime":[str(int(v)) for v in labels],"csv_row":valid_indices})
    if confidence is not None:
        frame["Assignment confidence"] = confidence
    chart(px.scatter(frame,x="z_x",y="z_y",color="Regime",hover_data=[c for c in ["csv_row","Assignment confidence"] if c in frame],
                     labels={"z_x":f"z{ix+1} · orthonormal log-Pi","z_y":f"z{iy+1} · orthonormal log-Pi"},height=550),"candidate_partition_scatter")
    physical = read_csv(dataset_path(case_id))
    numeric = physical.select_dtypes(include="number").columns.tolist()
    if len(numeric)>=2 and len(valid_indices) and int(max(valid_indices))<len(physical):
        with st.expander("Inspect the same assignments in physical coordinates"):
            pa,pb = st.columns(2)
            px_col = pa.selectbox("Physical coordinate x",numeric,key="candidate_physical_x")
            py_col = pb.selectbox("Physical coordinate y",numeric,index=1,key="candidate_physical_y")
            physical_view = physical.iloc[valid_indices.astype(int)].copy()
            physical_view["Regime"] = frame.Regime.to_numpy()
            physical_view["csv_row"] = valid_indices
            chart(px.scatter(physical_view,x=px_col,y=py_col,color="Regime",hover_data=["csv_row"],height=480),"candidate_physical_plot")
    table(frame,"selected_candidate_memberships.csv","candidate_memberships_csv")
    json_panel("Pointwise membership provenance",manifest,key="candidate_manifest")
    json_panel("Selected candidate details",read_json(chosen.with_suffix(".json")),key="candidate_detail_json")


def partition_view(detection: dict, run=None, case_id=None):
    candidate_memberships(run, case_id)
    candidates = detection.get("candidate_summary", [])
    if not candidates:
        st.info("No candidate-partition table is recorded for this source.")
        return
    rows = [{k:v for k,v in candidate.items() if not isinstance(v,(dict,list))} for candidate in candidates]
    table(pd.DataFrame(rows), "candidate_partitions.csv")
    for candidate in candidates:
        k = candidate.get("k", "?")
        with st.expander(f"K = {k} · cluster-level evidence"):
            clusters = candidate.get("clusters", [])
            if clusters:
                rows = [{key: value for key,value in cluster.items() if not isinstance(value,(dict,list))} for cluster in clusters]
                table(pd.DataFrame(rows), f"partition_k{k}_clusters.csv", f"clusters_{k}")
                if all("fraction" in row for row in rows):
                    fig = px.bar(pd.DataFrame(rows).assign(Regime=[str(i+1) for i in range(len(rows))]), x="Regime", y="fraction")
                    chart(fig, f"partition_fraction_{k}")
            json_panel(f"Full K={k} candidate", candidate, key=f"full_candidate_{k}")
    json_panel("Regime selection criteria", detection.get("acceptance_criteria", {}))
    json_panel("Final regime decision", {k: detection.get(k) for k in ["status","best_k","selection_basis","decision_evidence","k2_direction_action","single_regime_baseline"]})
    st.caption("Global leading-eigenvalue fraction is diagnostic only. Automatic multiple-regime selection requires an admissible candidate and K=2 directional separation. A prescribed multi-regime policy follows its own route.")


def landscape_view(case_id):
    folder = ROOT / "results" / "mi_landscapes" / case_id
    surface_path = folder / "response_surface.csv"
    if not surface_path.is_file():
        return
    st.markdown("#### MI landscape and its recorded search")
    metadata = read_json(folder / "run_metadata.json")
    result = read_json(folder / "optimization_result.json")
    st.caption("Source mode: " + str(metadata.get("mode", "recorded calculation")) + ". " + str(metadata.get("method", "Landscape grid and generation history from the supplied numerical reproduction.")))
    frame = read_csv(surface_path)
    histories = sorted(folder.glob("*generation_history.csv"))
    history = read_csv(histories[0]) if histories else pd.DataFrame()
    if not {"coefficient_x","coefficient_y","mutual_information"}.issubset(frame.columns):
        json_panel("Landscape metadata",metadata,key="landscape_metadata")
        return
    pivot = frame.pivot(index="coefficient_y",columns="coefficient_x",values="mutual_information")
    figure = go.Figure(mi_contour(pivot))
    selected_record = None
    ix, iy = 0, 1
    coefficients = (result.get("plot_info") or {}).get("selected_coefficients", [])
    if len(coefficients)==2:
        ix,iy = [int(name.split("_")[-1])-1 for name in coefficients]
    selected_generation = None
    if not history.empty and "generation" in history:
        generations = history.generation.tolist()
        selected_generation = st.select_slider("Landscape history generation",generations,value=generations[-1],key="landscape_generation") if len(generations)>1 else generations[0]
        subset = history[history.generation<=selected_generation]
        selected_record = subset.iloc[-1]
        xcol,ycol = f"mean_c_{ix}",f"mean_c_{iy}"
        if xcol in subset and ycol in subset:
            figure.add_trace(go.Scatter(x=subset[xcol],y=subset[ycol],mode="lines+markers",name="Recorded mean projection",
                                       line={"color":"#C17837","width":2},marker={"size":6},customdata=subset.generation,
                                       hovertemplate="Generation %{customdata}<br>x=%{x}<br>y=%{y}<extra></extra>"))
            figure.add_trace(go.Scatter(x=[selected_record[xcol]],y=[selected_record[ycol]],mode="markers",name="Selected generation",marker={"size":15,"symbol":"diamond","color":"#D25264"}))
        stats = st.columns(3)
        stats[0].metric("Recorded generation",int(selected_generation))
        if "global_best_mi" in selected_record:
            stats[1].metric("Best recorded MI",f"{selected_record['global_best_mi']:.6g}")
        if "sigma" in selected_record:
            stats[2].metric("Step size σ",f"{selected_record['sigma']:.6g}")
    figure.update_layout(xaxis_title=f"c{ix+1}",yaxis_title=f"c{iy+1}",height=590)
    chart(figure,"landscape_conditional_plot")
    st.caption("The contour is an exported conditional MI grid. The overlaid mean path is a projection of the actual search; unplotted coefficients can change along the path and need not equal the fixed slice values. Finite negative MI estimates are retained.")
    view = st.radio("Landscape rendering",["Contour","3D surface"],horizontal=True,key="landscape_rendering")
    if view=="3D surface":
        surface=go.Figure(go.Surface(x=pivot.columns,y=pivot.index,z=pivot.values,colorscale="Blues",
                                    colorbar={"title":{"text":"MI","side":"top"},"nticks":8,
                                              "tickformat":".3g","thickness":16,"len":.85}))
        surface.update_layout(scene={"xaxis_title":f"c{ix+1}","yaxis_title":f"c{iy+1}","zaxis_title":"MI"},height=580)
        chart(surface,"landscape_3d_surface")
    if not history.empty:
        if "global_best_mi" in history:
            chart(px.line(history[history.generation<=selected_generation],x="generation",y="global_best_mi",markers=True,labels={"global_best_mi":"Best recorded MI"}),"landscape_history_mi")
        table(history,"landscape_generation_history.csv","landscape_history_download")
    st.download_button("Download numeric landscape grid · CSV",surface_path.read_bytes(),file_name=f"{case_id}_mi_grid.csv",mime="text/csv",key="landscape_grid_download")
    json_panel("Landscape metadata",metadata,key="landscape_metadata")
    if result:
        json_panel("Landscape search result",result,key="landscape_result")
    from .components import figures
    with st.expander("Original landscape figure and vector downloads"):
        figures(folder,"landscape_original")


def mi_view(state: dict, run, case_id):
    landscape_view(case_id)
    locals_ = first(state, "local_workflows", default=[])
    if not locals_:
        st.info("This source has no recorded MI candidate path.")
    for i, local in enumerate(locals_ if isinstance(locals_, list) else []):
        st.markdown("#### " + str(local.get("scope_id", "Global")).replace("_", " ").title())
        records = []
        for candidate in local.get("dimension_path", []):
            records.append({"Stage": f"k={candidate.get('pi_count', '?')}",
                            "MI": candidate.get("mutual_information"),
                            "Information score": candidate.get("normalized_information_score")})
        for name, field in [("Sparse", "sparsification"), ("Rationalized", "rationalization")]:
            result = local.get(field) or {}
            solution = result.get("final", result.get("candidate", {})) or {}
            if solution:
                records.append({"Stage": name, "MI": solution.get("mutual_information"),
                                "Information score": solution.get("normalized_information_score")})
        frame = pd.DataFrame(records)
        if not frame.empty:
            a, b = st.columns(2)
            with a:
                chart(px.line(frame, x="Stage", y="MI", markers=True), f"mi_path_{i}")
            with b:
                chart(px.line(frame, x="Stage", y="Information score", markers=True), f"epsilon_path_{i}")
            table(frame, f"mi_path_{i}.csv")
        json_panel(f"Similarity screening · {i+1}", local.get("similarity_screening", {}), key=f"screen_{i}")
        json_panel(f"Count and simplification decisions · {i+1}", {key:local.get(key) for key in ["count_decisions","lock_reason","sparsification","rationalization"]}, key=f"simplify_{i}")
    if run:
        surfaces = list(run["path"].rglob("response_surface.csv"))
        for i, path in enumerate(surfaces):
            frame = read_csv(path)
            if {"coefficient_x","coefficient_y","mutual_information"}.issubset(frame.columns):
                pivot = frame.pivot(index="coefficient_y", columns="coefficient_x", values="mutual_information")
                chart(go.Figure(mi_contour(pivot)), f"mi_surface_{i}")
                st.caption("Conditional two-coordinate MI slice. Other coefficients are fixed to the exported slice definition.")
                table(frame, f"mi_surface_{i}.csv")
    st.caption("Restart and subsampling uncertainty measure different sources of variation. Full-precision values remain available in the saved state and tables.")


def trace_view(path):
    trace_file = path / "evidence" / "optimization_trace.jsonl" if path else None
    records = [r for r in json_lines(trace_file) if isinstance(r.get("generation"), (int, float))] if trace_file else []
    if not records:
        st.info("No per-generation trace is present in this run. Final coefficients and a figure do not establish a historical optimization path.")
        st.caption("New CMA-ES runs export actual post-tell generation records to evidence/optimization_trace.jsonl.")
        return
    # A generation reset marks a different optimizer instance even when its
    # scope, seed and candidate count coincide (screening versus final search).
    instances, current, previous_signature, previous_generation = [], [], None, None
    for record in records:
        signature = tuple(str(record.get(k,"")) for k in ["search_id","optimization_id","scope_id","pi_count","optimize_output","seed"])
        generation = record.get("generation", 0)
        if current and (signature != previous_signature or generation <= previous_generation):
            instances.append(current)
            current = []
        current.append(record)
        previous_signature, previous_generation = signature, generation
    if current:
        instances.append(current)
    def label(index):
        r = instances[index][0]
        return f"Search {index+1} · {r.get('scope_id','scope')} · k={r.get('pi_count','?')} · seed={r.get('seed','?')} · output scaling={r.get('optimize_output','?')}"
    index = st.selectbox("CMA-ES search", range(len(instances)), format_func=label, key="trace_search")
    selected = instances[index]
    frame = pd.DataFrame(selected)
    generations = sorted(frame["generation"].unique().tolist())
    generation = st.select_slider("Recorded generation", generations, value=generations[-1], key="trace_generation") if len(generations)>1 else generations[0]
    subset = frame[frame["generation"]<=generation]
    row = subset.iloc[-1]
    metrics = st.columns(4)
    for column, title, field in zip(metrics, ["Generation","Best penalized objective","Step size σ","Function evaluations"], ["generation","best_objective","sigma","evaluations"]):
        value = row.get(field)
        column.metric(title, f"{value:.6g}" if isinstance(value,(int,float,np.number)) else str(value))
    fields = [c for c in ["best_objective","generation_best_objective","mean_objective"] if c in subset]
    if fields:
        chart(px.line(subset.melt(id_vars="generation", value_vars=fields, var_name="Objective", value_name="Value"),
                      x="generation", y="Value", color="Objective"), "trace_objective")
    st.caption("These values are the actual penalized optimization objective; they are not mutual information. The slider reveals recorded history and does not rerun the search.")
    means = [r.get("mean_coefficients",[]) for r in selected]
    if means and isinstance(means[0],list) and len(means[0])>=2:
        size = min(len(v) for v in means)
        a, b = st.columns(2)
        ix = a.selectbox("Projected coefficient x", range(size), format_func=lambda i:f"θ{i+1}", key="trace_x")
        iy = b.selectbox("Projected coefficient y", range(size), index=1, format_func=lambda i:f"θ{i+1}", key="trace_y")
        points = pd.DataFrame({"x":[v[ix] for v in means],"y":[v[iy] for v in means],"generation":frame["generation"]})
        points = points[points["generation"]<=generation]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points.x,y=points.y, mode="lines+markers",marker={"color":points.generation,"colorscale":"Blues","size":7,
                                "colorbar":{"title":{"text":"Generation","side":"top"},"thickness":16,"len":.85,"nticks":7}},
                                customdata=points.generation,hovertemplate="θx=%{x}<br>θy=%{y}<br>Generation %{customdata}<extra></extra>",name="Recorded mean"))
        fig.add_trace(go.Scatter(x=[points.x.iloc[-1]],y=[points.y.iloc[-1]],mode="markers",marker={"size":15,"symbol":"diamond","color":"#C17837"},name="Selected generation"))
        fig.update_layout(xaxis_title=f"θ{ix+1}",yaxis_title=f"θ{iy+1}",height=530)
        chart(fig,"trace_projection")
        st.caption("Projection of the recorded search mean onto two free coefficients. Other coordinates also evolve; this is not a conditional objective surface.")
    json_panel("Selected generation record", selected[len(subset)-1], key="trace_row")
    table(frame.drop(columns=[c for c in frame if frame[c].map(lambda v:isinstance(v,(list,dict))).any()]), "optimization_trace.csv")
    st.download_button("Download complete generation trace · JSONL", trace_file.read_bytes(), file_name="optimization_trace.jsonl", mime="application/x-ndjson")


def render():
    masthead("Numerical diagnostics", "Inspect the evidence behind regime selection, similarity screening, dimensionality and optimization.", "Numerical inference")
    catalog = case_catalog()
    case_id = st.selectbox("Physical system", list(catalog), format_func=lambda k:catalog[k]["title"], key="diagnostic_case")
    run = selected_run(case_id, key="diagnostic_run")
    state = run["state"] if run else {}
    detection = first(state, "regime_detection", default={}) or {}
    tabs = st.tabs(["Gradient field", "Candidate partitions", "MI & simplification", "CMA-ES generations"])
    with tabs[0]: gradient_view(detection, case_id)
    with tabs[1]: partition_view(detection, run, case_id)
    with tabs[2]: mi_view(state, run, case_id)
    with tabs[3]: trace_view(run["path"] if run else None)
