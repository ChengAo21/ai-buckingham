"""Physical-regime views and the paper's dimensionless-learning results."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .components import chart, json_download
from .data import ROOT, dataset_path, first, read_csv, read_json, reproduction


def is_publication_run(case_id, run):
    if not run:
        return False
    index = read_json(ROOT / "results" / "index.json")
    selected = next((case for case in index.get("cases", []) if case["case_id"] == case_id), {})
    directory = ROOT / "results" / "cases" / case_id / str(selected.get("selected_run", ""))
    return run["path"].resolve() == directory.resolve()


def gradient_available(detection, case_id, *, allow_publication_fallback=True):
    coordinates = np.asarray(detection.get("feature_coordinates", []))
    gradients = np.asarray(detection.get("surrogate_gradients", []))
    if coordinates.ndim == 2 and coordinates.shape[1] >= 2 and coordinates.shape == gradients.shape:
        return bool(len(coordinates))
    return (allow_publication_fallback and case_id in {"cess_channel", "cess"}
            and not read_csv(reproduction("regime_comparison") / "data" / "recorded_gradient_membership.csv").empty)


def candidate_sources(run, case_id):
    if not run:
        return []
    direct = sorted((run["path"] / "evidence" / "regime_candidates").glob("*/manifest.json"))
    supplementary = ROOT / "results" / "regime_candidates" / case_id / run["path"].name / "manifest.json"
    return direct + ([supplementary] if is_publication_run(case_id, run) and supplementary.is_file() else [])


def gradient_view(detection: dict, case_id: str, *, allow_publication_fallback=True):
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
            st.caption("Normalized gradient directions in the displayed coordinates.")
            output = pd.concat([frame, pd.DataFrame(g, columns=[f"gradient_z{i+1}" for i in range(g.shape[1])])], axis=1)
            st.download_button("Download gradient field · CSV", output.to_csv(index=False).encode("utf-8"),
                               file_name="gradient_field.csv", mime="text/csv", key="gradient_field_csv")
            return
    if allow_publication_fallback and case_id in {"cess_channel", "cess"}:
        frame = read_csv(reproduction("regime_comparison") / "data" / "recorded_gradient_membership.csv")
        if not frame.empty:
            chart(px.scatter(frame, x="Re_tau", y="eta", color="original_regime", log_x=True, log_y=True,
                             hover_data=["row_id", "y_plus"], height=570), "recorded_cess_partition")
            st.caption("Gradient vectors are unavailable for this archived partition.")
            st.download_button("Download regime membership · CSV", frame.to_csv(index=False).encode("utf-8"),
                               file_name="recorded_gradient_membership.csv", mime="text/csv",
                               key="recorded_cess_membership_csv")
            return
    st.info("Gradient coordinates and vectors were not recorded for this source. Run an analysis to export the full numerical state.")


def candidate_memberships(run, case_id):
    if not run:
        return
    sources = candidate_sources(run, case_id)
    if not sources:
        st.info("Pointwise labels for unselected K candidates were not recorded in this run. Summary statistics alone cannot reconstruct those assignments.")
        return
    source = sources[0] if len(sources) == 1 else st.selectbox(
        "Pointwise partition evidence", sources,
        format_func=lambda p: read_json(p).get("mode", "recorded candidate partitions") + " · " + p.parent.name,
        key=f"candidate_membership_source_{case_id}_{run['path'].name}")
    manifest = read_json(source)
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
    dataset = Path(run["config"].get("dataset", ""))
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    if not dataset.is_file() and is_publication_run(case_id, run):
        dataset = dataset_path(case_id)
    physical = read_csv(dataset)
    numeric = physical.select_dtypes(include="number").columns.tolist()
    if len(numeric)>=2 and len(valid_indices) and int(max(valid_indices))<len(physical):
        with st.container():
            st.markdown("##### Physical coordinates")
            pa,pb = st.columns(2)
            px_col = pa.selectbox("Physical coordinate x",numeric,key="candidate_physical_x")
            py_col = pb.selectbox("Physical coordinate y",numeric,index=1,key="candidate_physical_y")
            physical_view = physical.iloc[valid_indices.astype(int)].copy()
            physical_view["Regime"] = frame.Regime.to_numpy()
            physical_view["csv_row"] = valid_indices
            chart(px.scatter(physical_view,x=px_col,y=py_col,color="Regime",hover_data=["csv_row"],height=480),"candidate_physical_plot")
    st.download_button("Download candidate membership · CSV", frame.to_csv(index=False).encode("utf-8"),
                       file_name="selected_candidate_memberships.csv", mime="text/csv",
                       key="candidate_memberships_csv")
    json_download("partition settings",manifest,key="candidate_manifest")
    json_download("selected candidate details",read_json(chosen.with_suffix(".json")),key="candidate_detail_json")


_LANDSCAPE_VIEWS = {
    "mhd": (50, 50),
    "rayleigh_benard": (50, 50),
    "pressure_gradient": (60, -70),
    "laser_keyhole": (50, 50),
    "blasius": (65, -20),
}


def landscape_figure(case_id, frame, history, result, trajectory=None):
    """Place the matching optimizer path on its saved conditional MI grid."""
    from scipy.interpolate import RegularGridInterpolator

    pivot = frame.pivot(index="coefficient_y", columns="coefficient_x", values="mutual_information")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]
    figure = go.Figure(go.Surface(
        x=x, y=y, z=z, colorscale="Viridis", opacity=.9,
        colorbar={"title": {"text": "MI", "side": "top"}, "thickness": 16,
                  "len": .72, "x": .99, "tickformat": ".3g", "nticks": 6},
        lighting={"ambient": .8, "diffuse": .5, "roughness": .8, "specular": .1},
        lightposition={"x": 100, "y": 200, "z": 300},
        hovertemplate="cₓ=%{x:.3f}<br>cᵧ=%{y:.3f}<br>MI=%{z:.4g}<extra></extra>",
        name="MI",
    ))
    coefficients = (result.get("plot_info") or {}).get("selected_coefficients", ["c_1", "c_2"])
    ix, iy = [int(name.split("_")[-1]) - 1 for name in coefficients]
    bounded = [f"bounded_mean_c_{axis}" for axis in (ix, iy)]
    columns = bounded if all(column in history for column in bounded) else [f"mean_c_{ix}", f"mean_c_{iy}"]
    positions = None
    if trajectory is not None and not trajectory.empty:
        positions = trajectory[["coefficient_x", "coefficient_y"]].to_numpy(dtype=float)
        heights = trajectory["mutual_information"].to_numpy(dtype=float)
        generations = trajectory["generation"].to_numpy()
        if len(positions) != len(history) or not np.array_equal(generations, history["generation"]):
            raise ValueError("The landscape trajectory does not match its optimization history")
        if not np.allclose(positions, history[columns].to_numpy(dtype=float), atol=1e-12, rtol=0):
            raise ValueError("The landscape trajectory uses different coefficient coordinates")
    elif not history.empty and all(column in history for column in columns):
        positions = history[columns].to_numpy(dtype=float)
        interpolator = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=np.nan)
        heights = interpolator(positions[:, [1, 0]])
        generations = history["generation"].to_numpy() if "generation" in history else np.arange(1, len(history) + 1)
    if positions is not None:
        offset = np.ptp(finite) * .015 if finite.size else 0.0
        figure.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=heights + offset,
            mode="lines", name="CMA-ES trajectory", connectgaps=False,
            line={"color": "#E63946", "width": 5},
            customdata=np.column_stack((generations, heights)),
            hovertemplate="Generation %{customdata[0]}<br>MI=%{customdata[1]:.4g}<extra></extra>",
        ))
        if np.isfinite(heights[0]):
            figure.add_trace(go.Scatter3d(
                x=[positions[0, 0]], y=[positions[0, 1]], z=[heights[0] + offset],
                mode="markers", name="Start",
                marker={"color": "white", "size": 6, "line": {"color": "#172431", "width": 2}},
                hovertemplate="Start<extra></extra>",
            ))
        if np.isfinite(heights[-1]):
            figure.add_trace(go.Scatter3d(
                x=[positions[-1, 0]], y=[positions[-1, 1]], z=[heights[-1] + offset],
                mode="text", text=["★"], textposition="middle center", name="End", showlegend=False,
                textfont={"family": "Times New Roman", "size": 28, "color": "#E6B800"},
                hovertemplate="End<extra></extra>",
            ))
            figure.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers", name="★ End",
                marker={"size": 0}, showlegend=True, hoverinfo="skip",
            ))
    bounds = np.column_stack((x[[0, -1]], y[[0, -1]]))
    if positions is not None:
        bounds = np.vstack((bounds, positions))
    minimum, maximum = np.nanmin(bounds, axis=0), np.nanmax(bounds, axis=0)
    padding = (maximum - minimum) * .025
    elevation, azimuth = np.radians(_LANDSCAPE_VIEWS.get(case_id, (50, 50)))
    figure.data[0].hovertemplate = (
        f"c{ix + 1}=%{{x:.3f}}<br>c{iy + 1}=%{{y:.3f}}<br>MI=%{{z:.4g}}<extra></extra>")
    figure.update_layout(
        template="plotly_white",
        font={"family": "Times New Roman, Times, serif", "size": 15, "color": "#20384b"},
        height=530, margin={"l": 0, "r": 35, "t": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis": {"title": {"text": f"c<sub>{ix + 1}</sub>"},
                      "range": [minimum[0] - padding[0], maximum[0] + padding[0]], "nticks": 5},
            "yaxis": {"title": {"text": f"c<sub>{iy + 1}</sub>"},
                      "range": [minimum[1] - padding[1], maximum[1] + padding[1]], "nticks": 5},
            "zaxis": {"title": {"text": "MI"}, "nticks": 5},
            "aspectmode": "manual", "aspectratio": {"x": 1.15, "y": 1.15, "z": .75},
            "camera": {"eye": {"x": float(2 * np.cos(elevation) * np.cos(azimuth)),
                               "y": float(2 * np.cos(elevation) * np.sin(azimuth)),
                               "z": float(2 * np.sin(elevation))}},
        },
        legend={"orientation": "h", "x": .05, "xanchor": "left", "y": 0, "yanchor": "top",
                "font": {"size": 13}},
    )
    return figure


def landscape_view(case_id):
    folder = ROOT / "results" / "mi_landscapes" / case_id
    surface_path = folder / "response_surface.csv"
    if not surface_path.is_file():
        return
    frame = read_csv(surface_path)
    if not {"coefficient_x", "coefficient_y", "mutual_information"}.issubset(frame.columns):
        return
    result = read_json(folder / "optimization_result.json")
    histories = sorted(folder.glob("*generation_history.csv"))
    history = read_csv(histories[0]) if histories else pd.DataFrame()
    trajectory = read_csv(folder / "trajectory_surface.csv")
    st.markdown("#### MI optimization landscape")
    st.plotly_chart(landscape_figure(case_id, frame, history, result, trajectory),
                    width="stretch", key="landscape_3d_surface",
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}})
    st.download_button("Download MI grid · CSV", surface_path.read_bytes(),
                       file_name=f"{case_id}_mi_grid.csv", mime="text/csv", key="landscape_grid_download")


def stage_comparison(local):
    """Read candidate and refinement values in their stage order."""
    from ..plotting import _candidate_uncertainty

    def stage_record(label, candidate):
        record = {"Stage": label, "MI": candidate.get("mutual_information"),
                  "ε": candidate.get("normalized_information_score")}
        subsampling = candidate.get("mi_score_subsampling_uq") or {}
        available = {}
        for metric, restart_field, subsample_field in (
                ("MI", "restart_mutual_information_std", "subsample_mi_std"),
                ("ε", "restart_information_score_std", "subsample_score_std")):
            for component, source, field in (("restart", candidate, restart_field),
                                             ("subsampling", subsampling, subsample_field)):
                value = source.get(field)
                provided = isinstance(value, (int, float)) and np.isfinite(value) and value >= 0
                record[f"{metric} {component} SD"] = value if provided else np.nan
                if provided:
                    if component == "restart":
                        available[field] = value
                    else:
                        available.setdefault("mi_score_subsampling_uq", {})[field] = value
        uncertainty = _candidate_uncertainty(available)
        for metric, combined_field in (("MI", "combined_mi_std"), ("ε", "combined_score_std")):
            provided = any(pd.notna(record[f"{metric} {component} SD"])
                           for component in ("restart", "subsampling"))
            record[f"{metric} SD"] = uncertainty[combined_field] if provided else np.nan
        return record

    candidates = {candidate["pi_count"]: candidate for candidate in local.get("dimension_path", [])
                  if candidate.get("pi_count") is not None}
    records = [stage_record(f"k={count}", candidate) for count, candidate in sorted(candidates.items())]
    for name, field in (("Sparse", "sparsification"), ("Rationalized", "rationalization")):
        result = local.get(field) or {}
        solution = result.get("final") or result.get("candidate") or {}
        if solution:
            records.append(stage_record(name, solution))
    return pd.DataFrame(records, columns=["Stage", "MI", "ε", "MI SD", "ε SD", "MI restart SD",
                                         "MI subsampling SD", "ε restart SD", "ε subsampling SD"])


def mi_view(state: dict, run, case_id):
    if is_publication_run(case_id, run):
        landscape_view(case_id)
    locals_ = first(state, "local_workflows", default=[])
    uncertainty_note_displayed = False
    for i, local in enumerate(locals_ if isinstance(locals_, list) else []):
        st.markdown("#### " + str(local.get("scope_id", "Global")).replace("_", " ").title())
        frame = stage_comparison(local)
        if frame.empty:
            continue
        if not uncertainty_note_displayed and frame[["MI SD", "ε SD"]].notna().any().any():
            st.caption("Error bars: ±1 SD.")
            uncertainty_note_displayed = True
        columns = st.columns(2)
        for column, metric, color in zip(columns, ("MI", "ε"), ("#245B78", "#209B88")):
            with column:
                figure = px.bar(frame, x="Stage", y=metric, text=metric, error_y=f"{metric} SD",
                                 color_discrete_sequence=[color], height=310,
                                 category_orders={"Stage": frame.Stage.tolist()})
                hover = []
                for _, row in frame.iterrows():
                    details = []
                    if pd.notna(row[f"{metric} SD"]):
                        details.append(f"SD={row[f'{metric} SD']:.3g}")
                    for component in ("restart", "subsampling"):
                        value = row[f"{metric} {component} SD"]
                        if pd.notna(value):
                            details.append(f"{component.title()} SD={value:.3g}")
                    hover.append(["<br>" + "<br>".join(details) if details else ""])
                figure.update_traces(texttemplate="%{y:.3g}", textposition="none", cliponaxis=False,
                                     error_y={"color": "#20384b", "thickness": 1.2, "width": 3},
                                     customdata=hover,
                                     hovertemplate=f"%{{x}}<br>{metric}=%{{y:.4g}}%{{customdata[0]}}<extra></extra>")
                values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
                label_errors = frame[f"{metric} SD"].fillna(0).to_numpy(dtype=float)
                valid = np.isfinite(values)
                for stage, value, error in zip(frame.Stage[valid], values[valid], label_errors[valid]):
                    figure.add_annotation(
                        x=stage, y=value + error, text=f"{value:.3g}", showarrow=False,
                        xanchor="center", yanchor="bottom", yshift=8,
                        font={"family": "Times New Roman, Times, serif", "size": 13, "color": "#20384b"})
                if valid.any():
                    lower = min(0, np.min(values[valid] - label_errors[valid]))
                    upper = max(0, np.max(values[valid] + label_errors[valid]))
                    span = upper - lower or 1
                    figure.update_yaxes(range=[lower - (.04 * span if lower < 0 else 0), upper + .2 * span])
                figure.update_layout(xaxis_title=None, bargap=.4, showlegend=False)
                chart(figure, f"{'mi' if metric == 'MI' else 'epsilon'}_path_{i}")
        st.download_button("Download stage results · CSV", frame.to_csv(index=False).encode("utf-8"),
                           file_name=f"{local.get('scope_id', 'global')}_mi_stages.csv",
                           mime="text/csv", key=f"mi_stages_{i}")
