"""The paper's dimensional-checking and physical comparison results."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .components import chart, masthead
from .data import ALIASES, case_catalog, read_csv, read_json, reproduction


def benchmark():
    base = reproduction("rag_ablation") / "results"
    frame = read_csv(base / "trial_audit.csv")
    if frame.empty:
        st.info("The dimensional-checking results are unavailable.")
        return
    models = st.multiselect("LLMs", sorted(frame.analyzer.unique()),
                            default=sorted(frame.analyzer.unique()),
                            format_func=lambda model: model.replace("-mini", " mini"), key="benchmark_models")
    left, right = st.columns(2)
    catalog = case_catalog()
    cases = left.multiselect("Cases", sorted(frame.case.unique()), default=sorted(frame.case.unique()),
                             format_func=lambda case: catalog.get(ALIASES.get(case.lower(), case.lower()), {}).get("title", case),
                             key="benchmark_cases")
    variants = right.multiselect("Prompt variants", sorted(frame.prompt_variant.unique()),
                                  default=sorted(frame.prompt_variant.unique()),
                                  format_func=lambda variant: {"original": "Original", "challenge": "Rephrased"}.get(variant, variant),
                                  key="benchmark_variants")
    selected = frame[frame.analyzer.isin(models) & frame.case.isin(cases)
                     & frame.prompt_variant.isin(variants)].copy()
    metrics = st.columns(4)
    metrics[0].metric("Selected scheduled trials", len(selected))
    metrics[1].metric("First-pass successes", int(selected.validated_result.eq("direct").sum()))
    metrics[2].metric("Recovered after revision", int(selected.validated_result.eq("recovered").sum()))
    metrics[3].metric("Final successes", int(selected.validated_result.isin(["direct", "recovered"]).sum()))
    if not selected.empty:
        counts = selected.assign(**{"Init.": selected.validated_result.eq("direct"),
                                     "Final": selected.validated_result.isin(["direct", "recovered"])})
        counts = counts.groupby("analyzer")[["Init.", "Final"]].sum().reset_index()
        counts = counts.melt(id_vars="analyzer", var_name="Stage", value_name="Successes")
        success = px.bar(counts, x="analyzer", y="Successes", color="Stage", text="Successes", barmode="group",
                         labels={"analyzer": "LLMs", "Stage": ""}, title="LLM ablation in semantic parsing",
                         category_orders={"Stage": ["Init.", "Final"]},
                         color_discrete_map={"Init.": "#0072B2", "Final": "#009E73"})
        success.update_xaxes(tickvals=list(models), ticktext=[model.replace("-mini", " mini") for model in models])
        chart(success, "benchmark_success")
        stages = {"initial_analyzer_known_tokens": "Init.", "first_checker_known_tokens": "Chk.1",
                  "analyzer_revision_known_tokens": "Rev.", "second_checker_known_tokens": "Chk.2"}
        usage = selected.groupby("analyzer")[list(stages)].mean().div(1000).reset_index()
        usage = usage.melt(id_vars="analyzer", var_name="Stage", value_name="Token usage (k tokens/trial)")
        usage["Stage"] = usage.Stage.map(stages)
        left, right = st.columns(2)
        with left:
            tokens = px.bar(usage, x="analyzer", y="Token usage (k tokens/trial)", color="Stage",
                            labels={"analyzer": "LLMs", "Stage": ""}, title="Token usage",
                            color_discrete_sequence=["#0072B2", "#009E73", "#E69F00", "#D55E00"])
            tokens.update_xaxes(tickvals=list(models), ticktext=[model.replace("-mini", " mini") for model in models])
            chart(tokens, "benchmark_tokens")
        with right:
            duration = selected.groupby("analyzer").all_api_seconds.mean().rename("API time (s/trial)").reset_index()
            timing = px.bar(duration, x="analyzer", y="API time (s/trial)", text_auto=".1f",
                            labels={"analyzer": "LLMs"}, title="API time")
            timing.update_xaxes(tickvals=list(models), ticktext=[model.replace("-mini", " mini") for model in models])
            chart(timing, "benchmark_api_time")
    with st.expander("Downloads"):
        for filename, label in [("table1_success.csv", "Dimensional-checking success table"),
                                ("table30_display.csv", "API usage table")]:
            path = base / filename
            if path.is_file():
                st.download_button("Download " + label + " · CSV", path.read_bytes(), file_name=filename,
                                   mime="text/csv", key="benchmark_" + filename)


def cess():
    base = reproduction("regime_comparison")
    figure = base / "results" / "Cess_ITPI_comparison.png"
    coordinates = read_csv(base / "data" / "recorded_gradient_membership.csv")
    settings = read_json(base / "results" / "figure_settings.json")
    membership_file = base / "results" / "regional_memberships.npz"
    if not coordinates.empty and membership_file.is_file():
        with np.load(membership_file, allow_pickle=False) as arrays:
            counts = [count for count in settings["regional_counts"]
                      if f"{settings['main_condition']}__M{count}__seed{settings['main_seed']}__category" in arrays]
            count = st.selectbox("Geometric regions M", counts, index=counts.index(settings["main_M"][0]), key="cess_m")
            prefix = f"{settings['main_condition']}__M{count}__seed{settings['main_seed']}"
            categories = arrays[prefix + "__category"].copy()
            regions = arrays[prefix + "__region"].copy()
        category_labels = {0: "Π<sub>1</sub><sup>IT</sup> dominant", 1: "Π<sub>2</sub><sup>IT</sup> dominant",
                           2: "Both Π<sub>1</sub><sup>IT</sup> and Π<sub>2</sub><sup>IT</sup>"}
        mi_colors = {category_labels[index]: settings["MI_colors"][name]
                     for index, name in enumerate(("X1", "X2", "Both"))}
        selected = coordinates.copy()
        selected["Dominant groups"] = [category_labels[int(value)] for value in categories]
        selected["Region"] = regions + 1
        left, right = st.columns(2)
        with left:
            comparison = px.scatter(selected, x="Re_tau", y="eta", color="Dominant groups", log_x=True, log_y=True,
                                    color_discrete_map=mi_colors, category_orders={"Dominant groups": list(mi_colors)},
                                    hover_data=["row_id", "y_plus", "Region"], title=f"IT–π, M = {count}")
            _cess_axes(comparison, settings)
            chart(comparison, "cess_regional", compact_title=True)
        with right:
            gradient = coordinates.copy()
            names = {"Inner": "Near-wall", "Mixed": "Intermediate", "Outer": "Outer"}
            gradient["Physical regime"] = gradient.original_regime.map(names)
            colors = {names[key]: value for key, value in settings["gradient_colors"].items()}
            comparison = px.scatter(gradient, x="Re_tau", y="eta", color="Physical regime", log_x=True, log_y=True,
                                    color_discrete_map=colors, category_orders={"Physical regime": ["Near-wall", "Intermediate", "Outer"]},
                                    hover_data=["row_id", "y_plus"], title=f"AI Buckingham, K = {gradient.original_regime.nunique()}")
            _cess_axes(comparison, settings)
            chart(comparison, "cess_gradient", compact_title=True)
    if not figure.is_file() and coordinates.empty:
        st.info("The Cess comparison figure is unavailable.")
        return
    if figure.is_file():
        st.image(str(figure), width="stretch")
    with st.expander("Downloads"):
        for extension in ("png", "pdf"):
            path = figure.with_suffix("." + extension)
            if path.is_file():
                st.download_button("Download comparison figure · " + extension.upper(), path.read_bytes(),
                                   file_name=path.name, mime="image/png" if extension == "png" else "application/pdf",
                                   key="cess_figure_" + extension)
        scores = read_csv(base / "results" / "regional_cell_scores.csv")
        if not scores.empty:
            selected = scores[(scores.conditioning == settings["main_condition"])
                              & scores.M.isin(settings["main_M"]) & (scores.seed == settings["main_seed"])]
            columns = ["M", "region", "n", "I_X1_Y", "I_X2_Y", "I_joint_Y", "ratio1", "ratio2", "category"]
            st.download_button("Download regional MI values · CSV", selected[columns].to_csv(index=False).encode("utf-8"),
                               file_name="cess_regional_MI.csv", mime="text/csv", key="cess_regional_scores")


def _cess_axes(figure, settings):
    figure.update_traces(marker={"size": 7, "opacity": settings["plot"]["scatter_alpha"],
                                  "line": {"width": .3, "color": "white"}})
    for trace in figure.data:
        trace.name = trace.name + f" (n={len(trace.x)})"
    figure.update_xaxes(title="Re<sub>τ</sub>", range=np.log10(settings["plot"]["xlim"]).tolist(),
                        tickvals=[1e3, 1e5, 1e7, 1e9],
                        ticktext=["10<sup>3</sup>", "10<sup>5</sup>", "10<sup>7</sup>", "10<sup>9</sup>"])
    figure.update_yaxes(title="y/h", range=np.log10(settings["plot"]["ylim"]).tolist(),
                        tickvals=[1e-6, 1e-4, 1e-2, 1],
                        ticktext=["10<sup>−6</sup>", "10<sup>−4</sup>", "10<sup>−2</sup>", "1"])
    band = settings["transition_range"]
    figure.add_hrect(y0=band["eta_min"], y1=band["eta_max"], layer="below", line_width=0,
                     fillcolor=settings["plot"]["band_color"], opacity=settings["plot"]["band_alpha"])
    figure.update_layout(height=500, legend_title_text="")


def velocity():
    base = reproduction("velocity_comparison") / "results"
    frame = read_csv(base / "transformed_native_profiles.csv")
    reference = read_csv(base / "incompressible_reference.csv")
    if frame.empty:
        st.info("The velocity-transformation results are unavailable.")
        return
    left, right = st.columns(2)
    selected_profiles = left.multiselect("DNS profiles", sorted(frame.case.unique().tolist()),
                                         default=sorted(frame.case.unique().tolist()), key="velocity_profiles")
    labels = {"Raw": "Untransformed", "TL": "TL", "IT-pi": "IT–π", "AI Buckingham": "AI Buckingham"}
    methods = right.multiselect("Transformations", ["Raw", "TL", "IT-pi", "AI Buckingham"],
                                 default=["Raw", "TL", "IT-pi", "AI Buckingham"],
                                 format_func=labels.get, key="velocity_methods")
    coordinates = {"Raw": ("y_plus", "U_plus"), "TL": ("x_TL", "U_TL"),
                   "IT-pi": ("x_IT", "U_IT"), "AI Buckingham": ("x_AI", "U_AI")}
    figure = go.Figure()
    colors = {"Raw": "#888888", "TL": "#0072B2", "IT-pi": "#D55E00", "AI Buckingham": "#009E73"}
    for method in methods:
        x, y = coordinates[method]
        for index, profile in enumerate(selected_profiles):
            subset = frame[frame.case == profile]
            figure.add_trace(go.Scatter(x=subset[x], y=subset[y], mode="lines", name=labels[method],
                                        legendgroup=method, showlegend=index == 0,
                                        line={"color": colors[method], "width": 2.8},
                                        hovertemplate=f"{labels[method]} · profile {profile}<br>x=%{{x:.3g}}<br>U=%{{y:.3g}}<extra></extra>"))
    if not reference.empty:
        figure.add_trace(go.Scatter(x=reference.y_plus, y=reference.U_plus, mode="lines", name="DNS reference",
                                    line={"color": "#142A3A", "width": 3.3, "dash": "dash"}))
    figure.update_layout(xaxis_type="log", xaxis_title="Matched wall-distance coordinate",
                         yaxis_title="Transformed velocity", height=580)
    chart(figure, "velocity_profiles_plot")
    left, right = st.columns(2)
    rms = read_csv(base / "six_profile_pointwise_RMS.csv")
    with left:
        if not rms.empty:
            errors = px.line(rms[rms.method.isin(methods)], x="y_plus_reference", y="RMS_Uplus", color="method",
                             log_x=True, log_y=True, color_discrete_map=colors, title="RMS over all six profiles",
                             labels={"y_plus_reference": "y⁺", "RMS_Uplus": "RMS error", "method": ""})
            errors.update_traces(line_width=2.8)
            for trace in errors.data:
                trace.name = labels[trace.name]
            chart(errors, "velocity_pointwise")
    intervals = read_csv(base / "normalized_RMS_intervals.csv")
    with right:
        if not intervals.empty:
            interval = st.radio("Wall-distance interval", ["0–30", "30–200"], horizontal=True, key="velocity_interval")
            low = 0 if interval == "0–30" else 30
            subset = intervals[(intervals.y_plus_min == low) & intervals.case.isin(selected_profiles)
                               & intervals.method.isin(methods)]
            errors = px.bar(subset, x="case", y="normalized_RMS_percent", color="method", barmode="group",
                            color_discrete_map=colors, title="Normalized RMS error (%)",
                            labels={"case": "Profile", "normalized_RMS_percent": "Normalized RMS error (%)", "method": ""})
            for trace in errors.data:
                trace.name = labels[trace.name]
            chart(errors, "velocity_interval_plot")
    comparison = base / "TL_comparison.png"
    if comparison.is_file():
        st.image(str(comparison), width="stretch")
    with st.expander("Downloads"):
        for filename, label in [("transformed_native_profiles.csv", "Transformed profiles"),
                                ("six_profile_pointwise_RMS.csv", "RMS across six profiles"),
                                ("normalized_RMS_intervals.csv", "Per-profile normalized RMS errors"),
                                ("normalized_RMS_across_profiles.csv", "Mean normalized RMS errors")]:
            path = base / filename
            if path.is_file():
                st.download_button("Download " + label + " · CSV", path.read_bytes(), file_name=filename,
                                   mime="text/csv", key="velocity_" + filename)
        vector = base / "TL_comparison.pdf"
        if vector.is_file():
            st.download_button("Download comparison figure · PDF", vector.read_bytes(), file_name=vector.name,
                               mime="application/pdf", key="velocity_comparison_figure")


def render():
    masthead("Comparisons")
    selection = st.radio("Comparison", ["LLM ablation and API usage in dimensional checking", "Cess regime comparison", "Velocity transformations"],
                          horizontal=True, label_visibility="collapsed", key="comparison_page")
    if selection == "LLM ablation and API usage in dimensional checking":
        benchmark()
    elif selection == "Cess regime comparison":
        cess()
    else:
        velocity()
