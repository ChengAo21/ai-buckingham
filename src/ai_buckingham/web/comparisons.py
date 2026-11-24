"""Interactive views over the unmodified recorded evaluation tables."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .components import chart, figures, json_panel, masthead, table
from .data import read_csv, read_json, reproduction


def benchmark():
    base = reproduction("rag_ablation")
    frame = read_csv(base / "results" / "trial_audit.csv")
    if frame.empty:
        st.info("The recorded benchmark table is not available.")
        return
    models = st.multiselect("Parsing models", sorted(frame.analyzer.unique()), default=sorted(frame.analyzer.unique()), key="benchmark_models")
    cols = st.columns(2)
    cases = cols[0].multiselect("Cases", sorted(frame.case.unique()), default=sorted(frame.case.unique()), key="benchmark_cases")
    variants = cols[1].multiselect("Prompt variants", sorted(frame.prompt_variant.unique()), default=sorted(frame.prompt_variant.unique()), key="benchmark_variants")
    selected = frame[frame.analyzer.isin(models) & frame.case.isin(cases) & frame.prompt_variant.isin(variants)].copy()
    metrics = st.columns(4)
    metrics[0].metric("Selected scheduled trials", len(selected))
    metrics[1].metric("First-pass successes", int(selected.validated_result.eq("direct").sum()))
    metrics[2].metric("Recovered after revision", int(selected.validated_result.eq("recovered").sum()))
    metrics[3].metric("Final successes", int(selected.validated_result.isin(["direct","recovered"]).sum()))
    if not selected.empty:
        counts = selected.groupby(["analyzer","validated_result"]).size().rename("Trials").reset_index()
        chart(px.bar(counts, x="analyzer", y="Trials", color="validated_result", labels={"analyzer":"Parsing model","validated_result":"Reference-scored outcome"},category_orders={"validated_result":["direct","recovered","failed"]}), "benchmark_success")
        usage_fields = [key for key in ["initial_analyzer_known_tokens","first_checker_known_tokens","analyzer_revision_known_tokens","second_checker_known_tokens"] if key in selected]
        if usage_fields:
            usage = selected.groupby("analyzer")[usage_fields].mean().reset_index().melt(id_vars="analyzer",var_name="Stage",value_name="Known tokens per scheduled trial")
            chart(px.bar(usage,x="analyzer",y="Known tokens per scheduled trial",color="Stage",labels={"analyzer":"Parsing model"}),"benchmark_tokens")
        table(selected, "selected_trial_audit.csv")
    st.caption("Three parsing LLMs, ten cases, two prompt variants and ten repeats produce 600 scheduled trials. "
               "GPT-5 mini checks every model's assignments. Final success includes at most one parsing revision. "
               "These comparisons do not constitute a separate RAG-on versus RAG-off experiment.")
    a,b = st.columns(2)
    with a:
        json_panel("Reference-scored model aggregates", read_json(base/"results"/"model_aggregates.json"))
    with b:
        json_panel("Recorded archive coverage", read_json(base/"results"/"coverage.json"))
    st.markdown("#### Retrieval replay across the recorded checking rounds")
    rounds = read_csv(base/"results"/"rag_replay_rounds.csv")
    if not rounds.empty:
        if models:
            rounds = rounds[rounds.analyzer.isin(models)]
        if cases:
            rounds = rounds[rounds.pair_id.str.split("_").str[0].isin(cases)]
        if variants:
            rounds = rounds[rounds.pair_id.str.split("_").str[1].isin(variants)]
        if not rounds.empty:
            chart(px.histogram(rounds,x="replayed_gate",color="analyzer",barmode="group",labels={"replayed_gate":"Replayed gate outcome"}),"benchmark_rounds")
        table(rounds,"selected_checker_rounds.csv")
    json_panel("Replay coverage and exactness",read_json(base/"results"/"rag_replay_summary.json"))
    with st.expander("Published summary tables and API accounting"):
        for filename in ["table1_success.csv","table30_display.csv","api_usage_by_variant.csv"]:
            table(read_csv(base/"results"/filename),filename,"benchmark_"+filename)
    st.caption("Known token totals do not impute missing usage from transport failures. API durations include recorded attempts and retries. Reference scoring can differ from the original gate outcome when the response column changed.")


def cess():
    base = reproduction("regime_comparison")
    coordinates = read_csv(base/"data"/"recorded_gradient_membership.csv")
    scores = read_csv(base/"results"/"regional_cell_scores.csv")
    sensitivity = read_csv(base/"results"/"regional_sensitivity.csv")
    if coordinates.empty or scores.empty:
        st.info("The recorded Cess comparison files are not available.")
        return
    a,b,c = st.columns(3)
    conditions = sorted(scores.conditioning.unique())
    condition = a.selectbox("MI conditioning",conditions,index=conditions.index("log_standardized") if "log_standardized" in conditions else 0,key="cess_condition")
    m_values = sorted(scores.M.unique().tolist())
    count = b.selectbox("Geometric regions M",m_values,index=m_values.index(3) if 3 in m_values else 0,key="cess_m")
    seed = c.selectbox("K-means seed",sorted(scores.seed.unique().tolist()),key="cess_seed")
    selected = scores[(scores.conditioning==condition)&(scores.M==count)&(scores.seed==seed)]
    membership_file=base/"results"/"regional_memberships.npz"
    categories=None
    if membership_file.is_file():
        prefix=f"{condition}__M{count}__seed{seed}"
        with np.load(membership_file,allow_pickle=False) as arrays:
            if prefix+"__category" in arrays:
                categories=arrays[prefix+"__category"].copy()
    left,right=st.columns(2)
    with left:
        if categories is not None and len(categories)==len(coordinates):
            frame=coordinates.copy()
            frame["Relevance category"]=[{0:"Pi₁ dominant",1:"Pi₂ dominant",2:"Both"}.get(int(v),str(v)) for v in categories]
            fig=px.scatter(frame,x="Re_tau",y="eta",color="Relevance category",log_x=True,log_y=True,hover_data=["row_id","y_plus"],title=f"IT-pi · M={count}")
            settings=read_json(base/"results"/"figure_settings.json")
            band=settings.get("transition_range",{})
            if band:
                fig.add_hrect(y0=band["eta_min"],y1=band["eta_max"],fillcolor="#8B6BA5",opacity=.14,line_width=0)
            chart(fig,"cess_regional")
        else:
            st.info("Pointwise regional category arrays are unavailable for this selection.")
    with right:
        chart(px.scatter(coordinates,x="Re_tau",y="eta",color="original_regime",log_x=True,log_y=True,hover_data=["row_id","y_plus"],
                         labels={"original_regime":"Physical regime"},
                         title="AI Buckingham<br><sup>Recorded gradient partition</sup>"),"cess_gradient")
    st.caption("Both panels use the same 1,000 Cess observations. M controls the geometric partition resolution; K identifies the gradient regimes. The shaded band spans the recorded intermediate samples.")
    table(selected,"selected_regional_scores.csv")
    if not sensitivity.empty:
        subset=sensitivity[(sensitivity.conditioning==condition)]
        fields=[c for c in ["Both_intermediate_precision","Both_intermediate_recall","category_agreement_with_first_seed"] if c in subset]
        if fields:
            metric=st.selectbox("Regional sensitivity metric",fields,key="cess_sensitivity_metric")
            chart(px.line(subset,x="M",y=metric,color=subset.seed.astype(str),markers=True,labels={"color":"Seed"}),"cess_sensitivity")
        table(subset,"regional_sensitivity.csv")
    json_panel("Recorded global fit configuration",read_json(base/"results"/"global_itpi_config.json"))
    figures(base/"results","cess_recorded")
    st.caption("This page reads the recorded global IT-pi fit and gradient memberships. Changing M or seed selects an archived regional calculation; it does not repeat the global optimization. Negative MI estimates and ratios outside [0,1] are retained in the data.")


def velocity():
    base=reproduction("velocity_comparison")
    frame=read_csv(base/"results"/"transformed_native_profiles.csv")
    reference=read_csv(base/"results"/"incompressible_reference.csv")
    if frame.empty:
        st.info("The recorded velocity-comparison files are not available.")
        return
    a,b=st.columns(2)
    selected_profiles=a.multiselect("DNS profiles",sorted(frame.case.unique().tolist()),default=sorted(frame.case.unique().tolist()),key="velocity_profiles")
    methods=b.multiselect("Transformations",["Raw","TL","IT-pi","AI Buckingham"],default=["Raw","TL","IT-pi","AI Buckingham"],key="velocity_methods")
    coordinates={"Raw":("y_plus","U_plus"),"TL":("x_TL","U_TL"),"IT-pi":("x_IT","U_IT"),"AI Buckingham":("x_AI","U_AI")}
    fig=go.Figure()
    colors={"Raw":"#95A3B1","TL":"#8B6BA5","IT-pi":"#C17837","AI Buckingham":"#245B78"}
    for method in methods:
        x,y=coordinates[method]
        for i, profile in enumerate(selected_profiles):
            subset=frame[frame.case==profile]
            fig.add_trace(go.Scatter(x=subset[x],y=subset[y],mode="lines",name=method,legendgroup=method,showlegend=i==0,
                                    line={"color":colors[method],"width":1.5},hovertemplate=f"{method} · profile {profile}<br>x=%{{x:.3g}}<br>U=%{{y:.3g}}<extra></extra>"))
    if not reference.empty:
        fig.add_trace(go.Scatter(x=reference.y_plus,y=reference.U_plus,mode="lines",name="Incompressible reference",line={"color":"#142A3A","width":3,"dash":"dash"}))
    fig.update_layout(xaxis_type="log",xaxis_title="Matched wall-distance coordinate",yaxis_title="Transformed velocity",height=580)
    chart(fig,"velocity_profiles_plot")
    st.caption("Four compressible channel profiles and two pipe profiles are compared with the same incompressible reference. Plot selections do not change the recorded numerical error calculation.")
    left,right=st.columns(2)
    rms=read_csv(base/"results"/"six_profile_pointwise_RMS.csv")
    with left:
        if not rms.empty:
            chart(px.line(rms[rms.method.isin(methods)],x="y_plus_reference",y="RMS_Uplus",color="method",log_x=True,title="RMS over all six profiles"),"velocity_pointwise")
    intervals=read_csv(base/"results"/"normalized_RMS_intervals.csv")
    with right:
        if not intervals.empty:
            interval=st.radio("Wall-distance interval",["0–30","30–200"],horizontal=True,key="velocity_interval")
            low=0 if interval=="0–30" else 30
            subset=intervals[(intervals.y_plus_min==low)&intervals.case.isin(selected_profiles)&intervals.method.isin(methods)]
            chart(px.bar(subset,x="case",y="normalized_RMS_percent",color="method",barmode="group",title="Normalized RMS error (%)"),"velocity_interval_plot")
    table(frame[frame.case.isin(selected_profiles)],"selected_velocity_profiles.csv")
    table(intervals,"normalized_RMS_intervals.csv")
    table(read_csv(base/"results"/"normalized_RMS_across_profiles.csv"),"normalized_RMS_across_profiles.csv")
    figures(base/"results","velocity_recorded")


def render():
    masthead("Comparisons & evaluation", "Explore recorded semantic-checking outcomes, regional relevance and velocity-transform accuracy.", "Evaluation")
    selection=st.radio("Evaluation",["LLM benchmark & retrieval replay","Cess regime comparison","Velocity transformations"],horizontal=True,key="comparison_page")
    if selection=="LLM benchmark & retrieval replay": benchmark()
    elif selection=="Cess regime comparison": cess()
    else: velocity()
