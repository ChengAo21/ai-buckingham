"""Variable-level retrieval and checking evidence from the historical benchmark."""
import pandas as pd
import plotly.express as px
import streamlit as st

from .components import chart, json_panel, masthead, table
from .data import case_catalog, first, read_csv, read_json, reproduction, runs, load_run


def _rounds_from_state(state, events=None):
    if state.get("checker_rounds"):
        return state["checker_rounds"]
    paired_rounds = []
    for event in events or []:
        update = event.get("update", {})
        if update.get("knowledge_retrieval"):
            audit = update.get("dimension_audit", {})
            paired_rounds.append({"retrieval": update["knowledge_retrieval"],
                                  "candidate_ledger": update.get("dimension_ledger", {}),
                                  "checker_response": audit.get("checker_audit", audit),
                                  "gate_status": audit.get("gate_status", "Not recorded"), "gate": audit})
    if paired_rounds:
        return paired_rounds
    retrieval = state.get("knowledge_retrieval")
    if retrieval:
        audit = state.get("dimension_audit", {})
        return [{"retrieval": retrieval, "candidate_ledger": state.get("dimension_ledger", {}),
                 "checker_response": audit.get("checker_audit", audit),
                 "gate_status": audit.get("gate_status", "Not recorded"), "gate": audit}]
    return []


def render():
    masthead("Retrieval evidence", "Inspect the exact query, retrieved definitions, dimensional assignments, and the checking decision for each variable.", "Grounded reasoning")
    base = reproduction("rag_ablation")
    source = st.radio("Evidence source", ["Archived 600-trial benchmark", "Included case run"], horizontal=True, key="rag_source")
    events = []
    if source == "Archived 600-trial benchmark":
        trials = base / "trials"
        models = sorted(p.name for p in trials.iterdir() if p.is_dir()) if trials.is_dir() else []
        if not models:
            st.info("The historical trial archive has not been installed.")
            return
        c1, c2, c3 = st.columns([1, 1, 2])
        model = c1.selectbox("Analyzer model", models, key="rag_model")
        available = sorted(p for p in (trials / model).iterdir() if p.is_dir() and (p / "on.json").is_file())
        case_codes = sorted({p.name.split("_")[0] for p in available})
        case_code = c2.selectbox("Benchmark case", case_codes, key="rag_case")
        selected = c3.selectbox("Trial identifier", [p for p in available if p.name.startswith(case_code + "_")], format_func=lambda p: p.name, key=f"rag_trial_{model}_{case_code}")
        record = read_json(selected / "on.json")
        initial = read_json(selected / "initial.json")
        st.caption("Historical record · Rephrased prompts are identified as challenge in the archive. IDs and source bytes are preserved.")
        json_panel("Initial Analyzer response", initial, key="rag_initial")
    else:
        available = runs()
        if not available:
            st.info("No saved case run has been recorded.")
            return
        path = st.selectbox("Case and run", available, format_func=lambda p: f"{p.parent.name} / {p.name}", key="rag_case_run")
        loaded = load_run(path)
        record, events = loaded["state"], loaded["events"]
        st.caption("Dimensional-checking results for the selected case.")
    rounds = _rounds_from_state(record, events)
    if not rounds:
        st.info("This source contains no recorded dimensional-checking round. A failed initial response may stop before retrieval.")
        return
    round_index = st.selectbox("Checking round", range(len(rounds)), format_func=lambda i: f"Round {i+1}", key="rag_round")
    selected_round = rounds[round_index]
    retrieval = selected_round.get("retrieval", selected_round.get("knowledge_retrieval", {})) or {}
    audit = selected_round.get("checker_response", selected_round.get("dimension_audit", {})) or {}
    if not audit:
        audit = (selected_round.get("checker_call") or {}).get("parsed_response", {})
    gate = selected_round.get("gate_status", (selected_round.get("gate") or {}).get("gate_status", "Not recorded"))
    metrics = st.columns(4)
    metrics[0].metric("Gate outcome", gate)
    metrics[1].metric("Retrieved records", retrieval.get("n_retrieved_records", "Not recorded"))
    metrics[2].metric("Embedding status", str(retrieval.get("embedding_status", "Not recorded")))
    metrics[3].metric("Retrieval mode", ", ".join(retrieval.get("retrieval_modes", [])) or str(retrieval.get("retrieval_mode", "Not recorded")))
    matches = retrieval.get("matches_by_variable", {})
    queries = retrieval.get("queries_by_variable", {})
    audit_items = audit.get("variables", []) if isinstance(audit, dict) else []
    variable_names = list(dict.fromkeys(list(queries) + list(matches) + [v.get("csv_name") for v in audit_items if v.get("csv_name")]))
    if not variable_names:
        json_panel("Recorded checking round", selected_round, key="rag_round_raw", expanded=True)
        return
    variable = st.selectbox("Physical quantity / CSV column", variable_names, key="rag_variable")
    st.markdown("### Query and ranked evidence")
    st.code(queries.get(variable, "Exact query was not recorded."), language=None, wrap_lines=True)
    records = matches.get(variable, [])
    if records:
        frame = pd.DataFrame(records)
        score_columns = [c for c in ["lexical_score", "semantic_score", "hybrid_score", "retrieval_score"] if c in frame]
        label = "canonical_quantity" if "canonical_quantity" in frame else "entry_id"
        if score_columns and label in frame:
            plot_data = frame.melt(id_vars=[label], value_vars=score_columns, var_name="Score component", value_name="Score")
            chart(px.bar(plot_data, x="Score", y=label, color="Score component", barmode="group", orientation="h"), "rag_ranking")
        preferred = ["entry_id", "canonical_quantity", "definition", "si_dimension", "si_unit", "lexical_score", "semantic_score", "hybrid_score", "source_identifier"]
        table(frame[[c for c in preferred if c in frame]], "retrieved_records.csv", "rag_records")
        for i, row in enumerate(records):
            json_panel(f"Record {i+1} · {row.get('entry_id', 'unidentified')}", row, key=f"rag_record_{i}")
    else:
        st.info("No retrieved record is stored for this quantity.")
    st.markdown("### Dimensional decision")
    decisions = [v for v in audit_items if v.get("csv_name") == variable]
    ledger = selected_round.get("candidate_ledger", {})
    quantities = [v for v in ledger.get("variables", []) if v.get("csv_name") == variable]
    if quantities:
        json_panel("Submitted quantity and SI exponents", quantities[0], key="rag_submitted", expanded=True)
    if decisions:
        decision = decisions[0]
        a, b = st.columns(2)
        a.markdown("**Decision:** " + str(decision.get("decision", "Not recorded")))
        b.markdown("**Support mode:** " + str(decision.get("support_mode", "Not recorded")))
        st.markdown(str(decision.get("reason", "No reason recorded.")))
        st.write("Cited records", decision.get("evidence_entry_ids", []))
        exponents = []
        for stage, key in [("Before", "before_si_exponents"), ("Proposed", "suggested_si_exponents")]:
            vector = decision.get(key)
            if isinstance(vector, dict):
                exponents.append({"Assignment": stage, **vector})
        if exponents:
            table(pd.DataFrame(exponents), "dimensional_comparison.csv", "rag_dimensions")
        json_panel("Complete quantity decision", decision, key="rag_decision")
    else:
        st.info("No per-quantity Checker decision is stored for this variable.")
    with st.expander("Exact evidence supplied to the Checker"):
        st.code(retrieval.get("evidence_markdown", "Not recorded"), language="markdown", wrap_lines=True)
    json_panel("Gate audit", selected_round.get("gate", {}), key="rag_gate")
    json_panel("Complete checking round", selected_round, key="rag_complete")
    st.caption("Historical replay recomputes all 247 semantic rankings for Sandgrain using archived query vectors. "
               "For the other nine cases, recorded top-k semantic scores are replayed; missing full query vectors limit full-ranking reconstruction.")
