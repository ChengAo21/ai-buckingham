"""Explicit live-analysis controls with per-session subprocess isolation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

import pandas as pd
import streamlit as st

from ..reports import render_report, report_sections, report_styles
from .components import json_panel, masthead, table
from .data import ROOT, archive_bytes, case_catalog, read_json, runs

MARKER = "AI_BUCKINGHAM_EVENT "


def _reader(pipe, updates, redactions):
    for line in iter(pipe.readline, ""):
        for value in redactions:
            if value:
                line = line.replace(value, "[redacted]")
        if line.startswith(MARKER):
            try:
                updates.put(json.loads(line[len(MARKER):]))
            except ValueError:
                pass
        elif line.strip():
            updates.put({"message": line.strip()[:1500]})
    pipe.close()


def start_job(config, credentials, output_dir, models):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT/"src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen([sys.executable, "-u", "-m", "ai_buckingham.web.runner"],
                               cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    process.stdin.write(json.dumps({"config":config,"credentials":credentials,"output_dir":str(output_dir),"models":models}))
    process.stdin.close()
    updates=queue.Queue()
    thread=threading.Thread(target=_reader,args=(process.stdout,updates,list(credentials.values())),daemon=True)
    thread.start()
    return {"process":process,"updates":updates,"events":[],"output_dir":str(output_dir),"thread":thread}


@st.fragment(run_every="2s")
def progress():
    job=st.session_state.get("analysis_job")
    if not job:
        return
    while True:
        try: job["events"].append(job["updates"].get_nowait())
        except queue.Empty: break
    process=job["process"]
    code=process.poll()
    path=Path(job["output_dir"])
    manifest=read_json(path/"manifest.json")
    if code is None:
        st.info("Analysis is running. Progress updates appear below.")
        if st.button("Cancel this calculation",key="cancel_job"):
            process.terminate()
            process.wait(timeout=10)
            job["events"].append({"status":"cancelled","message":"Calculation cancelled by this reader."})
            if manifest:
                manifest["status"]="cancelled"
                (path/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            st.rerun(scope="fragment")
    else:
        if manifest.get("status")=="completed":
            st.success("Calculation completed. Results and downloads are available below.")
        elif manifest.get("status")=="cancelled":
            st.warning("Calculation cancelled. Partial results remain available.")
        else:
            st.warning("Calculation stopped. See the error details below.")
    if job["events"]:
        progress_rows = [{key: event[key] for key in ("sequence", "node", "status", "message") if key in event}
                         for event in job["events"]]
        table(pd.DataFrame(progress_rows),"live_node_progress.csv","live_progress_csv")
    if manifest:
        json_panel("Current run manifest",manifest,key="live_manifest")
    trace=path/"evidence"/"optimization_trace.jsonl"
    if trace.is_file():
        st.caption("Open Case results after completion to inspect this run and its optimization history.")
    if code is not None and manifest:
        st.html(report_styles())
        for section in report_sections(path):
            if section["id"] == "results":
                st.html(section["html"])
        st.download_button("Download case report · HTML", render_report(path).encode("utf-8"),
                           file_name="analysis_report.html", mime="text/html", key="live_report")
        st.download_button("Download this run · ZIP",archive_bytes(path),file_name=path.name+".zip",mime="application/zip",key="live_bundle")
        if st.button("Clear finished job from this session",key="clear_job"):
            del st.session_state["analysis_job"]
            st.rerun()


def _open_saved_run(case_id, path):
    st.session_state["workflow_case"] = case_id
    st.session_state["workflow_run"] = path
    st.session_state["page_navigation"] = "Case results"


def saved_runs():
    available = runs()
    if not available:
        st.info("No saved results are available.")
        return
    catalog = case_catalog()
    case_ids = list(dict.fromkeys(path.parent.name for path in available))
    case_ids.sort(key=lambda case_id: list(catalog).index(case_id) if case_id in catalog else len(catalog))
    case_id = st.selectbox("Physical system", case_ids,
                           format_func=lambda value: catalog.get(value, {}).get("title", value),
                           key="saved_result_case")
    options = [path for path in available if path.parent.name == case_id]
    index = read_json(ROOT / "results" / "index.json")
    selected_id = next((item.get("selected_run") for item in index.get("cases", [])
                        if item.get("case_id") == case_id), None)
    default_index = next((i for i, path in enumerate(options) if path.name == selected_id), 0)
    selected = st.selectbox("Saved result", options, format_func=lambda path: path.name,
                            index=default_index, key=f"saved_result_run_{case_id}")
    st.button("Open case results", type="primary", key="open_saved_results",
              on_click=_open_saved_run, args=(case_id, selected))
    st.caption("Open the saved cards and figures without starting a new analysis.")
    with st.expander("Downloads", expanded=False):
        st.download_button("Download case report · HTML", render_report(selected).encode("utf-8"),
                           file_name=f"{case_id}_{selected.name}.html", mime="text/html", key="saved_run_report")
        st.download_button("Download run bundle · ZIP", archive_bytes(selected),
                           file_name=f"{case_id}_{selected.name}.zip", mime="application/zip", key="saved_run_bundle")


def numerical_settings(config, case_id):
    original = dict(config.get("manual_hyperparams") or {})
    hp = dict(original)
    with st.expander("Numerical settings", expanded=False):
        a, b = st.columns(2)
        similarity_labels = {None: "Use workflow selection", "auto": "Automatic selection",
                             "first_kind": "Type I · complete similarity", "second_kind": "Type II · incomplete similarity"}
        similarity_options = list(similarity_labels)
        initial_similarity = original.get("similarity_type")
        similarity = a.selectbox("Similarity setting", similarity_options,
                                 index=similarity_options.index(initial_similarity),
                                 format_func=similarity_labels.get, key=f"numerical_similarity_{case_id}")
        if similarity != initial_similarity:
            if similarity is None:
                hp.pop("similarity_type", None)
            else:
                hp["similarity_type"] = similarity
        denominator = b.number_input("Maximum rational denominator", min_value=1,
                                     value=int(original.get("max_rational_denominator", 5)),
                                     key=f"numerical_denominator_{case_id}")
        if denominator != original.get("max_rational_denominator", 5):
            hp["max_rational_denominator"] = int(denominator)

        method_labels = {None: "Use workflow selection", "auto": "Automatic selection",
                         "kraskov": "Kraskov estimator", "binning": "Binning estimator"}
        method_options = list(method_labels)
        initial_method = original.get("mi_method_used")
        method = a.selectbox("MI estimator", method_options, index=method_options.index(initial_method),
                             format_func=method_labels.get, key=f"numerical_mi_method_{case_id}")
        if method != initial_method:
            if method is None:
                hp.pop("mi_method_used", None)
            else:
                hp["mi_method_used"] = method
        parameter = b.number_input("MI parameter", min_value=1, value=original.get("mi_param_used"),
                                    placeholder="Use workflow selection", key=f"numerical_mi_parameter_{case_id}",
                                    help="Number of neighbors for the Kraskov estimator or bins for the binning estimator.")
        if parameter != original.get("mi_param_used"):
            if parameter is None:
                hp.pop("mi_param_used", None)
            else:
                hp["mi_param_used"] = int(parameter)

        st.markdown("##### Optimization budget")
        budget = st.radio("Execution budget", ["Case settings", "Reduced exploration budget"],
                          horizontal=True, label_visibility="collapsed", key="execution_budget")
        reduced = budget == "Reduced exploration budget"
        budget_defaults = {"cma_maxiter": 30 if reduced else 30000, "cma_popsize": 24 if reduced else 300}
        for column, (name, label) in zip(st.columns(2), [("cma_maxiter", "Maximum generations"), ("cma_popsize", "Population size")]):
            source_key = f"regime_{name}" if original.get(f"regime_{name}") is not None else name
            initial_value = budget_defaults[name] if reduced else original.get(source_key, budget_defaults[name])
            value = int(column.number_input(label, min_value=1 if name == "cma_maxiter" else 4,
                                             value=int(initial_value), key=f"numerical_{name}_{case_id}_{budget}"))
            if reduced or value != initial_value:
                hp[source_key] = value
        if reduced:
            st.caption("Reduced budgets use fewer generations and a smaller population than the case settings.")

        managed = {"similarity_type", "max_rational_denominator", "mi_method_used", "mi_param_used",
                   "cma_maxiter", "cma_popsize", "regime_cma_maxiter", "regime_cma_popsize"}
        if st.checkbox("Edit additional parameters", key=f"advanced_numerical_{case_id}"):
            additional = {name: value for name, value in original.items() if name not in managed}
            text = st.text_area("Additional parameters · JSON", json.dumps(additional, indent=2), height=220,
                                key=f"additional_parameters_{case_id}")
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Additional parameters must be a JSON object.")
                if managed.intersection(parsed):
                    raise ValueError("Use the controls above for similarity, MI and optimization-budget settings.")
                hp = {name: value for name, value in hp.items() if name in managed}
                hp.update(parsed)
            except ValueError as error:
                st.error(str(error))
                return False
        config["manual_hyperparams"] = hp
        st.download_button("Download analysis settings · JSON", json.dumps(config, ensure_ascii=False, indent=2),
                           file_name=f"{case_id}_settings.json", mime="application/json", key="live_config_download")
    return True


def render():
    masthead("Run analysis")
    job=st.session_state.get("analysis_job")
    if job:
        progress()
        if job["process"].poll() is None:
            return
    action = st.radio("Analysis mode", ["View saved results", "Run new analysis"], horizontal=True,
                       label_visibility="collapsed", key="analysis_action")
    if action == "View saved results":
        saved_runs()
        return
    catalog=case_catalog()
    mode=st.radio("Input",["Supplied physical case","My CSV and problem"],horizontal=True,key="live_input_mode")
    upload=None
    if mode=="Supplied physical case":
        case_id=st.selectbox("Case to execute",list(catalog),format_func=lambda k:catalog[k]["title"],key="live_case")
        config=dict(catalog[case_id]["config"])
        if not config:
            st.info("Case configuration is not available.")
            return
        prompt=st.text_area("Problem description",config.get("prompt",""),height=180,key=f"live_prompt_{case_id}")
        config["prompt"]=prompt
    else:
        case_id="custom"
        upload=st.file_uploader("CSV observations",type=["csv"],key="live_csv")
        prompt=st.text_area("Physical problem, intended response, governing quantities and any constraints",height=180,key="custom_prompt")
        config={"case_id":"custom","title":"Reader analysis","prompt":prompt,"manual_hyperparams":{}}
        if upload is not None:
            try:
                preview=pd.read_csv(upload)
                upload.seek(0)
                st.dataframe(preview.head(20),width="stretch")
                st.caption(f"{len(preview):,} rows · {len(preview.columns)} columns. The uploaded data will be included in this run's downloadable bundle.")
            except (ValueError,pd.errors.ParserError) as error:
                st.error(f"CSV could not be read: {error}")
                return
    if not numerical_settings(config, case_id):
        return
    with st.form("analysis_launch_form"):
        st.markdown("#### Model settings")
        st.caption("New analyses use the configured model services. Leave key fields empty to use host credentials.")
        a,b=st.columns(2)
        openai_key=a.text_input("OpenAI-compatible API key",type="password",key="live_openai_key")
        embedding_key=b.text_input("GLM embedding API key",type="password",key="live_embedding_key")
        endpoint=a.text_input("LLM base URL",value="https://api.openai.com/v1",key="live_llm_endpoint")
        embedding_endpoint=b.text_input("Embedding base URL",value="https://llmapi.paratera.com/v1",key="live_embedding_endpoint")
        analyzer=st.selectbox("LLM for semantic parsing",["gpt-5-mini","gpt-4.1-mini","gpt-4o-mini"],key="live_analyzer_model")
        st.caption("GPT-5 mini is used for the remaining workflow stages.")
        acknowledged=st.checkbox("Use these settings and send the analysis inputs to the configured model providers.",key="launch_acknowledged")
        submitted=st.form_submit_button("Start LangGraph analysis",type="primary")
    if submitted:
        if not acknowledged:
            st.error("Select the execution acknowledgement before starting a model-backed calculation.")
            return
        if not prompt.strip() or (mode=="My CSV and problem" and upload is None):
            st.error("Supply a problem description and a CSV dataset.")
            return
        if not (openai_key or os.getenv("OPENAI_API_KEY")) or not (embedding_key or os.getenv("PARATERA_API_KEY")):
            st.error("Both an LLM API key and an embedding API key are required for a fresh retrieval-grounded run.")
            return
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        output_dir=ROOT/"generated"/"cases"/case_id/run_id
        output_dir.mkdir(parents=True,exist_ok=False)
        if upload is not None:
            dataset=output_dir/"input.csv"
            dataset.write_bytes(upload.getvalue())
            config["dataset"]=dataset.relative_to(ROOT).as_posix()
        credentials={"OPENAI_API_KEY":openai_key,"OPENAI_BASE_URL":endpoint,
                     "PARATERA_API_KEY":embedding_key,"PARATERA_BASE_URL":embedding_endpoint}
        st.session_state["analysis_job"]=start_job(config,credentials,output_dir,{"analyzer_model":analyzer})
        st.rerun()
