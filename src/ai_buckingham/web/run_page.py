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

from .components import interpretation_commentary, json_panel, masthead, table
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
        table(pd.json_normalize(job["events"]),"live_node_progress.csv","live_progress_csv")
    if manifest:
        json_panel("Current run manifest",manifest,key="live_manifest")
    trace=path/"evidence"/"optimization_trace.jsonl"
    if trace.is_file():
        st.caption("Open Numerical diagnostics after completion to inspect the optimization history.")
    if code is not None and manifest:
        report=path/"report.html"
        if report.is_file():
            st.download_button("Download model report · HTML",report.read_bytes(),file_name="analysis_report.html",mime="text/html",key="live_report")
        st.download_button("Download this run · ZIP",archive_bytes(path),file_name=path.name+".zip",mime="application/zip",key="live_bundle")
        state=read_json(path/"state.json")
        interpretation_commentary(state.get("final_report"))
        if st.button("Clear finished job from this session",key="clear_job"):
            del st.session_state["analysis_job"]
            st.rerun()


def saved_runs():
    with st.expander("Saved runs and status", expanded=False):
        available = runs()
        if not available:
            st.info("No saved runs are available.")
            return
        catalog = case_catalog()
        records = []
        for path in available:
            manifest = read_json(path / "manifest.json")
            case_id = manifest.get("case_id", path.parent.name)
            records.append({"Case": catalog.get(case_id, {}).get("title", case_id), "Run": path.name,
                            "Status": manifest.get("status", "unknown"),
                            "Source": "Included run data" if path.is_relative_to(ROOT / "results") else "New analysis"})
        frame = pd.DataFrame(records)
        statuses = frame["Status"]
        columns = st.columns(3)
        columns[0].metric("Completed runs", int(statuses.isin(["completed", "complete", "success"]).sum()))
        columns[1].metric("Running runs", int(statuses.eq("running").sum()))
        columns[2].metric("Failed / stopped", int(statuses.isin(["failed", "blocked", "cancelled", "warning"]).sum()))
        table(frame, "saved_runs.csv", "saved_runs_csv")
        selected = st.selectbox("Saved run", available,
                                format_func=lambda path: f"{catalog.get(path.parent.name, {}).get('title', path.parent.name)} · {path.name}", key="saved_run_selection")
        json_panel("Saved run details", read_json(selected / "manifest.json"), key="saved_run_details")
        report = selected / "report.html"
        if report.is_file():
            st.download_button("Download raw run report · HTML", report.read_bytes(),
                               file_name=f"{selected.parent.name}_{selected.name}.html", mime="text/html", key="saved_run_report")
        st.download_button("Download run bundle · ZIP", archive_bytes(selected),
                           file_name=f"{selected.parent.name}_{selected.name}.zip", mime="application/zip", key="saved_run_bundle")


def render():
    masthead("Run analysis")
    st.markdown("A new run sends the problem description, relevant data previews and retrieval queries to the configured model providers. "
                "Credentials entered here are used only for this session's isolated calculation and are excluded from the saved scientific configuration.")
    saved_runs()
    job=st.session_state.get("analysis_job")
    if job:
        progress()
        if job["process"].poll() is None:
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
    with st.expander("Numerical settings",expanded=False):
        budget=st.radio("Execution budget",["Case settings","Reduced exploration budget"],horizontal=True,key="execution_budget")
        hp=dict(config.get("manual_hyperparams") or {})
        if budget=="Reduced exploration budget":
            a,b=st.columns(2)
            hp["cma_maxiter"]=int(a.number_input("Maximum CMA-ES generations",min_value=1,max_value=30000,value=30,key="exploration_generations"))
            hp["cma_popsize"]=int(b.number_input("Population size",min_value=4,max_value=1000,value=24,key="exploration_population"))
            st.caption("Reduced budgets generate an exploratory run and do not reproduce the paper's optimization budget.")
        text=st.text_area("Manual numerical overrides · JSON",json.dumps(hp,indent=2),height=190,key=f"manual_overrides_{case_id}_{budget}")
        try:
            parsed=json.loads(text)
            if not isinstance(parsed,dict):raise ValueError("Overrides must be a JSON object.")
            config["manual_hyperparams"]=parsed
        except ValueError as error:
            st.error(str(error))
            return
        json_panel("Submitted analysis configuration",config,key="live_config_preview")
    with st.form("analysis_launch_form"):
        st.markdown("#### Model connection")
        a,b=st.columns(2)
        openai_key=a.text_input("OpenAI-compatible API key",type="password",key="live_openai_key")
        embedding_key=b.text_input("GLM embedding API key",type="password",key="live_embedding_key")
        endpoint=a.text_input("LLM base URL",value="https://api.openai.com/v1",key="live_llm_endpoint")
        embedding_endpoint=b.text_input("Embedding base URL",value="https://llmapi.paratera.com/v1",key="live_embedding_endpoint")
        analyzer=st.selectbox("Analyzer model",["gpt-5-mini","gpt-4.1-mini","gpt-4o-mini"],key="live_analyzer_model")
        st.caption("Checker, planner and interpretation use GPT-5 mini. A key already configured on the host may be used when its field is left empty. Full scientific runs can be computationally expensive.")
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
