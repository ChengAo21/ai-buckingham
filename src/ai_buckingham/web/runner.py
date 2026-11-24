"""One isolated live execution; receives credentials through stdin, never argv.

Each Streamlit job gets a separate interpreter so the implementation's model
globals and environment cannot be shared between reader sessions.
"""
from __future__ import annotations

import json
import os
import sys

MARKER = "AI_BUCKINGHAM_EVENT "


def emit(event):
    print(MARKER + json.dumps(event, ensure_ascii=True), file=sys.__stdout__, flush=True)


def main():
    payload = json.loads(sys.stdin.read())
    credentials = payload.pop("credentials", {})
    for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "PARATERA_API_KEY", "PARATERA_BASE_URL"]:
        value = credentials.get(key)
        if value:
            os.environ[key] = value
    # No remote tracing or telemetry is implicitly enabled by the web form.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        from ai_buckingham import initialize_models
        from ai_buckingham.cli import run_analysis
        initialize_models(**payload.get("models", {}))
        emit({"status": "initialized", "message": "Model clients initialized; graph execution is starting."})
        state = run_analysis(payload["config"], output_dir=payload["output_dir"], initialize=False,
                             display_cards=False, on_event=emit)
        emit({"status": "finished", "workflow_status": state.get("workflow_status"), "run_dir": payload["output_dir"]})
        return 0 if state.get("workflow_status") == "success" else 1
    except BaseException as error:
        message = str(error)
        for value in credentials.values():
            if value:
                message = message.replace(value, "[redacted]")
        emit({"status": "failed", "error_type": type(error).__name__, "message": message[:2000]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
