"""Models extracted from the author implementation."""
from __future__ import annotations
import random
import numpy as np

import os
from langchain_openai import ChatOpenAI
from . import config as _config
from . import provenance as _provenance
from . import schemas as _schemas


def _configure_tracing_from_environment():
    """Respect explicit tracing; LANGSMITH_TRACING wins if both aliases exist."""
    requested = os.getenv("LANGSMITH_TRACING")
    if requested is None:
        requested = os.getenv("LANGCHAIN_TRACING_V2", "false")
    value = "true" if str(requested).strip().lower() in {"true", "1", "yes", "on"} else "false"
    os.environ["LANGSMITH_TRACING"] = value
    os.environ["LANGCHAIN_TRACING_V2"] = value
    return value


interpretation_llm = None


analyzer_llm = None


analyzer_raw_llm = None


checker_llm = None


output_binding_checker_llm = None


planner_llm = None


_MODEL_ROLE_CONFIGURATION = {}


def configure_api_credentials(prompt_for_missing=False):
    """Read credentials from environment; getpass is an explicit optional action."""
    if prompt_for_missing:
        from getpass import getpass
        for name in ("OPENAI_API_KEY", "PARATERA_API_KEY"):
            if not os.getenv(name):
                os.environ[name] = getpass(f"Enter {name}: ")
    return {"openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "embedding_key_configured": bool(os.getenv("PARATERA_API_KEY"))}


def setup_environment(model_name):
    """Construct a model client on explicit initialization."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Supply it through the environment or configure_api_credentials(prompt_for_missing=True).")
    _configure_tracing_from_environment()
    _config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {"model": model_name, "temperature": 0.0, "request_timeout": 500}
    endpoint = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if endpoint:
        kwargs["base_url"] = endpoint
    if os.getenv('OPENAI_PROXY'):
        model = ChatOpenAI(**kwargs)
        _provenance.attach_http_recording(model.http_client)
        return model
    kwargs['http_client'] = _provenance.RecordedHttpClient(
        base_url=endpoint or 'https://api.openai.com/v1', timeout=500
    )
    return ChatOpenAI(**kwargs)


def _require_model_ready(model, role):
    """Return an initialized model without adding execution switches."""
    if model is None:
        raise RuntimeError(
            f"The {role} model is not initialized. "
            "Call initialize_models() first."
        )
    return model


def initialize_models(
    analyzer_model="gpt-5-mini",
    checker_model="gpt-5-mini",
    planner_model="gpt-5-mini",
    interpretation_model="gpt-5-mini",
):
    """Initialize model clients; this function sends no model requests."""
    global interpretation_llm, analyzer_llm, analyzer_raw_llm, checker_llm, planner_llm, _MODEL_ROLE_CONFIGURATION
    global output_binding_checker_llm

    _configure_tracing_from_environment()
    initialize_random_state()

    output_binding_model = "gpt-5-mini"
    model_names = {
        analyzer_model, checker_model, planner_model, interpretation_model, output_binding_model
    }
    clients = {name: setup_environment(name) for name in model_names}

    interpretation_llm = clients[interpretation_model]
    analyzer_llm = clients[analyzer_model].with_structured_output(
        _schemas.AnalyzerDimensionLedger, method="function_calling"
    )
    analyzer_raw_llm = clients[analyzer_model].with_structured_output(
        _schemas.AnalyzerDimensionLedger, method="function_calling", include_raw=True
    )
    checker_llm = clients[checker_model].with_structured_output(
        _schemas.CheckerDimensionAudit, method="function_calling"
    )
    output_binding_checker_llm = clients[output_binding_model].with_structured_output(
        _schemas.OutputColumnSelection, method="function_calling"
    )
    planner_llm = clients[planner_model].with_structured_output(
        _schemas.WorkflowPlan, method="function_calling"
    )
    _MODEL_ROLE_CONFIGURATION = {
        "analyzer": analyzer_model, "checker": checker_model,
        "planner": planner_model, "interpretation": interpretation_model,
        "output_binding_checker": output_binding_model,
    }
    return {
        "analyzer": analyzer_model,
        "checker": checker_model,
        "planner": planner_model,
        "interpretation": interpretation_model,
        "basis": "deterministic_calc_basis_tool",
    }


_RUNTIME_INITIALIZED = False

def initialize_random_state():
    """Apply the original import-time seeds once, at explicit startup."""
    global _RUNTIME_INITIALIZED
    if not _RUNTIME_INITIALIZED:
        random.seed(_config.GLOBAL_RANDOM_SEED)
        np.random.seed(_config.GLOBAL_RANDOM_SEED)
        _RUNTIME_INITIALIZED = True
