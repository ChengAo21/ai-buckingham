"""AI Buckingham: LangGraph dimensional analysis and reproducible evidence.

Importing the package does not create model clients, display a graph, seed the
process RNG, or make network requests.  Model initialization is explicit.
"""
from __future__ import annotations

__version__ = "1.0.0"


def build_agent_graph():
    from .graph import build_agent_graph as build
    return build()


def initialize_models(*args, **kwargs):
    from .models import initialize_models as initialize
    return initialize(*args, **kwargs)


def run_case(case_id, output_dir=None, record=True):
    """Run a configured case after explicit model initialization.

The standard runner exports the same artifacts used by notebooks and the web
viewer.  ``record`` remains an explicit compatibility argument; the standard
runner always retains scientific evidence.
    """
    if not record:
        raise ValueError("The public runner retains run evidence; use a graph directly for an unrecorded invocation.")
    from .cli import run_analysis
    return run_analysis(case_id, output_dir=output_dir, initialize=False)


def __getattr__(name):
    if name == "build_adaptive_agent_graph":
        return build_agent_graph
    if name == "run_analysis":
        from .cli import run_analysis
        return run_analysis
    if name in {"save_run_report", "save_State_2md"}:
        from .presentation import save_run_report
        return save_run_report
    raise AttributeError(name)
