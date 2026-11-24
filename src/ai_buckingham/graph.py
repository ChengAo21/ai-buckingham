"""Graph extracted from the author implementation."""
from __future__ import annotations

from langgraph.graph import StateGraph, END, START
from . import dimensions as _dimensions
from . import plotting as _plotting
from . import provenance as _provenance
from . import schemas as _schemas
from . import workflow as _workflow


def route_after_basis(state: _schemas.AdaptiveWorkflowState):
    return "ready" if state.get("workflow_status") == "basis_ready" else "failed"


def build_adaptive_agent_graph():
    """Build the default planner-driven, runtime-adaptive LangGraph agent."""
    builder = StateGraph(_schemas.AdaptiveWorkflowState)
    builder.add_node("Analyzer", _provenance.observe_workflow_node('Analyzer', _dimensions.analyze_problem_node))
    builder.add_node("Hallu_Checker", _provenance.observe_workflow_node('Hallu_Checker', _dimensions.physics_consistency_node))
    builder.add_node("Basis_Decider", _provenance.observe_workflow_node('Basis_Decider', _dimensions.adaptive_basis_decider_node))
    builder.add_node("Basis_Executor", _provenance.observe_workflow_node('Basis_Executor', _dimensions.adaptive_basis_executor_node))
    builder.add_node("Workflow_Planner", _provenance.observe_workflow_node('Workflow_Planner', _workflow.workflow_planner_node))
    builder.add_node("Workflow_Executor", _provenance.observe_workflow_node('Workflow_Executor', _workflow.workflow_executor_node))
    builder.add_node("Workflow_Replanner", _provenance.observe_workflow_node('Workflow_Replanner', _workflow.workflow_replanner_node))
    builder.add_node("Workflow_Report", _provenance.observe_workflow_node('Workflow_Report', _workflow.workflow_report_node))
    builder.add_node("Visualizer", _provenance.observe_workflow_node('Visualizer', _plotting.adaptive_visualizer_node))

    builder.add_edge(START, "Analyzer")
    builder.add_edge("Analyzer", "Hallu_Checker")
    builder.add_conditional_edges(
        "Hallu_Checker",
        route_after_check,
        {"retry": "Analyzer", "passed": "Basis_Decider", "blocked": END},
    )
    builder.add_edge("Basis_Decider", "Basis_Executor")
    builder.add_conditional_edges(
        "Basis_Executor",
        route_after_basis,
        {"ready": "Workflow_Planner", "failed": END},
    )
    builder.add_edge("Workflow_Planner", "Workflow_Executor")
    builder.add_conditional_edges(
        "Workflow_Executor",
        _workflow.route_after_workflow_execution,
        {"replan": "Workflow_Replanner", "report": "Workflow_Report"},
    )
    builder.add_edge("Workflow_Replanner", "Workflow_Executor")
    builder.add_edge("Workflow_Report", "Visualizer")
    builder.add_edge("Visualizer", END)
    return builder.compile()


def route_after_check(state: dict):
    last = (state.get("hallucination_decision") or [None])[-1]
    gate = state.get("dimension_audit") or {}
    if last == "RETRY" and gate.get("gate_status") == "RETRY":
        return "retry"
    if last == "PASSED" and gate.get("gate_status") == "PASSED" and gate.get("ledger_sha256") == state.get("dimension_ledger_sha256"):
        try:
            if _dimensions.ledger_sha256(state.get("dimension_ledger")) == gate.get("ledger_sha256"):
                return "passed"
        except Exception:
            pass
    return "blocked"


# Conventional public API; legacy name remains a compatible alias.
build_agent_graph = build_adaptive_agent_graph
