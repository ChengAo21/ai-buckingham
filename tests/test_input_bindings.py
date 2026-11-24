"""Explicit case order is bound by name, with the raw Analyzer reply retained."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

from langgraph.graph import END, START, StateGraph
import numpy as np
import pandas as pd
import pytest

from ai_buckingham import cli, dimensions, regimes, schemas, workflow
from ai_buckingham.bindings import bind_analyzer_order, frontend_bindings
from ai_buckingham.schemas import FrontendContractError


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case_and_ledger():
    config = json.loads((ROOT / "configs/cases/rough_pipe.json").read_text(encoding="utf-8"))
    ledger = {"output_variable_name": config["output_column"],
              "output_variable_reasoning": "Configured response",
              "output_variable_confidence": "high",
              "input_variable_names": list(config["input_columns"]),
              "variables": [{"csv_name": name, "quantity_name": name,
                             "physical_definition": "Physical input selected by exact name",
                             "si_exponents": {axis: str(value) for axis, value in
                                              config["expected_dimensions"]["inputs"][name].items()}}
                            for name in config["input_columns"]]}
    return config, ledger


def compile_geometry(config, ledger):
    frame = pd.read_csv(ROOT / config["dataset"])
    expected = frontend_bindings(config)["frontend_expected_inputs"]
    bound, evidence = bind_analyzer_order(ledger, expected)
    compiled = dimensions.compile_ledger(bound, list(frame.columns), expected_inputs=expected,
                                         expected_output=config["output_column"])
    basis = dimensions.calc_basis_tool.invoke({"D_in": compiled["contract"]["D_in"],
                                              "var_names": bound["input_variable_names"]})
    matrix = np.asarray(basis["basis_matrices"])
    coordinates = np.log(frame[bound["input_variable_names"]].to_numpy(dtype=float)) @ regimes._schmidt_orthonormalize_basis_rows(matrix).T
    return compiled, matrix, coordinates, evidence


def test_explicit_contract_binds_permuted_names_and_records_to_identical_geometry(case_and_ledger):
    config, ledger = case_and_ledger
    baseline = compile_geometry(config, ledger)
    shuffled = deepcopy(ledger)
    shuffled["input_variable_names"] = list(reversed(shuffled["input_variable_names"]))
    shuffled["variables"] = shuffled["variables"][2:] + shuffled["variables"][:2]
    snapshot = deepcopy(shuffled)
    replay = compile_geometry(config, shuffled)
    assert baseline[0] == replay[0]
    np.testing.assert_array_equal(baseline[1], replay[1])
    np.testing.assert_array_equal(baseline[2], replay[2])
    assert shuffled == snapshot
    assert replay[3]["observed_input_order"] == snapshot["input_variable_names"]
    assert replay[3]["observed_record_order"] == [r["csv_name"] for r in snapshot["variables"]]
    assert replay[3]["bound_order"] == config["input_columns"]
    assert replay[3]["reordered"] is True


def test_without_explicit_contract_custom_input_order_is_unchanged(case_and_ledger):
    _, ledger = case_and_ledger
    ledger["input_variable_names"].reverse()
    bound, evidence = bind_analyzer_order(ledger)
    assert bound == ledger
    assert evidence["authority"] == "analyzer"
    assert evidence["reordered"] is False
    assert frontend_bindings({"prompt": "Custom analysis"}) == {}


def test_dimensions_remain_attached_to_exact_variable_names(case_and_ledger):
    config, ledger = case_and_ledger
    reference = {record["csv_name"]: record["si_exponents"] for record in ledger["variables"]}
    ledger["variables"].reverse()
    bound, _ = bind_analyzer_order(ledger, config["input_columns"])
    assert {record["csv_name"]: record["si_exponents"] for record in bound["variables"]} == reference
    assert reference["rho"] != reference["mu"]


def test_missing_and_duplicate_names_are_not_normalized_away(case_and_ledger):
    config, ledger = case_and_ledger
    missing = deepcopy(ledger)
    missing["variables"] = missing["variables"][:-1]
    with pytest.raises(FrontendContractError):
        bind_analyzer_order(missing, config["input_columns"])
    duplicate = deepcopy(ledger)
    duplicate["input_variable_names"][0] = duplicate["input_variable_names"][1]
    with pytest.raises(FrontendContractError):
        bind_analyzer_order(duplicate, config["input_columns"])
    with pytest.raises(FrontendContractError):
        frontend_bindings({"input_columns": ["rho", "rho"]})
    with pytest.raises(FrontendContractError):
        frontend_bindings({"input_columns": ["rho"], "output_column": "rho"})


def test_provisional_coordinates_reach_existing_validator_unchanged(case_and_ledger):
    config, ledger = case_and_ledger
    ledger["variables"][0]["si_exponents"]["I"] = "null"
    bound, _ = bind_analyzer_order(ledger, config["input_columns"])
    assert bound["variables"][0]["si_exponents"]["I"] == "null"
    columns = list(pd.read_csv(ROOT / config["dataset"], nrows=0).columns)
    compiled = dimensions.compile_ledger(bound, columns, expected_inputs=config["input_columns"],
                                         null_policy="provisional_zero")
    assert compiled["ledger"]["variables"][0]["si_exponents"]["I"] == "0"
    assert ledger["variables"][0]["si_exponents"]["I"] == "null"


def test_binding_preserves_the_existing_exact_nullspace(case_and_ledger):
    config, ledger = case_and_ledger
    compiled, basis, _, _ = compile_geometry(config, ledger)
    rows = compiled["contract"]["D_in"]
    actual = dimensions.calc_basis_tool.invoke({"D_in": list(reversed(rows)),
                                               "var_names": config["input_columns"]})
    np.testing.assert_array_equal(actual["basis_matrices"], basis)
    np.testing.assert_array_equal(basis, [[1, -1, 1, 1, 0], [0, 0, -1, 0, 1]])


def test_final_validation_seed_does_not_override_gradient_seed(case_and_ledger):
    config, _ = case_and_ledger
    plan = schemas.WorkflowPlan(template="auto_mechanism_adaptive", rationale="Offline contract test")
    first = workflow.compile_workflow_plan(plan, 2, {**config["manual_hyperparams"], "gpr_random_state": 123})
    second = workflow.compile_workflow_plan(plan, 2, {**config["manual_hyperparams"], "gpr_random_state": 987})
    assert first["resolved_gradient_regime_options"] == second["resolved_gradient_regime_options"]
    assert first["resolved_gradient_regime_options"]["surrogate"]["random_state"] == 42
    assert first["resolved_gradient_regime_options"]["clustering"]["random_state"] == 42


@pytest.mark.parametrize("configured", [True, False])
def test_cli_passes_only_explicit_role_contracts_to_graph(tmp_path, configured, monkeypatch):
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b,target\n1,2,3\n", encoding="utf-8")
    config = {"case_id": "custom", "dataset": str(dataset), "prompt": "Offline payload test"}
    if configured:
        config.update(input_columns=["b", "a"], output_column="target")
    original_config = deepcopy(config)
    captured = {}

    class CaptureGraph:
        def stream(self, payload, **kwargs):
            captured.update(deepcopy(payload))
            yield "values", {**payload, "workflow_status": "success"}

    state = cli.run_analysis(config, output_dir=tmp_path / "run", initialize=False,
                             display_cards=False, agent_graph=CaptureGraph())
    assert config == original_config
    if configured:
        assert captured["frontend_expected_inputs"] == ["b", "a"]
        assert captured["frontend_expected_output"] == "target"
        assert state["frontend_expected_inputs"] == ["b", "a"]
    else:
        assert "frontend_expected_inputs" not in captured
        assert "frontend_expected_output" not in captured


def test_real_analyzer_node_retains_raw_reply_and_binding_in_graph_state(monkeypatch, case_and_ledger):
    config, ledger = case_and_ledger
    shuffled = deepcopy(ledger)
    shuffled["input_variable_names"].reverse()
    shuffled["variables"] = shuffled["variables"][1:] + shuffled["variables"][:1]
    raw_snapshot = deepcopy(shuffled)
    monkeypatch.setattr(dimensions._models, "analyzer_llm", SimpleNamespace(
        invoke=lambda messages: deepcopy(shuffled)))
    monkeypatch.setattr(dimensions, "_bind_analyzer_output", lambda response, *args: response)
    for name in ("display_frontend_stage", "display_agent_run_header", "display_frontend_details", "display_analysis_result"):
        monkeypatch.setattr(dimensions._presentation, name, lambda *args, **kwargs: None)
    state = {"messages": [], "prompt_problem": config["prompt"],
             "csv_FilePath": str(ROOT / config["dataset"]),
             "manual_hyperparams": {"show_workflow_progress": False}, **frontend_bindings(config)}
    builder = StateGraph(schemas.AdaptiveWorkflowState)
    builder.add_node("analysis", dimensions.analyze_problem_node)
    builder.add_edge(START, "analysis")
    builder.add_edge("analysis", END)
    result = builder.compile().invoke(state)
    assert result["analyzer_raw_ledger"] == raw_snapshot
    assert result["dimension_ledger"]["input_variable_names"] == config["input_columns"]
    assert result["input_order_binding"]["authority"] == "explicit_case_contract"
    assert result["input_order_binding"]["observed_input_order"] == raw_snapshot["input_variable_names"]
    assert result["input_order_binding"]["bound_order"] == config["input_columns"]
    assert shuffled == raw_snapshot
    hints = get_type_hints(schemas.AdaptiveWorkflowState)
    assert {"analyzer_raw_ledger", "input_order_binding"} <= hints.keys()
