"""Scientific invariants of the extracted engine and its evidence observers.

These tests need neither credentials nor the author's original monolithic file.
They run a small real CMA search, while keeping publication run budgets intact.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ai_buckingham import config, knowledge, optimization, recording


def _numpy_rng_state():
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2:]


def _assert_numpy_rng_equal(first, second):
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_graph_builds_without_credentials_or_network():
    """A fresh reader process can inspect the graph before initializing clients."""
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    for name in ("OPENAI_API_KEY", "PARATERA_API_KEY", "LANGSMITH_API_KEY"):
        environment.pop(name, None)
    environment.update({"PYTHONPATH": str(project / "src"), "MPLBACKEND": "Agg",
                        "PYTHONDONTWRITEBYTECODE": "1"})
    script = """
import httpx, socket
def forbidden(*args, **kwargs):
    raise AssertionError('Graph construction attempted network access')
httpx.Client.send = forbidden
socket.create_connection = forbidden
from ai_buckingham import build_agent_graph, models
graph = build_agent_graph()
assert models.analyzer_llm is None
assert models.checker_llm is None
assert models.planner_llm is None
assert models.interpretation_llm is None
assert {'Analyzer', 'Hallu_Checker', 'Workflow_Executor', 'Visualizer'} <= set(graph.get_graph().nodes)
print('offline graph construction passed')
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=project,
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "offline graph construction passed" in completed.stdout


def test_real_cma_observation_preserves_result_rng_and_mi_evaluations(monkeypatch, tmp_path):
    """Recording observes real evaluated generations, without extra fitting."""
    rng = np.random.default_rng(123)
    features = np.exp(rng.normal(size=(40, 2)))
    response = (features[:, 0] ** 2 * features[:, 1]).reshape(-1, 1)
    hyperparameters = {"cma_maxiter": 3, "cma_popsize": 6,
                       "enable_gpr_validation": False, "enable_score_uq": False,
                       "show_workflow_progress": False, "enable_workflow_visualization": False}
    arguments = dict(X=features, Y=response, basis_matrices=np.eye(2),
                     input_vars=["a", "b"], pi_num_input=1, optimize_output=False,
                     fixed_in_power=None, fixed_out_power=None,
                     mi_method="kraskov", mi_param=3, hp=hyperparameters, seed=42)
    native_mi = optimization.global_calc_MI
    calls = []

    def counted_mi(*args, **kwargs):
        calls.append(1)
        return native_mi(*args, **kwargs)

    monkeypatch.setattr(optimization, "global_calc_MI", counted_mi)
    random.seed(1234)
    np.random.seed(5678)
    plain = optimization._run_local_cma_once(**arguments)
    plain_count = len(calls)
    plain_python_rng, plain_numpy_rng = random.getstate(), _numpy_rng_state()

    calls.clear()
    random.seed(1234)
    np.random.seed(5678)
    with recording.record_run(tmp_path / "observed") as trace:
        observed = optimization._run_local_cma_once(**arguments)
    observed_python_rng, observed_numpy_rng = random.getstate(), _numpy_rng_state()

    assert plain["status"] == observed["status"] == "success"
    assert plain == observed
    assert len(calls) == plain_count
    assert plain_python_rng == observed_python_rng
    _assert_numpy_rng_equal(plain_numpy_rng, observed_numpy_rng)
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    generations = [row for row in rows if row["event"] == "optimizer_generation"]
    assert [row["generation"] for row in generations] == [1, 2, 3]
    assert [row["evaluations"] for row in generations] == [6, 12, 18]
    assert generations[-1]["evaluations"] <= plain_count
    assert all(row["seed"] == 42 and row["pi_count"] == 1 for row in generations)
    assert all(row["sigma"] > 0 and len(row["mean_coefficients"]) == 2 for row in generations)
    assert rows[-1]["event"] == "optimizer_end"
    assert rows[-1]["generation"] is None
    assert trace.warnings == []


def _restart_arguments():
    return dict(X=np.ones((4, 2)), Y=np.ones((4, 1)), basis_matrices=np.eye(2),
                input_vars=["a", "b"], pi_count=1, optimize_output=False,
                fixed_in_power=None, fixed_out_power=None, mi_method="kraskov",
                mi_param=3, hp={}, restart_count=3, primary_seed=42)


def _diagnostic_candidate(seed, mi):
    return {"status": "success", "seed": int(seed), "mutual_information": mi,
            "normalized_information_score": float(np.exp(-mi)),
            "input_physical_exponents": [[1.0, 0.0]]}


def test_primary_seed_is_kept_when_a_diagnostic_restart_scores_higher(monkeypatch):
    visited = []

    def candidate(**kwargs):
        seed = int(kwargs["seed"])
        visited.append(seed)
        return _diagnostic_candidate(seed, 1.0 if seed == 42 else 10.0)

    monkeypatch.setattr(optimization, "_run_local_cma_once", candidate)
    result = optimization._run_local_restarts(**_restart_arguments())
    assert visited[0] == 42 and len(visited) == 3
    assert result["selection_mode"] == "primary_seed_restart"
    assert result["primary_seed"] == result["seed"] == 42
    assert result["mutual_information"] == 1.0
    assert max(result["restart_mutual_information_values"]) == 10.0


def test_primary_failure_is_not_replaced_by_a_diagnostic_success(monkeypatch):
    def candidate(**kwargs):
        seed = int(kwargs["seed"])
        if seed == 42:
            return {"status": "optimization_failed", "seed": seed}
        return _diagnostic_candidate(seed, 10.0)

    monkeypatch.setattr(optimization, "_run_local_cma_once", candidate)
    result = optimization._run_local_restarts(**_restart_arguments())
    assert result["status"] == "primary_restart_failed"
    assert result["primary_seed"] == 42
    assert len(result["successful_diagnostic_seeds"]) == 2


def test_published_inputs_are_protected_and_cache_rebuilds_are_private(monkeypatch, tmp_path):
    for path in (config.DATA_DIR / "example.csv", config.KNOWLEDGE_DIR / "indexes" / "index.npz"):
        with pytest.raises(PermissionError):
            knowledge._assert_private_cache_write(path)
    private = tmp_path / "private-cache"
    monkeypatch.setenv("DIMAGENT_CACHE_DIR", str(private))
    published_index = config.KNOWLEDGE_DIR / "indexes" / "physical_quantity_index.npz"
    retriever = SimpleNamespace(embedding_cache_path=published_index,
                               knowledge_base_path=config.KNOWLEDGE_DIR / "dimensional_physical_quantities.md",
                               embedding_model="GLM-Embedding-3", _kb_hash="test-knowledge-hash")
    rebuilt = knowledge._private_rebuild_cache_path(retriever)
    assert rebuilt.is_relative_to(private.resolve())
    assert retriever.embedding_cache_path == rebuilt
    assert retriever.embedding_cache_source == str(published_index.resolve())
    knowledge._assert_private_cache_write(rebuilt)


def test_candidate_partition_archive_preserves_arrays_and_rng(tmp_path):
    candidate = {"k": 2, "labels": np.array([0, 1], dtype=int),
                 "assignment_confidence": np.array([0.25, 0.75]), "converged": True,
                 "clusters": [{"regime_id": 0}, {"regime_id": 1}]}
    common = {"feature_coordinates": [[0.0, 1.0], [1.0, 0.0]],
              "surrogate_gradients": [[1.0, 0.0], [0.0, 1.0]],
              "schmidt_basis": [[1.0, 0.0], [0.0, 1.0]], "valid_indices": [2, 4]}
    decision = {"status": "success", "best_k": 2, "selected_candidate": candidate}
    original_candidate, original_common = deepcopy(candidate), deepcopy(common)
    python_rng, numpy_rng = random.getstate(), _numpy_rng_state()
    with recording.record_run(tmp_path / "run") as trace:
        recording.record_regime_candidates([candidate], common, decision)
    assert random.getstate() == python_rng
    _assert_numpy_rng_equal(numpy_rng, _numpy_rng_state())
    np.testing.assert_array_equal(candidate["labels"], original_candidate["labels"])
    np.testing.assert_array_equal(candidate["assignment_confidence"], original_candidate["assignment_confidence"])
    assert common == original_common
    manifests = list((trace.run_dir / "evidence" / "regime_candidates").glob("*/manifest.json"))
    assert len(manifests) == 1
    folder = manifests[0].parent
    assert json.loads(manifests[0].read_text())["candidate_counts"] == [2]
    with np.load(folder / "k_2.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["labels"], candidate["labels"])
        np.testing.assert_array_equal(saved["assignment_confidence"], candidate["assignment_confidence"])
    with np.load(folder / "coordinates.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["valid_indices"], common["valid_indices"])
    assert trace.warnings == []


def test_nested_run_paths_are_restored_after_an_exception(tmp_path):
    original = config.OUTPUT_DIR.resolve()
    with recording.record_run(tmp_path / "outer"):
        assert config.OUTPUT_DIR.resolve() == (tmp_path / "outer").resolve()
        with pytest.raises(RuntimeError):
            with recording.record_run(tmp_path / "inner"):
                assert config.OUTPUT_DIR.resolve() == (tmp_path / "inner").resolve()
                raise RuntimeError("interrupted run")
        assert config.OUTPUT_DIR.resolve() == (tmp_path / "outer").resolve()
    assert config.OUTPUT_DIR.resolve() == original
