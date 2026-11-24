"""Observe actual optimizer generations without reevaluating the objective.

Use ``with record_run(run_dir): ...`` around a graph invocation or stream.
JSONL is appended and flushed after every generation, so an interrupted run
still contains the generations that really completed.  Objective values include
the original feasibility and collinearity penalties; they are not MI values.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .paths import _RUN_OUTPUT


_TRACE_SINK = ContextVar("ai_buckingham_trace_sink", default=None)
_TRACE_SCOPE = ContextVar("ai_buckingham_trace_scope", default={})


def _json_value(value):
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


class OptimizationTrace:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "evidence" / "optimization_trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self.warnings = []
        self.records = 0

    def write(self, record):
        with self._lock:
            self._stream.write(json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False) + "\n")
            self._stream.flush()
            self.records += 1

    def close(self):
        self._stream.close()
        if self.warnings:
            (self.run_dir / "evidence" / "optimization_trace_warnings.json").write_text(
                json.dumps(self.warnings, indent=2) + "\n", encoding="utf-8"
            )


@contextmanager
def record_run(run_dir):
    """Select this run's output directory and append-only optimizer observer."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sink = OptimizationTrace(run_dir)
    output_token = _RUN_OUTPUT.set(run_dir)
    sink_token = _TRACE_SINK.set(sink)
    scope_token = _TRACE_SCOPE.set({})
    try:
        yield sink
    finally:
        _TRACE_SCOPE.reset(scope_token)
        _TRACE_SINK.reset(sink_token)
        _RUN_OUTPUT.reset(output_token)
        sink.close()


def trace_scope(phase):
    """Attach scope and candidate identities without changing call arguments."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def observed(*args, **kwargs):
            if _TRACE_SINK.get() is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            context = dict(_TRACE_SCOPE.get())
            if "scope_id" in bound.arguments:
                context["scope_id"] = str(bound.arguments["scope_id"])
            if phase == "restart_batch":
                context["search_id"] = str(uuid.uuid4())
                context["candidate"] = f"k={bound.arguments.get('pi_count')}"
            else:
                context["phase"] = phase
            token = _TRACE_SCOPE.set(context)
            try:
                return function(*args, **kwargs)
            finally:
                _TRACE_SCOPE.reset(token)

        return observed
    return decorate


def _observe(factory):
    sink = _TRACE_SINK.get()
    if sink is None:
        return
    try:
        sink.write({"recorded_at_unix": time.time(), **_TRACE_SCOPE.get(), **factory()})
    except Exception as error:
        # A recording failure must not change scientific decisions or RNG state.
        warning = {"error_type": type(error).__name__, "where": "optimization_trace"}
        if warning not in sink.warnings:
            sink.warnings.append(warning)


def record_cma_generation(es, solutions, function_values, *, seed, pi_count,
                          optimize_output, reconstruct, basis, input_vars):
    """Read the population already evaluated and the state after ``es.tell``."""
    def snapshot():
        import numpy as np
        values = np.asarray(function_values, dtype=float)
        best_index = int(np.argmin(values))
        in_vectors, out_vector = reconstruct(es.mean)
        return {
            "event": "optimizer_generation", "optimizer": "CMA-ES",
            "generation": int(es.countiter), "seed": int(seed),
            "pi_count": int(pi_count), "optimize_output": bool(optimize_output),
            "best_objective": float(es.result.fbest),
            "generation_best_objective": float(values[best_index]),
            "mean_objective": float(np.mean(values)),
            "sigma": float(es.sigma), "evaluations": int(es.countevals),
            "mean_coefficients": np.asarray(es.mean, dtype=float).tolist(),
            "best_coefficients": np.asarray(es.result.xbest, dtype=float).tolist(),
            "generation_best_coefficients": np.asarray(solutions[best_index], dtype=float).tolist(),
            "mean_input_basis_coefficients": [np.asarray(v).tolist() for v in in_vectors],
            "mean_output_basis_coefficients": np.asarray(out_vector).tolist(),
            "basis_matrix": np.asarray(basis).tolist(), "input_variables": list(input_vars),
            "objective_definition": "negative MI for feasible candidates; original penalties otherwise",
        }
    _observe(snapshot)


def record_optimizer_end(optimizer, parameters, objective, *, seed, pi_count, optimize_output, status=None):
    """Record CMA, scalar and direct-evaluation endpoints without inventing generations."""
    _observe(lambda: {
        "event": "optimizer_end", "optimizer": optimizer,
        "generation": None, "seed": int(seed), "pi_count": int(pi_count),
        "optimize_output": bool(optimize_output), "best_objective": float(objective),
        "status": status or ("success" if math.isfinite(float(objective)) and float(objective) < 1e4 else "failed"),
        "best_coefficients": parameters,
    })


def record_regime_candidates(candidates, common, decision):
    """Archive existing candidate arrays without reclustering or changing state."""
    sink = _TRACE_SINK.get()
    if sink is None:
        return
    try:
        import numpy as np
        folder = sink.run_dir / "evidence" / "regime_candidates" / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        arrays = {key: np.asarray(common[key]) for key in
                  ("feature_coordinates", "surrogate_gradients", "schmidt_basis", "valid_indices") if key in common}
        np.savez_compressed(folder / "coordinates.npz", **arrays)
        metadata = {"mode": "observed_during_execution", "candidate_counts": [],
                    "decision": {key: value for key, value in decision.items() if key != "selected_candidate"},
                    "row_mapping": "valid_indices refer to cleaned numerical input; use original_csv_indices from workflow results for CSV alignment"}
        for candidate in candidates:
            k = int(candidate["k"])
            metadata["candidate_counts"].append(k)
            np.savez_compressed(folder / f"k_{k}.npz", labels=np.asarray(candidate["labels"], dtype=int),
                                assignment_confidence=np.asarray(candidate["assignment_confidence"], dtype=float))
            details = {key: value for key, value in candidate.items() if key not in {"labels", "assignment_confidence"}}
            (folder / f"k_{k}.json").write_text(json.dumps(_json_value(details), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(_json_value(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as error:
        warning = {"error_type": type(error).__name__, "where": "regime_candidates"}
        if warning not in sink.warnings:
            sink.warnings.append(warning)
