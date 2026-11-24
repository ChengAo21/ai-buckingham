"""Provenance extracted from the author implementation."""
from __future__ import annotations
from pathlib import Path
from . import config as _config
from . import presentation as _presentation
from . import graph as _graph

import os
import json
import hashlib
import warnings
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import importlib.metadata
import threading
import time
import uuid
from openai import DefaultHttpxClient
from . import models as _models


_RUN_RECORD_RELEASE = "ai-buckingham"


_RUN_ENGINE_SHA256 = hashlib.sha256(b"".join(path.name.encode("utf-8") + b"\0" + path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()


_LAST_RUN_PROVENANCE = None


_LAST_RUN_ID = None


_RUN_LEDGERS = {}


_REGISTRY_LOCK = threading.RLock()


_ACTIVE_OBSERVATION = ContextVar("agent_run_observation", default=None)


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_json(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _warning(ledger, where, error):
    try:
        item = {"where": where, "error_type": type(error).__name__}
        with ledger.lock:
            if item not in ledger.data["observation_warnings"]:
                ledger.data["observation_warnings"].append(item)
    except BaseException:
        pass


def _metadata(state):
    started_utc = _utc()
    warnings = []
    dataset_path = str(state.get("csv_FilePath") or "")
    digest = None
    if dataset_path:
        try:
            hasher = hashlib.sha256()
            with open(dataset_path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        except BaseException as error:
            warnings.append({"where": "dataset_hash", "error_type": type(error).__name__})
    packages = {}
    for name in ("numpy", "pandas", "scipy", "sympy", "scikit-learn", "cma", "pydantic", "openai", "httpx", "langchain-core", "langchain-openai", "langgraph"):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:
            packages[name] = None
    return {"run_id": str(uuid.uuid4()), "release": globals().get("_RUN_RECORD_RELEASE", _config.AGENT_VERSION),
            "engine_sha256": globals().get("_RUN_ENGINE_SHA256", globals().get("_SCIENTIFIC_CORE_FINGERPRINT")),
            "started_utc": started_utc, "finished_utc": None, "status": "running",
            "dataset": {"name": os.path.basename(dataset_path), "sha256": digest},
            "problem_sha256": hashlib.sha256(str(state.get("prompt_problem") or "").encode("utf-8")).hexdigest(),
            "manual_hyperparams_sha256": _hash_json(state.get("manual_hyperparams") or {}),
            "model_configuration": deepcopy(_models._MODEL_ROLE_CONFIGURATION), "package_versions": packages,
            "tracing_enabled": str(os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false"))).strip().lower() in {"true", "1", "yes", "on"},
            "wall_seconds": 0.0, "node_timings": [], "http_attempts": [], "observation_warnings": warnings}


def _totals(data):
    attempts = data["http_attempts"]
    result = {"http_attempt_count": len(attempts),
              "llm_http_attempt_count": sum(x.get("kind") == "llm" for x in attempts),
              "embedding_http_attempt_count": sum(x.get("kind") == "embedding" for x in attempts),
              "usage_missing_attempt_count": sum(not x.get("usage_reported", False) for x in attempts),
              "http_duration_seconds_sum": sum(x.get("duration_seconds") or 0.0 for x in attempts),
              "node_duration_seconds_sum": sum(x.get("duration_seconds") or 0.0 for x in data["node_timings"])}
    for outcome in ("http_success", "http_error", "transport_error"):
        result[outcome + "_count"] = sum(x.get("outcome") == outcome for x in attempts)
    for key in _TOKEN_FIELDS:
        result["known_" + key] = sum(x[key] for x in attempts if x.get(key) is not None)
        result[key + "_missing_count"] = sum(x.get(key) is None for x in attempts)
    return result


class _RunObservationLedger:
    def __init__(self, state, restored=None):
        self.clock_start = time.perf_counter()
        self.lock = threading.RLock()
        self.data = deepcopy(restored) if restored is not None else _metadata(state)
        self.data.setdefault("observation_warnings", [])
        self.data.setdefault("node_timings", [])
        self.data.setdefault("http_attempts", [])
        self.prior_wall = float(self.data.get("wall_seconds") or 0.0) if restored is not None else 0.0
        self.stopped_wall = None
        self.displayed = False
        self.visits = {}
        for item in self.data["node_timings"]:
            self.visits[item.get("node")] = max(self.visits.get(item.get("node"), 0), item.get("visit", 0))
        if restored is not None:
            self.data["observation_warnings"].append({"where": "restored_provenance", "error_type": "WallTimeExcludesUnobservedProcessGap"})
        self.data["status"] = "running"
        self.data["finished_utc"] = None

    def snapshot(self):
        with self.lock:
            value = deepcopy(self.data)
            value["wall_seconds"] = self.stopped_wall if self.stopped_wall is not None else self.prior_wall + time.perf_counter() - self.clock_start
            value["totals"] = _totals(value)
            return value

    def finish(self, status):
        with self.lock:
            self.data["status"] = status
            self.data["finished_utc"] = _utc()
            self.stopped_wall = self.prior_wall + time.perf_counter() - self.clock_start


def _ledger_for_state(state):
    incoming = state.get("run_provenance") if isinstance(state, dict) else None
    run_id = incoming.get("run_id") if isinstance(incoming, dict) else None
    with _REGISTRY_LOCK:
        if run_id and run_id in _RUN_LEDGERS:
            return _RUN_LEDGERS[run_id]
        ledger = _RunObservationLedger(state, incoming if run_id else None)
        _RUN_LEDGERS[ledger.data["run_id"]] = ledger
        return ledger


def get_run_provenance(state=None):
    """Return an independent snapshot; completed observations only, never guesses."""
    if isinstance(state, dict):
        value = state.get("run_provenance", state if "http_attempts" in state else None)
        if isinstance(value, dict):
            return deepcopy(value)
        return None
    with _REGISTRY_LOCK:
        ledger = _RUN_LEDGERS.get(_LAST_RUN_ID)
    if ledger is not None:
        return ledger.snapshot()
    return deepcopy(_LAST_RUN_PROVENANCE)


def _publish(ledger, terminal=False):
    global _LAST_RUN_PROVENANCE, _LAST_RUN_ID
    try:
        snapshot = ledger.snapshot()
        with _REGISTRY_LOCK:
            _LAST_RUN_ID = snapshot["run_id"]
            _LAST_RUN_PROVENANCE = deepcopy(snapshot)
        if terminal:
            with ledger.lock:
                show = not ledger.displayed
                ledger.displayed = True
            renderer = _presentation.display_run_provenance
            if show and callable(renderer):
                try:
                    renderer(deepcopy(snapshot))
                except BaseException as error:
                    _warning(ledger, "provenance_display", error)
        return snapshot
    except BaseException as error:
        _warning(ledger, "provenance_snapshot", error)
        return None


def observe_workflow_node(name, function):
    """Add observation metadata only; preserve scientific values and exceptions."""
    @wraps(function)
    def observed(state, *args, **kwargs):
        ledger = None
        token = None
        started = time.perf_counter()
        try:
            ledger = _ledger_for_state(state)
            with ledger.lock:
                visit = ledger.visits.get(name, 0) + 1
                ledger.visits[name] = visit
            token = _ACTIVE_OBSERVATION.set((ledger, name, visit))
            _publish(ledger)
        except BaseException:
            ledger = None
        try:
            result = function(state, *args, **kwargs)
        except BaseException as error:
            if ledger is not None:
                try:
                    with ledger.lock:
                        ledger.data["node_timings"].append({"node": name, "visit": visit, "duration_seconds": time.perf_counter() - started, "status": "failed", "error_type": type(error).__name__})
                    ledger.finish("failed")
                    _publish(ledger, terminal=True)
                except BaseException as observation_error:
                    _warning(ledger, "failed_node_observation", observation_error)
            raise
        else:
            if ledger is not None:
                try:
                    with ledger.lock:
                        ledger.data["node_timings"].append({"node": name, "visit": visit, "duration_seconds": time.perf_counter() - started, "status": "returned", "error_type": None})
                    merged = {**state, **result} if isinstance(result, dict) else dict(state)
                    with ledger.lock:
                        ledger.data["workflow_status"] = merged.get("workflow_status")
                        workflow = merged.get("workflow_result")
                        ledger.data["scientific_status"] = workflow.get("status") if isinstance(workflow, dict) else None
                    terminal_status = None
                    if name == "Visualizer": terminal_status = "completed"
                    elif name == "Hallu_Checker" and merged.get("workflow_status") == "terminal_unresolved": terminal_status = "terminal_unresolved"
                    elif name == "Basis_Executor":
                        route = _graph.route_after_basis
                        if callable(route) and route(merged) == "failed": terminal_status = "basis_failed"
                    if terminal_status: ledger.finish(terminal_status)
                    snapshot = _publish(ledger, terminal=terminal_status is not None)
                    if isinstance(result, dict) and snapshot is not None:
                        return {**result, "run_provenance": snapshot}
                except BaseException as error:
                    _warning(ledger, "returned_node_observation", error)
            return result
        finally:
            if token is not None:
                try: _ACTIVE_OBSERVATION.reset(token)
                except BaseException: pass
    return observed


def _number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def record_http_send(native_send, request, *args, **kwargs):
    ledger = None
    try:
        context = _ACTIVE_OBSERVATION.get()
    except BaseException:
        context = None
    if context is None:
        return native_send(request, *args, **kwargs)
    try:
        ledger, node, visit = context
        started = time.perf_counter()
        record = {"attempt_id": str(uuid.uuid4()), "node": node, "node_visit": visit,
                  "kind": "embedding" if str(request.url.path).rstrip("/").endswith("/embeddings") else "llm",
                  "requested_model": None, "response_model": None, "http_status": None,
                  "outcome": None, "duration_seconds": None, "started_utc": _utc(),
                  **{key: None for key in _TOKEN_FIELDS}, "usage_reported": False,
                  "sdk_retry_index": None, "error_type": None, "request_sha256": None, "response_id": None}
    except BaseException as error:
        if ledger is not None:
            _warning(ledger, "http_observation_initialization", error)
        return native_send(request, *args, **kwargs)
    try:
        content = request.content
        record["request_sha256"] = hashlib.sha256(content).hexdigest()
        try:
            payload = json.loads(content) if content else {}
            record["requested_model"] = payload.get("model") if isinstance(payload, dict) else None
        except (ValueError, UnicodeError):
            pass
        retry = request.headers.get("x-stainless-retry-count")
        record["sdk_retry_index"] = int(retry) if retry is not None else None
    except BaseException as error:
        _warning(ledger, "request_observation", error)
    try:
        response = native_send(request, *args, **kwargs)
    except BaseException as error:
        if record is not None:
            try:
                record.update(outcome="transport_error", error_type=type(error).__name__, duration_seconds=time.perf_counter() - started)
                with ledger.lock: ledger.data["http_attempts"].append(record)
            except BaseException as observation_error:
                _warning(ledger, "transport_error_observation", observation_error)
        raise
    else:
        if record is not None:
            try:
                record.update(http_status=int(response.status_code), outcome="http_success" if 200 <= response.status_code < 300 else "http_error",
                              duration_seconds=time.perf_counter() - started, response_id=response.headers.get("x-request-id"))
                # Client.send has already buffered a non-stream response. Do
                # not read a streamed response just to obtain usage metadata.
                if response.is_stream_consumed:
                    try:
                        body = response.json()
                    except (ValueError, UnicodeError):
                        body = {}
                    if isinstance(body, dict):
                        record["response_model"] = body.get("model")
                        record["response_id"] = body.get("id") or record["response_id"]
                        usage = body.get("usage") or {}
                        if isinstance(usage, dict):
                            record["input_tokens"] = _number(usage.get("input_tokens", usage.get("prompt_tokens")))
                            record["output_tokens"] = _number(usage.get("output_tokens", usage.get("completion_tokens")))
                            record["total_tokens"] = _number(usage.get("total_tokens"))
                            inputs = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
                            outputs = usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
                            record["cached_input_tokens"] = _number(inputs.get("cached_tokens")) if isinstance(inputs, dict) else None
                            record["reasoning_tokens"] = _number(outputs.get("reasoning_tokens")) if isinstance(outputs, dict) else None
                            record["usage_reported"] = any(record[key] is not None for key in _TOKEN_FIELDS)
            except BaseException as error:
                _warning(ledger, "response_observation", error)
            finally:
                try:
                    with ledger.lock: ledger.data["http_attempts"].append(record)
                except BaseException as error:
                    _warning(ledger, "response_record_append", error)
        return response


class RecordedHttpClient(DefaultHttpxClient):
    """Native default client, with observation around its unchanged send method."""
    def send(self, request, *args, **kwargs):
        return record_http_send(super().send, request, *args, **kwargs)


def attach_http_recording(client):
    """Observe an existing native/proxy client without replacing its policies."""
    if client is None or isinstance(client, RecordedHttpClient):
        return client
    try:
        native_send = client.send
        if getattr(native_send, "_provenance_recording", False):
            return client
        @wraps(native_send)
        def observed_send(request, *args, **kwargs):
            return record_http_send(native_send, request, *args, **kwargs)
        observed_send._provenance_recording = True
        client.send = observed_send
    except BaseException:
        pass
    return client
