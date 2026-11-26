"""Verify archived requests, model responses and embedding response metadata."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

RAW_FILENAMES = {
    "attempt.json",
    "request.json",
    "response.json",
    "response_envelope.json",
    "pre_send_auth_observation.json",
}
MESSAGE_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _provenance(record):
    return record.get("provenance") or (record.get("observation") or {}).get("provenance") or {}


def _record_list(records):
    if isinstance(records, dict):
        if "provenance" in records or "observation" in records:
            return [records]
        return list(records.values())
    return list(records)


def _model_calls(records):
    calls = {}
    for record in records:
        if "capture_status" in record or "origin" in record:
            for visit, call in enumerate(record.get("calls") or [], 1):
                calls[("Analyzer", visit)] = call
        if record.get("call") and "bound_output" in record:
            calls[("Output_Binding_Checker", 1)] = record["call"]
        for visit, call in enumerate(record.get("revision_calls") or [], 2):
            calls[("Analyzer", visit)] = call
        for visit, round_ in enumerate(record.get("checker_rounds") or [], 1):
            calls[("Hallu_Checker", visit)] = round_.get("checker_call") or {}
        for visit, call in enumerate((record.get("column_verification") or {}).get("calls") or [], 1):
            calls[("Output_Binding_Checker", visit)] = call
    return calls


def _usage_fields(response):
    usage = response.get("usage") or {}
    input_details = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
    output_details = usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": input_details.get("cached_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
    }


def _verify_embedding_metadata(response, envelope):
    """Check retained metadata without claiming to reconstruct the omitted body."""
    if envelope.get("body_omitted") != "query_embedding_vector" or "body_text" in envelope:
        raise ValueError("Embedding response must explicitly omit its original vector-bearing body")
    if type(envelope.get("body_byte_length")) is not int or envelope["body_byte_length"] <= 0:
        raise ValueError("Embedding response must retain the original body byte count")
    data = response["data"]
    if (response.get("object") != "list" or len(data) != 1
            or data[0] != {"index": 0, "object": "embedding"}):
        raise ValueError("Expected metadata for one embedding query, without its vector")


def _verify_call(call, request, response):
    messages = [
        {"role": MESSAGE_ROLES[message["type"]], "content": message["content"]}
        for message in call["model_messages"]
    ]
    if messages != request["messages"]:
        raise ValueError("Raw request messages differ from the captured model call")
    choices = response["choices"]
    if len(choices) != 1:
        raise ValueError("Expected one recorded LLM choice")
    tools = choices[0]["message"].get("tool_calls") or []
    if tools != (call.get("raw_tool_calls") or []):
        raise ValueError("Raw response tool calls differ from the captured model call")
    if call.get("parsed_response") is not None:
        if len(tools) != 1 or json.loads(tools[0]["function"]["arguments"]) != call["parsed_response"]:
            raise ValueError("Decoded raw tool arguments differ from the captured model result")


def load_trial_evidence(trial_directory, records):
    """Return retained-evidence counts and validation issues.

    Repeated provenance snapshots are deduplicated by HTTP attempt ID. Failed
    transport attempts retain their accounting through the supplied audit file.
    Embedding responses retain metadata and historical body digests only; their
    original response-body hashes cannot be recomputed after vector omission.
    A trial without a request archive returns available=False.
    """
    trial = Path(trial_directory)
    request_root = trial / "requests"
    result = {
        "available": request_root.is_dir(),
        "successful_attempts": 0,
        "full_response_bodies": 0,
        "embedding_metadata_only_responses": 0,
        "transport_attempts": 0,
        "duplicate_receipts": 0,
        "issues": [],
    }
    if not result["available"]:
        return result
    rows = _record_list(records)
    receipts = {}
    for record in rows:
        for receipt in _provenance(record).get("http_attempts") or []:
            identifier = receipt["attempt_id"]
            if identifier in receipts:
                result["duplicate_receipts"] += 1
                if receipt != receipts[identifier]:
                    result["issues"].append(f"{identifier}: conflicting copies of HTTP receipt")
            else:
                receipts[identifier] = receipt
    model_calls = _model_calls(rows)
    used = Counter()
    for folder in sorted(request_root.iterdir()):
        identifier = folder.name
        used[identifier] += 1
        result["successful_attempts"] += 1
        try:
            if not folder.is_dir() or {path.name for path in folder.iterdir()} != RAW_FILENAMES:
                raise ValueError("Raw request directory has an incomplete or unexpected file set")
            attempt = _read_json(folder / "attempt.json")
            request = _read_json(folder / "request.json")
            response = _read_json(folder / "response.json")
            envelope = _read_json(folder / "response_envelope.json")
            observation = _read_json(folder / "pre_send_auth_observation.json")
            if attempt["attempt_id"] != identifier or attempt != receipts.get(identifier):
                raise ValueError("Raw attempt identity differs from the scientific receipt")
            if attempt["outcome"] != "http_success":
                raise ValueError("Raw success directory does not contain a successful HTTP attempt")
            if sha256(request["body_text"].encode("utf-8")).hexdigest() != attempt["request_sha256"]:
                raise ValueError("Request body checksum differs from the scientific receipt")
            body = json.loads(request["body_text"])
            if body != request["body"] or body["model"] != attempt["requested_model"]:
                raise ValueError("Request body or model identity is inconsistent")
            digest = envelope["body_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Historical response checksum is invalid")
            if envelope["body_sha256"] != attempt["response_sha256"]:
                raise ValueError("Response checksum differs from the scientific receipt")
            if attempt["kind"] == "embedding":
                _verify_embedding_metadata(response, envelope)
                result["embedding_metadata_only_responses"] += 1
            else:
                if sha256(envelope["body_text"].encode("utf-8")).hexdigest() != digest:
                    raise ValueError("Response envelope checksum is inconsistent")
                if json.loads(envelope["body_text"]) != response:
                    raise ValueError("Response JSON differs from its captured envelope")
                result["full_response_bodies"] += 1
            if response.get("model") != attempt["response_model"]:
                raise ValueError("Response model differs from the scientific receipt")
            if any(attempt.get(key) != value for key, value in _usage_fields(response).items()):
                raise ValueError("Reported usage differs from the raw response")
            if (
                observation["attempt_id"] != identifier
                or observation["request_sha256"] != attempt["request_sha256"]
                or observation["model"] != attempt["requested_model"]
            ):
                raise ValueError("Pre-send observation is linked to a different request")
            if attempt["kind"] == "embedding":
                if (attempt["node"], attempt["requested_model"]) != ("Hallu_Checker", "GLM-Embedding-3"):
                    raise ValueError("Embedding request has an unexpected workflow role or model")
                query = body["input"]
                if not isinstance(query, str):
                    raise ValueError("Expected an exact single-query string")
            else:
                call_key = (attempt["node"], attempt["node_visit"])
                _verify_call(model_calls[call_key], body, response)
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["issues"].append(f"{identifier}: {type(error).__name__}: {error}")
    accounting_file = trial / "audit" / "retry_accounting_only.json"
    if accounting_file.is_file():
        for attempt in _read_json(accounting_file)["attempts"]:
            identifier = attempt["attempt_id"]
            used[identifier] += 1
            result["transport_attempts"] += 1
            receipt = receipts.get(identifier) or {}
            fields = ("node", "node_visit", "kind", "duration_seconds", "request_sha256")
            if receipt.get("outcome") != "transport_error" or any(
                attempt.get(field) != receipt.get(field) for field in fields
            ):
                result["issues"].append(f"{identifier}: retry accounting differs from HTTP receipt")
    if set(used) != set(receipts):
        missing = sorted(set(receipts) - set(used))
        extra = sorted(set(used) - set(receipts))
        result["issues"].append(f"Request coverage differs from HTTP receipts: missing={missing}, extra={extra}")
    repeated = sorted(identifier for identifier, count in used.items() if count != 1)
    if repeated:
        result["issues"].append(f"HTTP evidence must account for each attempt once: {repeated}")
    return result
