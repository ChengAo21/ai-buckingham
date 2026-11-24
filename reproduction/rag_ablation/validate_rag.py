"""Replay recorded retrieval evidence and Checker decisions without API calls."""

import csv
import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

import rag
from raw_evidence import load_trial_evidence


class ArchivedScoreRetriever(rag.DimensionalKnowledgeRetriever):
    """Re-rank the archived scores using zero as the lower bound for missing scores."""

    def __init__(self):
        super().__init__(embedding_cache_path=rag.DATA_DIRECTORY / "physical_quantity_index.npz")
        self.documents = self._ensure_embeddings()
        self.row_by_id = {record["entry_id"]: i for i, record in enumerate(self._records)}

    def use_records(self, matches_by_variable):
        self.matches_by_variable = matches_by_variable
        self.query_history = []

    vectors_by_query = None

    def _semantic_scores(self, query):
        current = self.query_history[-1]
        if current["query"] != query:
            raise ValueError("Archived query record differs from the replayed query")
        if self.vectors_by_query is not None:
            vector = np.asarray(self.vectors_by_query[query], dtype=float)
            vector = vector / max(float(np.linalg.norm(vector)), 1.0e-15)
            return np.maximum(self.documents @ vector, 0.0)
        scores = np.zeros(len(self.row_by_id))
        for match in self.matches_by_variable[current["variable_name"]]:
            score = match["semantic_score"]
            if not math.isfinite(score) or not 0.0 <= score <= 1.0 + 1e-12:
                raise ValueError("Archived semantic score lies outside the clipped cosine range")
            scores[self.row_by_id[match["entry_id"]]] = score
        return scores

    def _get_embedding_client(self):
        raise RuntimeError("Query embeddings and document embeddings are never requested in replay")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_schedule(data_directory):
    schedule = {}
    by_dataset = {}
    for row in read_json(data_directory / "schedule.json"):
        dataset_name = row["dataset"]
        if dataset_name not in by_dataset:
            dataset = data_directory / "Data_csv" / dataset_name
            with dataset.open(encoding="utf-8-sig", newline="") as stream:
                columns = next(csv.reader(stream))
            if not columns or len(columns) != len(set(columns)):
                raise ValueError(f"Missing or nonunique CSV header in {dataset_name}")
            by_dataset[dataset_name] = (rag._sha256_file(dataset), columns)
        digest, columns = by_dataset[dataset_name]
        if digest != row["dataset_sha256"]:
            raise ValueError(f"Dataset hash does not match schedule for {dataset_name}")
        pair_id = row["pair_id"]
        if pair_id in schedule:
            raise ValueError(f"Duplicate pair ID {pair_id}")
        schedule[pair_id] = (row["prompt"], columns)
    return schedule


def embedding_http_digests(retrieval):
    digests = {}
    for name, query in retrieval["queries_by_variable"].items():
        # Historical SDK request serialization is compact UTF-8 JSON.
        payload = {"input": query, "model": retrieval["embedding_model"], "encoding_format": "base64"}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digests[name] = hashlib.sha256(encoded).hexdigest()
    return digests


def checker_tool_payload(model_call):
    calls = model_call.get("raw_tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "CheckerDimensionAudit":
        return None
    try:
        return json.loads(calls[0]["function"]["arguments"])
    except (ValueError, TypeError):
        return None


def malformed_checker_payload(payload):
    try:
        rag.CheckerDimensionAudit.model_validate(payload)
    except ValueError:
        return True
    return False


def replay_round(archived_round, prompt, columns, retriever, embedding_request_digests, prior_retries):
    archived = archived_round["retrieval"]
    ledger = rag.validate_ledger(archived_round["candidate_ledger"])
    digest = rag.ledger_sha256(ledger)
    inputs = ledger["input_variable_names"]
    variable_names = [variable["csv_name"] for variable in ledger["variables"]]
    checks = {
        "ledger_hash": digest == archived_round["ledger_sha256"],
        "ledger_columns": inputs == variable_names and len(inputs) == len(set(inputs))
        and set(inputs).issubset(columns) and ledger["output_variable_name"] in columns
        and ledger["output_variable_name"] not in inputs,
    }
    provisional = archived_round["provisional_zero_coordinates"]
    zero_coordinates = {
        (variable["csv_name"], base)
        for variable in ledger["variables"]
        for base, value in variable["si_exponents"].items() if value == "0"
    }
    marked = {(item["csv_name"], item["base"]) for item in provisional}
    checks["provisional_zero_coordinates"] = len(marked) == len(provisional) and marked <= zero_coordinates

    retriever.use_records(archived["matches_by_variable"])
    contexts = rag.render_retrieval_interpretation(ledger)
    replayed = rag.retrieve_dimension_evidence(
        problem_description=prompt,
        input_variable_names=inputs,
        dimension_detect="\n".join(contexts.values()),
        dimension_ledger=ledger,
        retriever=retriever,
    )
    for key in (
        "status", "kb_sha256", "embedding_model", "embedding_provider", "embedding_base_url",
        "retrieval_modes", "retrieved_variables", "n_retrieved_records", "top_k_per_variable",
        "max_evidence_rows", "lexical_weight", "semantic_weight", "min_hybrid_score",
        "allow_lexical_fallback", "matches_by_variable", "evidence_markdown",
        "evidence_records_visible_by_variable", "queries_by_variable", "query_sha256_by_variable",
        "query_observed_in_retriever_by_variable",
    ):
        checks[key] = replayed[key] == archived[key]

    score_error = 0.0
    if retriever.vectors_by_query is not None:
        # Recomputing normalized dot products may change the last floating-point
        # bit across numerical libraries. All record identities/order and text
        # remain exact; only semantic/hybrid scores allow an absolute 1e-12 error.
        records_match = set(replayed["matches_by_variable"]) == set(archived["matches_by_variable"])
        for name, actual_rows in replayed["matches_by_variable"].items():
            expected_rows = archived["matches_by_variable"].get(name, [])
            records_match &= len(actual_rows) == len(expected_rows)
            for actual, expected in zip(actual_rows, expected_rows):
                records_match &= set(actual) == set(expected)
                for key in actual:
                    if key in ("semantic_score", "hybrid_score"):
                        error = abs(actual[key] - expected[key])
                        score_error = max(score_error, error)
                        records_match &= error <= 1e-12
                    else:
                        records_match &= actual[key] == expected[key]
        checks["matches_by_variable"] = records_match

    archived_call = archived_round["checker_call"]
    expected_messages = rag.checker_messages(
        prompt, ledger, digest, provisional, replayed["evidence_markdown"]
    )
    checks["checker_messages"] = archived_call["model_messages"] == expected_messages
    response = archived_round["checker_response"]
    tool_response = checker_tool_payload(archived_call)
    checks["checker_tool_response"] = response == archived_call["parsed_response"]
    if response is None:
        checks["checker_tool_response"] &= malformed_checker_payload(tool_response)
    else:
        checks["checker_tool_response"] &= tool_response == response

    # The gate decision must be replayed against the evidence actually delivered
    # to the historical Checker, even when another audit field differs.
    archived_gate = archived_round["gate"]
    if response is None:
        gate_status = "BLOCKED"
        checks["gate"] = (
            archived_gate["gate_status"] == gate_status
            and archived_gate["ledger_sha256"] == digest
            and len(archived_gate["issues"]) == 1
            and archived_gate["issues"][0].startswith("terminal_unresolved: Checker contract invalid: ")
        )
    else:
        try:
            replayed_gate = rag.gate_checker_audit(ledger, response, archived)
        except (ValueError, SyntaxError) as exc:
            gate_status = "BLOCKED"
            checks["gate"] = (
                archived_gate["gate_status"] == gate_status
                and archived_gate["ledger_sha256"] == digest
                and archived_gate["issues"] == [f"terminal_unresolved: Checker contract invalid: {exc}"]
            )
        else:
            if replayed_gate["gate_status"] == "RETRY" and prior_retries >= 1:
                replayed_gate = {
                    **replayed_gate,
                    "gate_status": "BLOCKED",
                    "retry_budget_exhausted": True,
                    "feedback": (
                        replayed_gate["feedback"]
                        + "\nRevision limit 1 exhausted; terminal_unresolved, no basis execution."
                    ),
                }
            gate_status = replayed_gate["gate_status"]
            checks["gate"] = replayed_gate == archived_gate
    checks["gate_status"] = gate_status == archived_round["gate_status"]
    stored_query_digests = archived["query_sha256_by_variable"]
    replayed_query_digests = replayed["query_sha256_by_variable"]
    digest_mismatches = sorted(
        name for name in stored_query_digests.keys() | replayed_query_digests.keys()
        if stored_query_digests.get(name) != replayed_query_digests.get(name)
    )
    expected_request_digests = embedding_http_digests(replayed)
    checks["embedding_request_sha256"] = set(expected_request_digests.values()) == embedding_request_digests
    request_mismatches = sorted(
        name for name, digest in expected_request_digests.items() if digest not in embedding_request_digests
    )

    return {
        "archived_gate": archived_round["gate_status"],
        "replayed_gate": gate_status,
        "archived_embedding_status": archived["embedding_status"],
        "semantic_replay": "full_query_vectors" if retriever.vectors_by_query is not None else "recorded_scores",
        "maximum_semantic_or_hybrid_score_error": score_error,
        "variables": len(inputs),
        "retrieved_records": archived["n_retrieved_records"],
        "query_digest_mismatch_variables": ",".join(digest_mismatches),
        "embedding_request_digest_mismatch_variables": ",".join(request_mismatches),
        "discrepancies": ",".join(name for name, ok in checks.items() if not ok),
    }


def main():
    # Lexical scoring is unchanged. Memoize pure normalizers shared across queries.
    # These caches have no influence on ranking or evidence generation.
    rag._normalize_text = lru_cache(maxsize=8192)(rag._normalize_text)
    rag._tokenize = lru_cache(maxsize=8192)(rag._tokenize)
    rag._split_multi_value_cell = lru_cache(maxsize=2048)(rag._split_multi_value_cell)
    rag.PARATERA_BASE_URL = rag.PARATERA_DEFAULT_BASE_URL

    schedule = load_schedule(rag.DATA_DIRECTORY)
    retriever = ArchivedScoreRetriever()
    if retriever.documents is None or not np.isfinite(retriever.documents).all():
        raise ValueError("Packaged embedding index is missing, incompatible or non-finite")

    rounds = []
    trials_by_model = Counter()
    trials_without_rounds = []
    checked = set()
    raw_evidence_rows = []
    for trial_path in sorted((rag.PACKAGE_DIR / "trials").glob("*/*/on.json")):
        analyzer, pair_id = trial_path.parent.parent.name, trial_path.parent.name
        trial = read_json(trial_path)
        if trial["pair_id"] != pair_id:
            raise ValueError(f"Trial pair ID does not match its directory for {analyzer}/{pair_id}")
        prompt, columns = schedule[pair_id]
        retriever.vectors_by_query = None
        if (trial_path.parent / "requests").is_dir():
            records = [trial, read_json(trial_path.with_name("initial.json"))]
            binding_path = trial_path.with_name("binding.json")
            if binding_path.is_file():
                records.append(read_json(binding_path))
            evidence = load_trial_evidence(trial_path.parent, records)
            retriever.vectors_by_query = evidence.pop("vectors_by_query")
            raw_evidence_rows.append({"analyzer": analyzer, "pair_id": pair_id, **evidence})
        trials_by_model[analyzer] += 1
        checked.add((analyzer, pair_id))
        if not trial["checker_rounds"]:
            trials_without_rounds.append({"analyzer": analyzer, "pair_id": pair_id, "status": trial["workflow_status"]})
        provenance = trial["provenance"] or {}
        attempts = provenance.get("http_attempts", [])
        prior_retries = 0
        for number, archived_round in enumerate(trial["checker_rounds"], start=1):
            identity = {"analyzer": analyzer, "pair_id": pair_id, "checker_round": number}
            embedding_request_digests = {
                attempt["request_sha256"] for attempt in attempts
                if attempt["kind"] == "embedding" and attempt["node_visit"] == number
            }
            round_checks = replay_round(
                archived_round, prompt, columns, retriever, embedding_request_digests, prior_retries,
            )
            prior_retries += round_checks["replayed_gate"] == "RETRY"
            rounds.append({**identity, **round_checks})

    models = ("GPT-4.1-mini", "GPT-5-mini", "GPT-4o-mini")
    expected = {(analyzer, pair_id) for analyzer in models for pair_id in schedule}
    if checked != expected:
        raise ValueError(f"Trial grid mismatch. Missing: {sorted(expected - checked)}. Extra: {sorted(checked - expected)}.")
    mismatches = [row for row in rounds if row["query_digest_mismatch_variables"]]
    request_mismatches = [row for row in rounds if "embedding_request_sha256" in row["discrepancies"].split(",")]
    discrepant_rounds = sum(bool(row["discrepancies"]) for row in rounds)
    summary = {
        "trials": dict(trials_by_model),
        "trials_without_checker_round": trials_without_rounds,
        "checker_rounds": len(rounds),
        "replayed_gate_status": dict(Counter(row["replayed_gate"] for row in rounds)),
        "archived_embedding_status": dict(Counter(row["archived_embedding_status"] for row in rounds)),
        "rounds_with_replay_discrepancies": discrepant_rounds,
        "discrepancy_fields": dict(Counter(field for row in rounds for field in row["discrepancies"].split(",") if field)),
        "archived_query_digest_mismatches": {
            "rounds": len(mismatches),
            "query_digests": sum(len(row["query_digest_mismatch_variables"].split(",")) for row in mismatches),
            "rounds_by_analyzer": dict(Counter(row["analyzer"] for row in mismatches)),
            "cases": sorted({row["pair_id"].split("_", maxsplit=1)[0] for row in mismatches}),
        },
        "archived_embedding_request_digest_mismatches": {
            "rounds": len(request_mismatches),
            "requests": sum(len(row["embedding_request_digest_mismatch_variables"].split(","))
                            for row in request_mismatches if row["embedding_request_digest_mismatch_variables"]),
            "rounds_by_analyzer": dict(Counter(row["analyzer"] for row in request_mismatches)),
            "cases": sorted({row["pair_id"].split("_", maxsplit=1)[0] for row in request_mismatches}),
        },
        "raw_evidence": {
            "trials": len(raw_evidence_rows),
            "successful_attempts": sum(row["successful_attempts"] for row in raw_evidence_rows),
            "transport_attempts": sum(row["transport_attempts"] for row in raw_evidence_rows),
            "trials_with_issues": sum(bool(row["issues"]) for row in raw_evidence_rows),
        },
        "semantic_replay_rounds": dict(Counter(row["semantic_replay"] for row in rounds)),
        "maximum_semantic_or_hybrid_score_error": max(
            row["maximum_semantic_or_hybrid_score_error"] for row in rounds
        ),
        "semantic_score_absolute_tolerance": 1e-12,
        "document_embedding_index": {
            "kb_sha256": retriever.kb_hash,
            "index_sha256": rag._sha256_file(Path(retriever.embedding_cache_path)),
            "record_count": retriever.documents.shape[0],
            "embedding_dimension": retriever.documents.shape[1],
        },
        "semantic_replay": (
            "For trials with archived raw embedding responses, replay recomputes all 247 cosine "
            "similarities from the original query vector and document index, then verifies the full "
            "retrieval ranking. Other trials retain the recorded top-k semantic scores and assign "
            "a zero lower bound to unreturned records. Both paths verify lexical scores, hybrid "
            "arithmetic, evidence, Checker messages and gate decisions."
        ),
    }
    output = rag.PACKAGE_DIR / "results"
    output.mkdir(exist_ok=True)
    with (output / "rag_replay_rounds.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rounds[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rounds)
    (output / "rag_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "raw_evidence_audit.json").write_text(
        json.dumps(raw_evidence_rows, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(f"Replayed {len(rounds)} Checker rounds from {sum(trials_by_model.values())} trials")
    print(f"Rounds with replay discrepancies: {discrepant_rounds}")
    print(f"Rounds with archived query digest discrepancies: {len(mismatches)}")
    print(f"Rounds with archived embedding request digest discrepancies: {len(request_mismatches)}")
    return int(bool(discrepant_rounds) or any(row["issues"] for row in raw_evidence_rows))


if __name__ == "__main__":
    raise SystemExit(main())
