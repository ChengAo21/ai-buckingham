"""Check immutable historical evidence and reported experiment denominators offline."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = ROOT / "reproduction"
RAG = REPRODUCTION / "rag_ablation"
MODELS = {"GPT-4.1-mini", "GPT-5-mini", "GPT-4o-mini"}
HISTORICAL_CASES = {
    "MHD", "PRESGRAD", "RAYLEIGH", "KEYHOLE", "BLASIUS", "SANDGRAIN",
    "VELOTRANS", "GOLDENFELD", "CESS", "CIRCLEPIPE",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def trials():
    return read_csv(RAG / "results" / "trial_audit.csv")


def test_published_integrity_entry_point_works_outside_repository_without_changing_snapshots(tmp_path):
    sentinels = [RAG / "results" / "table1_success.csv", RAG / "SHA256SUMS",
                 REPRODUCTION / "regime_comparison" / "results" / "regional_memberships.npz",
                 REPRODUCTION / "velocity_comparison" / "results" / "transformed_native_profiles.csv"]
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in sentinels}
    output = tmp_path / "verification.json"
    completed = subprocess.run(
        [sys.executable, "-B", str(REPRODUCTION / "verify_integrity.py"), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = read_json(output)
    assert report["status"] == "passed"
    assert report["counts_by_archive"]["rag_ablation"] == 4640
    assert report["counts_by_archive"]["case_inputs"] == 12
    assert report["counts_by_archive"]["regime_comparison"] > 0
    assert report["counts_by_archive"]["velocity_comparison"] > 0
    assert report["files_verified"] == sum(report["counts_by_archive"].values())
    assert report["scientific_calculations_repeated"] is False
    assert before == {path: hashlib.sha256(path.read_bytes()).digest() for path in sentinels}


def test_trial_audit_retains_all_scheduled_cells_and_failures(trials):
    coverage = read_json(RAG / "results" / "coverage.json")
    assert len(trials) == coverage["scheduled_trials"] == coverage["archived_trials"] == 600
    assert {row["case"] for row in trials} == HISTORICAL_CASES
    assert {row["analyzer"] for row in trials} == MODELS
    assert {row["prompt_variant"] for row in trials} == {"original", "challenge"}
    cells = Counter((row["case"], row["analyzer"], row["prompt_variant"]) for row in trials)
    assert len(cells) == 10 * 3 * 2 and set(cells.values()) == {10}
    assert len({(row["analyzer"], row["pair_id"]) for row in trials}) == 600
    assert dict(Counter(row["validated_result"] for row in trials)) == coverage["validated_counts"]
    assert dict(Counter(row["pipeline_result"] for row in trials)) == coverage["recorded_pipeline_counts"]
    assert coverage["validated_counts"]["failed"] > 0
    for row in trials:
        assert (RAG / "trials" / row["analyzer"] / row["pair_id"] / "initial.json").is_file()


def test_reference_scoring_preserves_the_wrong_output_exclusion(trials):
    excluded = [row for row in trials if row["pipeline_result"] in {"direct", "recovered"}
                and row["validated_result"] == "failed"]
    assert len(excluded) == 1
    row = excluded[0]
    assert (row["analyzer"], row["pair_id"]) == ("GPT-4.1-mini", "MHD_challenge_02")
    assert row["evaluated_ledger_correct_output"] == "False"
    assert row["evaluated_ledger_output"] == "u" and row["expected_output"] == "u*"


def test_checker_summary_distinguishes_vector_replay_from_recorded_scores():
    summary = read_json(RAG / "results" / "rag_replay_summary.json")
    rounds = read_csv(RAG / "results" / "rag_replay_rounds.csv")
    assert len(rounds) == summary["checker_rounds"] == 730
    assert summary["trials"] == dict.fromkeys(MODELS, 200)
    assert summary["semantic_replay_rounds"] == {"recorded_scores": 670, "full_query_vectors": 60}
    assert dict(Counter(row["semantic_replay"] for row in rounds)) == summary["semantic_replay_rounds"]
    assert dict(Counter(row["replayed_gate"] for row in rounds)) == summary["replayed_gate_status"]
    assert summary["rounds_with_replay_discrepancies"] == 0
    assert summary["maximum_semantic_or_hybrid_score_error"] <= summary["semantic_score_absolute_tolerance"]
    for row in rounds:
        assert row["archived_gate"] == row["replayed_gate"]
        assert not row["discrepancies"]
        assert not row["query_digest_mismatch_variables"]
        assert not row["embedding_request_digest_mismatch_variables"]


def test_success_table_denominators_and_recoveries_match_the_trial_audit(trials):
    table = read_csv(RAG / "results" / "table1_success.csv")
    assert {row["Case"] for row in table} == HISTORICAL_CASES | {"Total"}
    expected_columns = {f"{model} {variant}" for model in MODELS for variant in ("original", "challenge")}
    assert set(table[0]) == expected_columns | {"Case"}
    for row in table:
        for column in expected_columns:
            model, variant = column.rsplit(" ", 1)
            selected = [trial for trial in trials if trial["analyzer"] == model
                        and trial["prompt_variant"] == variant
                        and (row["Case"] == "Total" or trial["case"] == row["Case"])]
            match = re.fullmatch(r"(\d+)\+(\d+)=(\d+)/(\d+)", row[column])
            assert match, row[column]
            direct, recovered, success, denominator = map(int, match.groups())
            assert denominator == len(selected) == (100 if row["Case"] == "Total" else 10)
            assert direct == sum(trial["validated_result"] == "direct" for trial in selected)
            assert recovered == sum(trial["validated_result"] == "recovered" for trial in selected)
            assert success == direct + recovered <= denominator


def test_api_usage_tables_cover_all_thirty_model_case_pairs_and_preserve_units(trials):
    table = read_csv(RAG / "results" / "table2_api_usage.csv")
    display = read_csv(RAG / "results" / "table30_display.csv")
    expected_pairs = {(case, model) for case in HISTORICAL_CASES for model in MODELS}
    assert len(table) == len(display) == 30
    assert {(row["Case"], row["Analyzer"]) for row in table} == expected_pairs
    assert {(row["Case"], row["Analyzer"]) for row in display} == expected_pairs
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial["case"], trial["analyzer"])].append(trial)
    token_columns = {
        "Initial Analyzer (tokens)": "initial_analyzer_known_tokens",
        "First Checker (tokens)": "first_checker_known_tokens",
        "Analyzer revision (tokens)": "analyzer_revision_known_tokens",
        "Second Checker (tokens)": "second_checker_known_tokens",
        "Embedding (tokens)": "embedding_known_tokens",
        "LLM total (known tokens)": "analyzer_and_checker_known_tokens",
    }
    for row in table:
        selected = grouped[(row["Case"], row["Analyzer"])]
        assert len(selected) == 20
        for column, field in token_columns.items():
            assert int(row[column]) == sum(int(trial[field]) for trial in selected)
        seconds = sum(float(trial["all_api_seconds"]) for trial in selected)
        assert float(row["All API seconds"]) == pytest.approx(seconds, abs=0.00051)
        assert float(row["Mean API seconds per trial"]) == pytest.approx(seconds / 20, abs=0.00051)
    for row in display:
        assert set(row) == {"Case", "Analyzer", "Init", "Emb1", "Chk1", "Rev", "Emb2", "Chk2", "Sum", "API"}
        assert float(row["Sum"]) == pytest.approx(sum(float(row[key]) for key in ("Init", "Chk1", "Rev", "Chk2")), abs=0.025)
