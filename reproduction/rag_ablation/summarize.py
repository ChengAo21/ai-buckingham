"""Recompute the benchmark tables and archive audit without model calls."""

from collections import Counter, defaultdict
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
import csv
import json
import math


ROOT = Path(__file__).resolve().parent
MODELS = ("GPT-4.1-mini", "GPT-5-mini", "GPT-4o-mini")
VARIANTS = ("original", "challenge")
AXES = {"M", "L", "T", "I", "Theta", "N", "J"}
LLM_ROLES = (
    "initial_analyzer",
    "first_checker",
    "analyzer_revision",
    "second_checker",
    "output_binding_checker",
)
ROLE_BY_ATTEMPT = {
    ("Analyzer", 1, "llm"): "initial_analyzer",
    ("Hallu_Checker", 1, "llm"): "first_checker",
    ("Analyzer", 2, "llm"): "analyzer_revision",
    ("Hallu_Checker", 2, "llm"): "second_checker",
    ("Output_Binding_Checker", 1, "llm"): "output_binding_checker",
    ("Hallu_Checker", 1, "embedding"): "embedding",
    ("Hallu_Checker", 2, "embedding"): "embedding",
}
TABLE2_ROLES = {
    "Initial Analyzer (tokens)": "initial_analyzer",
    "First Checker (tokens)": "first_checker",
    "Analyzer revision (tokens)": "analyzer_revision",
    "Second Checker (tokens)": "second_checker",
    "Embedding (tokens)": "embedding",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def archive_hash_check(recorded, content):
    if not recorded:
        return None
    if recorded == sha256(content).hexdigest():
        return "match"
    return "mismatch"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_schedule(gold):
    schedule = read_json(ROOT / "reference" / "schedule.json")
    expected_keys = {
        f"{case}_{variant}_{number:02d}"
        for case in gold
        for variant in VARIANTS
        for number in range(1, 11)
    }
    if len(schedule) != len(expected_keys) or {r["pair_id"] for r in schedule} != expected_keys:
        raise ValueError("Schedule must contain ten repetitions per case and prompt variant")
    dataset_check = {}
    prompts = defaultdict(set)
    for row in schedule:
        case, variant, _ = row["pair_id"].split("_")
        if (case, variant) != (row["case"], row["prompt_version"]):
            raise ValueError(f"Inconsistent schedule entry: {row['pair_id']}")
        prompts[(case, variant)].add(row["prompt"])
        dataset = row["dataset"]
        if dataset not in dataset_check:
            path = ROOT / "reference" / "Data_csv" / dataset
            with path.open(encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle))
            dataset_check[dataset] = (file_hash(path), set(columns))
        digest, columns = dataset_check[dataset]
        target = gold[case]
        if row["dataset_sha256"] != digest or not {target["output"], *target["inputs"]} <= columns:
            raise ValueError(f"Reference dataset or gold columns disagree with schedule: {dataset}")
        if any(set(dimensions) != AXES for dimensions in target["inputs"].values()):
            raise ValueError(f"Gold SI dimensions are incomplete: {case}")
    for case in gold:
        original, challenge = (prompts[(case, variant)] for variant in VARIANTS)
        if len(original) != 1 or len(challenge) != 1 or original == challenge:
            raise ValueError(f"Prompt variants are inconsistent: {case}")
    return sorted(schedule, key=lambda row: row["pair_id"])


def same_inputs(ledger, gold):
    """Require each input exactly once and all seven SI exponents explicitly."""
    if not ledger:
        return False
    expected = gold["inputs"]
    names = ledger.get("input_variable_names") or []
    if set(names) != set(expected) or len(names) != len(expected):
        return False
    variables = ledger.get("variables") or []
    if {variable["csv_name"] for variable in variables} != set(expected) or len(variables) != len(expected):
        return False
    for variable in variables:
        coordinates = variable.get("si_exponents") or {}
        if set(coordinates) != AXES:
            return False
        try:
            actual = {axis: Fraction(str(value)) for axis, value in coordinates.items()}
        except (ValueError, ZeroDivisionError):
            return False
        if actual != expected[variable["csv_name"]]:
            return False
    return True


def provenance_of(record):
    return record.get("provenance") or (record.get("observation") or {}).get("provenance") or {}


def collect_attempts(records):
    unique = {}
    duplicate_count = 0
    for record in records:
        for attempt in provenance_of(record).get("http_attempts", []):
            attempt_id = attempt["attempt_id"]
            if attempt_id in unique:
                if attempt != unique[attempt_id]:
                    raise ValueError(f"Conflicting duplicate API attempt: {attempt_id}")
                duplicate_count += 1
                continue
            unique[attempt_id] = attempt
    return list(unique.values()), duplicate_count


def api_metrics(attempts):
    grouped = defaultdict(list)
    for attempt in attempts:
        key = (attempt["node"], attempt["node_visit"], attempt["kind"])
        grouped[ROLE_BY_ATTEMPT[key]].append(attempt)
    totals = {}
    for role in (*LLM_ROLES, "embedding", "first_embedding", "second_embedding"):
        if role in ("first_embedding", "second_embedding"):
            visit = 1 if role == "first_embedding" else 2
            calls = [attempt for attempt in grouped["embedding"] if attempt["node_visit"] == visit]
        else:
            calls = grouped[role]
        reported = [attempt["total_tokens"] for attempt in calls if attempt["total_tokens"] is not None]
        durations = [attempt["duration_seconds"] for attempt in calls if attempt["duration_seconds"] is not None]
        totals[f"{role}_known_tokens"] = sum(reported)
        totals[f"{role}_attempts"] = len(calls)
        totals[f"{role}_attempts_missing_tokens"] = len(calls) - len(reported)
        totals[f"{role}_api_seconds"] = math.fsum(durations)
        totals[f"{role}_attempts_missing_duration"] = len(calls) - len(durations)
    totals["llm_known_tokens"] = sum(totals[f"{role}_known_tokens"] for role in LLM_ROLES)
    totals["analyzer_and_checker_known_tokens"] = (
        totals["llm_known_tokens"] - totals["output_binding_checker_known_tokens"]
    )
    totals["all_api_attempts"] = len(attempts)
    totals["all_api_seconds"] = math.fsum(
        totals[f"{role}_api_seconds"] for role in (*LLM_ROLES, "embedding")
    )
    totals["api_seconds_excluding_output_binding"] = (
        totals["all_api_seconds"] - totals["output_binding_checker_api_seconds"]
    )
    totals["all_api_attempts_missing_tokens"] = sum(
        totals[f"{role}_attempts_missing_tokens"] for role in (*LLM_ROLES, "embedding")
    )
    totals["all_api_attempts_missing_duration"] = sum(
        totals[f"{role}_attempts_missing_duration"] for role in (*LLM_ROLES, "embedding")
    )
    totals["transport_errors"] = sum(attempt["outcome"] == "transport_error" for attempt in attempts)
    return totals


def classify(run, target):
    rounds = run["checker_rounds"]
    statuses = [rnd["gate_status"] for rnd in rounds]
    if len(statuses) > 2 or len(run["revision_calls"]) > 1:
        raise ValueError("Found a trial with more than one Analyzer revision")
    final = run.get("final_ledger")
    checked = rounds[-1]["candidate_ledger"] if rounds else None
    evaluated = final or checked
    exact_inputs = same_inputs(evaluated, target) if evaluated is not None else None
    correct_output = evaluated.get("output_variable_name") == target["output"] if evaluated else None
    basis = run.get("basis") or {}
    basis_ready = run["basis_built"] and run["workflow_status"] == "basis_ready"
    if basis_ready:
        if not final or final != checked or statuses not in (["PASSED"], ["RETRY", "PASSED"]):
            raise ValueError(f"No matching passed Checker ledger for ready basis: {run['pair_id']}")
        if basis.get("input_vars") != final["input_variable_names"] or len(run["revision_calls"]) != len(statuses) - 1:
            raise ValueError(f"Basis or revision log disagrees with checked ledger: {run['pair_id']}")
        pipeline_result = "direct" if len(statuses) == 1 else "recovered"
        correct_output = correct_output and basis.get("output_var") == target["output"]
    else:
        pipeline_result = "failed"
    validated_result = pipeline_result if exact_inputs and correct_output else "failed"
    return {
        "pipeline_result": pipeline_result,
        "validated_result": validated_result,
        "workflow_status": run["workflow_status"],
        "first_gate": statuses[0] if statuses else None,
        "second_gate": statuses[1] if len(statuses) > 1 else None,
        "revision_attempted": bool(run["revision_calls"]),
        "evaluated_ledger_exact_inputs": exact_inputs,
        "evaluated_ledger_correct_output": correct_output,
        "evaluated_ledger_output": evaluated.get("output_variable_name") if evaluated else None,
        "basis_output": basis.get("output_var"),
        "expected_output": target["output"],
        "first_checker_exact_inputs": same_inputs(rounds[0]["candidate_ledger"], target) if rounds else None,
        "first_checker_output": rounds[0]["candidate_ledger"].get("output_variable_name") if rounds else None,
    }


def summarize_trial(model, scheduled, target):
    pair_id = scheduled["pair_id"]
    trial = ROOT / "trials" / model / pair_id
    files = {path.name for path in trial.glob("*.json")}
    if files not in ({"initial.json", "on.json"}, {"initial.json", "on.json", "binding.json"}):
        raise ValueError(f"Unexpected archive contents for {model}/{pair_id}: {sorted(files)}")
    initial_path = trial / "initial.json"
    initial = read_json(initial_path)
    run = read_json(trial / "on.json")
    binding = read_json(trial / "binding.json") if "binding.json" in files else {}
    if run["pair_id"] != pair_id or (binding and binding["pair_id"] != pair_id):
        raise ValueError(f"Mismatched trial identifier: {model}/{pair_id}")
    records = {"initial": initial, "on": run, "binding": binding}
    prompt_bytes = scheduled["prompt"].encode("utf-8")
    initial_messages = [
        message["content"]
        for call in initial.get("calls", [])
        for message in call.get("model_messages", [])
        if message["type"] == "human"
    ]
    prompt_prefix = f"Problem statement:\n{scheduled['prompt']}\n\nCSV columns:"
    if any(not message.startswith(prompt_prefix) for message in initial_messages):
        raise ValueError(f"Recorded initial request text disagrees with schedule: {model}/{pair_id}")
    prompt_checks = {}
    for name, record in records.items():
        provenance = provenance_of(record)
        prompt_hash = provenance.get("problem_sha256") or record.get("prompt_sha256")
        dataset_hash = (provenance.get("dataset") or {}).get("sha256") or record.get("dataset_sha256")
        if dataset_hash and dataset_hash != scheduled["dataset_sha256"]:
            raise ValueError(f"Dataset provenance disagrees with schedule: {model}/{pair_id}/{name}.json")
        prompt_checks[f"{name}_prompt_sha256_check"] = archive_hash_check(prompt_hash, prompt_bytes)
    attempts, duplicate_attempts = collect_attempts(records.values())
    for attempt in attempts:
        role = ROLE_BY_ATTEMPT[(attempt["node"], attempt["node_visit"], attempt["kind"])]
        expected_model = {
            "initial_analyzer": model.lower(), "analyzer_revision": model.lower(), "embedding": "GLM-Embedding-3",
        }.get(role, "gpt-5-mini")
        if attempt["requested_model"] != expected_model:
            raise ValueError(f"Unexpected model for {role}: {model}/{pair_id}")
    if not any(
        (attempt["node"], attempt["node_visit"], attempt["kind"]) == ("Analyzer", 1, "llm")
        and attempt["outcome"] == "http_success"
        for attempt in attempts
    ):
        raise ValueError(f"No initial Analyzer API response was captured: {model}/{pair_id}")
    usage = api_metrics(attempts)
    if usage["output_binding_checker_attempts"] == 0 and run["workflow_status"] != "initial_analyzer_failed":
        raise ValueError(f"No output binding API attempt was captured: {model}/{pair_id}")
    archived_hash = binding.get("initial_sha256") or run.get("initial_sha256")
    initial_ledger_hash = initial.get("ledger_sha256")
    return {
        "case": scheduled["case"],
        "analyzer": model,
        "prompt_variant": scheduled["prompt_version"],
        "pair_id": pair_id,
        **classify(run, target),
        "binding_or_run_initial_file_sha256_check": archive_hash_check(
            archived_hash, initial_path.read_bytes()
        ),
        "initial_ledger_sha256_matches_checker_input": (
            initial_ledger_hash == run.get("initial_ledger_sha256") if initial_ledger_hash else None
        ),
        "initial_request_text_archived": bool(initial_messages),
        **prompt_checks,
        "duplicate_http_records": duplicate_attempts,
        **usage,
    }


def score_cell(rows, field):
    counts = Counter(row[field] for row in rows)
    direct, recovery = (counts[key] for key in ("direct", "recovered"))
    return f"{direct}+{recovery}={direct + recovery}/{len(rows)}"


def score_rows(by_cell, cases, field):
    for case in (*cases, "Total"):
        yield {
            "Case": case,
            **{
                f"{model} {variant}": score_cell(by_cell[(case, model, variant)], field)
                for model in MODELS
                for variant in VARIANTS
            },
        }


def combine_api_metrics(rows):
    fields = api_metrics([])
    result = {}
    for field in fields:
        values = [row[field] for row in rows]
        result[field] = math.fsum(values) if field.endswith("_seconds") else sum(values)
    result["trials"] = len(rows)
    result["mean_api_seconds_per_trial"] = result["all_api_seconds"] / len(rows)
    return result


def usage_rows(by_cell, cases):
    for case in cases:
        for model in MODELS:
            merged = [row for variant in VARIANTS for row in by_cell[(case, model, variant)]]
            stats = combine_api_metrics(merged)
            yield {
                "Case": case,
                "Analyzer": model,
                **{heading: stats[f"{role}_known_tokens"] for heading, role in TABLE2_ROLES.items()},
                "LLM total (known tokens)": stats["analyzer_and_checker_known_tokens"],
                "All API seconds": round(stats["all_api_seconds"], 3),
                "Mean API seconds per trial": round(stats["mean_api_seconds_per_trial"], 3),
            }


def full_precision_usage_rows(by_cell, cases):
    for case in (*cases, "Total"):
        for model in MODELS:
            rows = [row for variant in VARIANTS for row in by_cell[(case, model, variant)]]
            stats = combine_api_metrics(rows)
            yield {
                "case": case,
                "analyzer": model,
                **stats,
                **{
                    f"{role}_mean_known_tokens": stats[f"{role}_known_tokens"] / len(rows)
                    for role in (*LLM_ROLES, "embedding", "first_embedding", "second_embedding")
                },
                "analyzer_and_checker_mean_known_tokens": stats["analyzer_and_checker_known_tokens"] / len(rows),
                "all_llm_mean_known_tokens": stats["llm_known_tokens"] / len(rows),
            }


def display_usage_rows(by_cell, cases):
    labels = {
        "Init": "initial_analyzer", "Emb1": "first_embedding", "Chk1": "first_checker",
        "Rev": "analyzer_revision", "Emb2": "second_embedding", "Chk2": "second_checker",
    }
    for case in cases:
        for model in ("GPT-4o-mini", "GPT-4.1-mini", "GPT-5-mini"):
            rows = [row for variant in VARIANTS for row in by_cell[(case, model, variant)]]
            stats = combine_api_metrics(rows)
            denominator = Decimal(len(rows)) * 1000
            values = {
                label: str((Decimal(stats[f"{role}_known_tokens"]) / denominator).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                ))
                for label, role in labels.items()
            }
            subtotal = Decimal(stats["analyzer_and_checker_known_tokens"]) / denominator
            api = Decimal(str(stats["all_api_seconds"])) / Decimal(len(rows))
            yield {
                "Case": case, "Analyzer": model, **values,
                "Sum": str(subtotal.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
                "API": str(api.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
            }


def provenance_checks(rows):
    return {
        "mismatch": [
            {"analyzer": row["analyzer"], "pair_id": row["pair_id"], "fields": fields}
            for row in rows
            if (fields := [key for key, value in row.items() if key.endswith("_sha256_check") and value == "mismatch"])
        ]
    }


def main():
    gold = read_json(ROOT / "reference" / "expected_dimensions.json")
    schedule = load_schedule(gold)
    trials_root = ROOT / "trials"
    if {p.name for p in trials_root.iterdir() if p.is_dir()} != set(MODELS):
        raise ValueError("Archived trial folders do not cover the three benchmark Analyzers")
    rows = []
    by_cell = defaultdict(list)
    expected_pairs = {scheduled["pair_id"] for scheduled in schedule}
    for model in MODELS:
        found_pairs = {p.name for p in (trials_root / model).iterdir() if p.is_dir()}
        if found_pairs != expected_pairs:
            raise ValueError(
                f"{model}: missing {sorted(expected_pairs - found_pairs)}, "
                f"extra {sorted(found_pairs - expected_pairs)}"
            )
        for scheduled in schedule:
            case, variant = scheduled["case"], scheduled["prompt_version"]
            row = summarize_trial(model, scheduled, gold[case])
            rows.append(row)
            by_cell[(case, model, variant)].append(row)
            by_cell[("Total", model, variant)].append(row)
    cases = tuple(gold)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    write_csv(output / "trial_audit.csv", rows)
    write_csv(output / "table1_success.csv", list(score_rows(by_cell, cases, "validated_result")))
    write_csv(output / "recorded_pipeline_outcomes.csv", list(score_rows(by_cell, cases, "pipeline_result")))
    write_csv(output / "table2_api_usage.csv", list(usage_rows(by_cell, cases)))
    write_csv(output / "api_usage_by_variant.csv", [
        {
            "case": case,
            "analyzer": model,
            "prompt_variant": variant,
            **combine_api_metrics(by_cell[(case, model, variant)]),
        }
        for case in (*cases, "Total")
        for model in MODELS
        for variant in VARIANTS
    ])
    write_csv(output / "table30_display.csv", list(display_usage_rows(by_cell, cases)))
    write_csv(output / "api_usage_full_precision.csv", list(full_precision_usage_rows(by_cell, cases)))
    model_aggregates = {}
    for model in MODELS:
        selected = [row for row in rows if row["analyzer"] == model]
        counts = Counter(row["validated_result"] for row in selected)
        stats = combine_api_metrics(selected)
        revision_tokens = stats["analyzer_revision_known_tokens"] + stats["second_checker_known_tokens"]
        model_aggregates[model] = {
            "trials": len(selected),
            "initial_successes": counts["direct"],
            "recoveries": counts["recovered"],
            "final_successes": counts["direct"] + counts["recovered"],
            "initial_success_percent": 100 * counts["direct"] / len(selected),
            "final_success_percent": 100 * (counts["direct"] + counts["recovered"]) / len(selected),
            "success_gain_percentage_points": 100 * counts["recovered"] / len(selected),
            "revision_and_second_checker_tokens": revision_tokens,
            "four_stage_tokens": stats["analyzer_and_checker_known_tokens"],
            "revision_token_share_percent": 100 * revision_tokens / stats["analyzer_and_checker_known_tokens"],
            "mean_api_seconds_per_trial": stats["mean_api_seconds_per_trial"],
        }
    (output / "model_aggregates.json").write_text(
        json.dumps(model_aggregates, indent=2) + "\n", encoding="utf-8",
    )
    summary = {
        "scheduled_trials": len(schedule) * len(MODELS),
        "archived_trials": len(rows),
        "trials_with_initial_request_text": sum(row["initial_request_text_archived"] for row in rows),
        "trials_without_initial_request_text": sum(not row["initial_request_text_archived"] for row in rows),
        "validated_counts": dict(Counter(row["validated_result"] for row in rows)),
        "recorded_pipeline_counts": dict(Counter(row["pipeline_result"] for row in rows)),
        "api": combine_api_metrics(rows),
        "duplicate_attempt_records_removed": sum(row["duplicate_http_records"] for row in rows),
        "pipeline_passes_excluded": [
            {"analyzer": row["analyzer"], "pair_id": row["pair_id"]}
            for row in rows
            if row["pipeline_result"] != row["validated_result"]
        ],
        "provenance_sha256_checks": provenance_checks(rows),
        "initial_ledger_sha256_changes": [
            {"analyzer": row["analyzer"], "pair_id": row["pair_id"]}
            for row in rows
            if row["initial_ledger_sha256_matches_checker_input"] is False
        ],
    }
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote tables and audit for {len(rows)} scheduled trials to {output}")


if __name__ == "__main__":
    main()
