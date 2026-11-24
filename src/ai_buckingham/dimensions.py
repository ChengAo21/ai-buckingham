"""Dimensions extracted from the author implementation."""
from __future__ import annotations

import re
import json
import hashlib
import pandas as pd
import numpy as np
from typing import List, Any, Optional
import sympy
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import ast
import math
import sympy as sp
import time
from copy import deepcopy
from . import bindings as _bindings
from . import config as _config
from . import knowledge as _knowledge
from . import models as _models
from . import presentation as _presentation
from . import provenance as _provenance
from . import schemas as _schemas
from . import workflow as _workflow


def normalize_analyzer_quoted_nulls(payload):
    """Accept the exact quoted null sentinel only in Analyzer SI coordinates."""
    payload = _schemas.model_dict(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("variables"), list):
        return payload
    variables = []
    for record in payload["variables"]:
        exponents = record.get("si_exponents") if isinstance(record, dict) else None
        if isinstance(exponents, dict):
            exponents = {
                base: None if base in _config.BASE_DIMENSIONS and value == "null" else value
                for base, value in exponents.items()
            }
            record = {**record, "si_exponents": exponents}
        variables.append(record)
    return {**payload, "variables": variables}


def ledger_sha256(value):
    ledger = _schemas.validate_ledger(value)
    return hashlib.sha256(json.dumps(ledger, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def uses_provisional_zero(null_policy):
    if not isinstance(null_policy, str) or null_policy not in ("strict", "provisional_zero"):
        raise _schemas.FrontendContractError("frontend_null_policy must be 'strict' or 'provisional_zero'")
    return null_policy == "provisional_zero"


def compile_ledger(payload, csv_columns, expected_inputs=None, expected_output=None, *, null_policy="strict"):
    provisional = uses_provisional_zero(null_policy)
    if provisional:
        payload = normalize_analyzer_quoted_nulls(payload)
    ledger = _schemas.validate_ledger(payload)
    columns = list(csv_columns)
    names = ledger["input_variable_names"]
    variable_names = [v["csv_name"] for v in ledger["variables"]]
    output = ledger["output_variable_name"]
    if len(columns) != len(set(columns)):
        raise _schemas.FrontendContractError("CSV has duplicate column names")
    if len(names) != len(set(names)) or len(variable_names) != len(set(variable_names)):
        raise _schemas.FrontendContractError("Duplicate input name/variable record")
    if output in names or output in variable_names:
        # The separately declared output is authoritative for this role cleanup.
        names = [name for name in names if name != output]
        ledger["input_variable_names"] = names
        ledger["variables"] = [record for record in ledger["variables"] if record["csv_name"] != output]
        variable_names = [record["csv_name"] for record in ledger["variables"]]
    if not names:
        raise _schemas.FrontendContractError("At least one input must remain after excluding the output variable")
    if set(names) != set(variable_names):
        raise _schemas.FrontendContractError("Variable records must cover exactly the explicitly ordered input names")
    if output not in columns or any(name not in columns for name in names):
        raise _schemas.FrontendContractError("Every selected name must be an exact existing CSV column")
    if expected_output is not None and output != expected_output:
        raise _schemas.FrontendContractError("Output contradicts explicit manual role override")
    if expected_inputs is not None and names != list(expected_inputs):
        raise _schemas.FrontendContractError("Input order/set contradicts explicit manual role contract")
    by_name = {v["csv_name"]: v for v in ledger["variables"]}
    ledger["variables"] = [by_name[name] for name in names]
    unknown = [{"csv_name": v["csv_name"], "base": base}
               for v in ledger["variables"] for base in _config.BASE_DIMENSIONS if v["si_exponents"][base] is None]
    provisional_zeroes = []
    if provisional and unknown:
        provisional_zeroes = unknown
        for item in provisional_zeroes:
            by_name[item["csv_name"]]["si_exponents"][item["base"]] = "0"
        unknown = []
    digest = ledger_sha256(ledger)
    descriptions = "\n".join(f"{v['csv_name']}: {v['quantity_name']}; {v['physical_definition']}; named SI={json.dumps(v['si_exponents'], ensure_ascii=False)}" for v in ledger["variables"])
    contract = {"ledger_sha256": digest, "status": "unknown" if unknown else "complete", "var_names": names,
                "output_var": output, "n": len(names), "r": None, "k": None, "row_order": [], "D_exact": None, "D_in": None, "unknown_coordinates": unknown,
                "physical_correctness_scope": "Structure/rank only; physical assignments require audit and evidence."}
    if provisional_zeroes:
        contract["provisional_zero_coordinates"] = provisional_zeroes
    if not unknown:
        matrix = sp.Matrix([[sp.Rational(by_name[name]["si_exponents"][base]) for name in names] for base in _config.BASE_DIMENSIONS])
        active = [i for i in range(len(_config.BASE_DIMENSIONS)) if any(matrix[i, j] != 0 for j in range(len(names)))] or [0]
        D = matrix[active, :]
        numeric = [[float(x) for x in row] for row in D.tolist()]
        for exact_row, numeric_row in zip(D.tolist(), numeric):
            for value, converted in zip(exact_row, numeric_row):
                if not math.isfinite(converted) or (value != 0 and converted == 0):
                    raise _schemas.FrontendContractError("Exact SI coordinate overflows/underflows the unchanged basis-tool numeric interface")
                if sp.nsimplify(converted) != value:
                    raise _schemas.FrontendContractError("Original basis-tool nsimplify cannot preserve this exact SI rational coordinate")
        # Preserve exact rank at the float/decimal interface used by the existing
        # tool. Refuse a numeric rank change instead of silently changing nullity.
        numeric_exact = sp.Matrix([[sp.Rational(str(x)) for x in row] for row in numeric])
        if numeric_exact.rank() != D.rank():
            raise _schemas.FrontendContractError("Numeric tool conversion changes exact SI rank")
        contract.update(row_order=[_config.BASE_DIMENSIONS[i] for i in active], D_exact=[[str(x) for x in row] for row in D.tolist()],
                        D_in=numeric, r=int(D.rank()), k=len(names)-int(D.rank()))
        display_matrix = "Rows = " + str(contract["row_order"]) + "\nColumns = " + str(names) + "\n" + "\n".join(f"{base}: {row}" for base, row in zip(contract["row_order"], contract["D_exact"]))
        counts = f"n={contract['n']}, exact rank r={contract['r']}, k=n-r={contract['k']}"
    else:
        display_matrix = "UNRESOLVED: unknown named SI coordinates; no numeric matrix compiled."
        counts = f"n={len(names)}; r and k unresolved until every SI coordinate is supplied."
    return {"ledger": ledger, "ledger_sha256": digest, "contract": contract,
            "legacy_fields": {"analyzed_output_var": output, "analyzed_output_reasoning": ledger["output_variable_reasoning"],
                              "analyzed_output_confidence": ledger["output_variable_confidence"], "analyzed_input_vars": names,
                              "dimension_detect": descriptions, "dimension_mat": display_matrix, "dimension_numK": counts}}


def render_retrieval_interpretation(ledger):
    data = _schemas.validate_ledger(ledger)
    return {v["csv_name"]: f"exact_column: {v['csv_name']}; quantity_name: {v['quantity_name']}; physical_definition: {v['physical_definition']}" for v in data["variables"]}


def format_evidence_records(matches, max_rows=30):
    lines, visible = [], {name: [] for name in matches}
    count = 0
    for name, records in matches.items():
        lines.append(f"### Input variable `{name}`")
        if not records:
            lines.append("No sufficiently relevant record found; absence alone is not a contradiction.")
        for record in records:
            if count >= max_rows:
                break
            count += 1
            visible[name].append(str(record.get("entry_id", "")))
            lines.append("- " + "; ".join(f"{key}: {record.get(key, '')}" for key in (
                "entry_id", "canonical_quantity", "symbols", "aliases", "definition", "si_dimension", "dimension_vector", "si_unit", "domain", "ambiguity_notes", "source", "source_identifier", "confidence")))
    return "\n".join(lines), visible


def record_vector(record):
    value = record.get("dimension_vector")
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise _schemas.FrontendContractError("Referenced KB vector must contain exactly seven named coordinates")
    converted = [_schemas.rational_string(x) for x in value]
    if any(x is None for x in converted):
        raise _schemas.FrontendContractError("Referenced KB vector is incomplete")
    return dict(zip(_config.BASE_DIMENSIONS, converted))


def gate_checker_audit(ledger, audit, retrieval):
    """Validate identity/coverage mechanically; unresolved physics stays RETRY.

    IDs identify the actual selected KB entities. No free-text canonical-name echo,
    fuzzy alias matching or unselected-record anchor is an additional gate.
    """
    ledger = _schemas.validate_ledger(ledger)
    audit = _schemas.CheckerDimensionAudit.model_validate(_schemas.model_dict(audit)).model_dump()
    digest = ledger_sha256(ledger)
    if audit["ledger_sha256"] != digest:
        raise _schemas.FrontendContractError("Checker reviewed a stale or different ledger hash")
    names = ledger["input_variable_names"]
    rows = audit["variables"]
    if len({x["csv_name"] for x in rows}) != len(rows) or {x["csv_name"] for x in rows} != set(names):
        raise _schemas.FrontendContractError("Checker must audit each current input exactly once")
    audits = {x["csv_name"]: x for x in rows}
    results, feedback, used_records, issues = [], [], [], []
    enabled = retrieval.get("status") == "success"
    matches = retrieval.get("matches_by_variable") or {}
    visible = retrieval.get("evidence_records_visible_by_variable")
    for variable in ledger["variables"]:
        name, before = variable["csv_name"], variable["si_exponents"]
        item = audits[name]
        if item["before_si_exponents"] != before:
            raise _schemas.FrontendContractError(f"Checker before vector differs from actual ledger for {name}")
        ids = item["evidence_entry_ids"]
        if len(ids) != len(set(ids)):
            raise _schemas.FrontendContractError("Duplicate evidence entry ID")
        available = {str(x.get("entry_id")): x for x in matches.get(name, [])} if enabled else {}
        selected = []
        for entry_id in ids:
            if not enabled or entry_id not in available or (visible is not None and entry_id not in visible.get(name, [])):
                raise _schemas.FrontendContractError(f"Unseen, wrong-variable or unavailable evidence ID for {name}: {entry_id}")
            selected.append(available[entry_id])
        if item["support_mode"] == "KB_RECORD" and not selected:
            raise _schemas.FrontendContractError(f"KB_RECORD support requires a real cited entry for {name}")
        if item["support_mode"] != "KB_RECORD" and selected:
            raise _schemas.FrontendContractError(f"Evidence IDs require KB_RECORD support mode for {name}")
        suggested = item["suggested_si_exponents"]
        if item["decision"] == "PASSED" and suggested is not None and suggested != before:
            raise _schemas.FrontendContractError(f"PASSED cannot ignore a changed suggested vector for {name}")
        if item["decision"] == "PASSED" and item["support_mode"] == "UNCERTAIN":
            raise _schemas.FrontendContractError(f"PASSED cannot use UNCERTAIN support for {name}")
        target = before if item["decision"] == "PASSED" else suggested
        local_issues = []
        before_unknown = any(value is None for value in before.values())
        if before_unknown:
            local_issues.append("Current SI vector contains genuinely unknown coordinates; Analyzer must resolve them explicitly, never substitute0")
        if item["decision"] == "CONFLICT":
            if target is None or any(value is None for value in target.values()):
                local_issues.append("Checker correction is incomplete; preserve its unknowns and request Analyzer revision, no automatic KB-to-candidate filling")
            elif target == before:
                local_issues.append("CONFLICT has no exponent change; Analyzer must explicitly address the reported physical meaning or reasoning")
        cited_vectors = [record_vector(record) for record in selected]
        if cited_vectors and not all(vector == cited_vectors[0] for vector in cited_vectors):
            raise _schemas.FrontendContractError(f"Cited KB records disagree for {name}")
        reference = cited_vectors[0] if cited_vectors else None
        known_reference_conflicts = {base: {"candidate_or_suggestion": target[base], "cited_record": reference[base]}
                                     for base in _config.BASE_DIMENSIONS
                                     if target is not None and reference is not None and target[base] is not None and target[base] != reference[base]}
        if known_reference_conflicts:
            if item["decision"] == "PASSED":
                local_issues.append("Checker PASSED contradicts the known coordinates of its actual cited KB vector; program forces RETRY")
                # A complete candidate can receive the cited full vector as
                # correction feedback; no ledger is mutated. Unknown coordinates
                # remain unknown until Analyzer explicitly supplies them.
                if not before_unknown:
                    target = reference
            else:
                raise _schemas.FrontendContractError(f"Cited KB vector differs from known Checker correction coordinates for {name}")
        if item["decision"] != "PASSED":
            local_issues.append("Checker decision " + item["decision"])
        diff = {base: {"before": before[base], "suggested": target[base]} for base in _config.BASE_DIMENSIONS
                if target is not None and before[base] != target[base]}
        entities = [{"entry_id": str(record.get("entry_id")),
                     "canonical_quantity": record.get("canonical_quantity", ""),
                     "source_identifier": record.get("source_identifier", "")}
                    for record in selected]
        feedback.append(f"Variable {name}: decision={item['decision']}; reason={item['reason']}\nActual named SI={json.dumps(before)}\nSuggested named SI={json.dumps(target)}\nExact coordinate diff={json.dumps(diff)}\nCited entity IDs/canonical names (program-read, not another model echo)={json.dumps(entities, ensure_ascii=False)}\nKnown coordinate conflicts with cited record={json.dumps(known_reference_conflicts)}\nHard gate issues={json.dumps(local_issues)}")
        for record in selected:
            used_records.append({"csv_name": name, "record": record})
            feedback.append("Referenced KB original record for " + name + ":\n" + json.dumps(record, ensure_ascii=False, sort_keys=True))
        results.append({"csv_name": name, "decision": item["decision"], "before": before, "suggested": target,
                        "coordinate_diff": diff, "issues": local_issues,
                        "referenced_entry_ids": ids, "selected_entities": entities,
                        "cited_reference_si_exponents": reference,
                        "known_coordinate_conflicts_with_reference": known_reference_conflicts})
        issues.extend(f"{name}: {issue}" for issue in local_issues)
    status = "RETRY" if issues else "PASSED"
    return {"gate_status": status, "ledger_sha256": digest, "checker_audit": audit, "variable_results": results,
            "issues": issues, "referenced_records": used_records,
            "feedback": "Typed ledger " + digest + "; program gate=" + status + "\n" + "\n\n".join(feedback),
            "entity_policy": "Only Checker-selected visible record IDs identify cited canonical entities; no unselected-record or Analyzer-name anchor. Equal vectors do not establish equal physical entities.",
            "physical_correctness_scope": "A passed contract is not a theorem of physical correctness. Record ID and coordinate checks establish citation consistency; semantic relevance/applicability in the reason and internal-reasoning physics remain model judgements."}


@tool
def calc_basis_tool(
    D_in: list[list[float]],
    var_names: list[str],
) -> dict:
    """Compute a full-precision null-space basis for Buckingham Pi groups.

    Numerical coefficients returned in ``basis_matrices`` are not rounded.
    Rounding is used only in display-oriented formula strings.
    """
    try:
        if not D_in or not D_in[0]:
            return {
                "status": "error",
                "message": "D_in must be a non-empty 2D matrix.",
            }
        if any(len(row) != len(D_in[0]) for row in D_in):
            return {
                "status": "error",
                "message": "All rows of D_in must have the same length.",
            }
        if len(D_in[0]) != len(var_names):
            return {
                "status": "error",
                "message": (
                    "The number of columns in D_in must match the "
                    "number of variables in var_names."
                ),
            }

        # nsimplify prevents decimal inputs such as 1.0 from unnecessarily
        # degrading the exact null-space calculation.
        M = sympy.Matrix([
            [sympy.nsimplify(value) for value in row]
            for row in D_in
        ])
        null_space_vectors = M.nullspace()
        if not null_space_vectors:
            return {
                "status": "error",
                "message": (
                    "The dimensional matrix has a trivial null space; "
                    "no independent Pi group exists."
                ),
            }

        pi_formulas = {}
        basis_matrices = []
        display_basis_matrices = []

        def format_display_exponent(value):
            simplified = sympy.simplify(value)
            if simplified == 1:
                return ""
            if simplified.is_Integer:
                return str(int(simplified))
            if simplified.is_Rational:
                return (
                    f"{int(simplified.p)}/{int(simplified.q)}"
                )
            return f"{float(simplified.evalf(12)):.8g}"

        for i, vec in enumerate(null_space_vectors):
            full_precision_vector = [
                float(value.evalf(17))
                for value in vec
            ]
            basis_matrices.append(full_precision_vector)
            display_basis_matrices.append([
                round(value, 6)
                for value in full_precision_vector
            ])

            numerator = []
            denominator = []
            for var, exact_exp in zip(var_names, vec):
                exact_exp = sympy.simplify(exact_exp)
                if exact_exp == 0:
                    continue

                positive_exp = abs(exact_exp)
                exponent_text = format_display_exponent(
                    positive_exp
                )
                item = (
                    var
                    if exponent_text == ""
                    else f"{var}^{{{exponent_text}}}"
                )
                if exact_exp > 0:
                    numerator.append(item)
                else:
                    denominator.append(item)

            numerator_text = (
                " * ".join(numerator)
                if numerator
                else "1"
            )
            denominator_text = " * ".join(denominator)
            formula = (
                f"({numerator_text}) / ({denominator_text})"
                if denominator_text
                else numerator_text
            )
            pi_formulas[f"Pi_{i + 1}"] = formula

        B = np.asarray(basis_matrices, dtype=float)
        D_numeric = np.asarray(D_in, dtype=float)
        null_space_residual = float(
            np.linalg.norm(D_numeric @ B.T, ord=np.inf)
        )
        if null_space_residual > 1e-9:
            return {
                "status": "error",
                "message": (
                    "Computed Pi basis failed the dimensional-null-space "
                    f"check: residual={null_space_residual:.3e}."
                ),
            }

        return {
            "status": "success",
            "pi_formulas": pi_formulas,
            "basis_matrices": basis_matrices,
            "basis_matrices_display": display_basis_matrices,
            "null_space_residual_inf": null_space_residual,
            "basis_precision_note": (
                "basis_matrices stores full floating-point precision; "
                "basis_matrices_display is rounded for display only."
            ),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Error computing null space: "
                f"{str(exc)}"
            ),
        }


def _normalize_variable_token(value: Any) -> str:
    """Normalize a variable label for deterministic alias matching."""
    text = str(value or "").strip()
    text = text.replace("*", "star").replace("+", "plus")
    text = text.replace("−", "-").replace("–", "-")
    return re.sub(r"[^0-9A-Za-z]+", "", text).lower()


def _is_obvious_metadata_column(name: str) -> bool:
    token = _normalize_variable_token(name)
    raw = str(name).strip().lower()
    return (
        raw.startswith("unnamed:")
        or token in {
            "index", "row", "rowid", "id", "sample", "sampleid",
            "case", "caseid", "group", "label",
        }
    )


def _match_requested_column(requested: Optional[str], columns: List[str], excluded: set[str]) -> Optional[str]:
    """Match an analyzer/manual variable name to a real CSV column."""
    if requested is None:
        return None
    requested = str(requested).strip().strip("`\"'")
    available = [column for column in columns if column not in excluded]
    if requested in available:
        return requested

    normalized = _normalize_variable_token(requested)
    exact_aliases = [
        column for column in available
        if _normalize_variable_token(column) == normalized
    ]
    if len(exact_aliases) == 1:
        return exact_aliases[0]

    # Conservative containment fallback for common spellings such as u* -> u_star.
    containment = [
        column for column in available
        if normalized
        and (
            normalized in _normalize_variable_token(column)
            or _normalize_variable_token(column) in normalized
        )
    ]
    if len(containment) == 1:
        return containment[0]
    return None


def resolve_output_variable(
    columns: List[str],
    input_vars: List[str],
    analyzed_output_var: Optional[str],
    problem_prompt: str,
    manual_hyperparams: Optional[dict] = None,
) -> dict:
    """Resolve the dependent CSV column with explicit and reproducible precedence.

    Precedence:
      1. Explicit manual override, when supplied.
      2. Structured Analyzer inference.
      3. The sole non-input, non-metadata CSV column.

    The function never silently chooses among multiple unresolved candidates.
    """
    hp = dict(manual_hyperparams or {})
    columns = [str(column) for column in columns]
    excluded = set(str(value) for value in input_vars)
    candidates = [
        column for column in columns
        if column not in excluded and not _is_obvious_metadata_column(column)
    ]

    explicit_key = next(
        (
            key for key in ("output_var", "target_var", "dependent_var")
            if hp.get(key) is not None
        ),
        None,
    )
    if explicit_key is not None:
        requested = hp[explicit_key]
        matched = _match_requested_column(requested, columns, excluded)
        if matched is None:
            return {
                "status": "invalid_manual_output_variable",
                "requested": str(requested),
                "source": f"manual_hyperparams.{explicit_key}",
                "candidates": candidates,
                "columns": columns,
            }
        return {
            "status": "success",
            "output_var": matched,
            "source": f"manual_hyperparams.{explicit_key}",
            "requested": str(requested),
            "candidates": candidates,
        }

    matched = _match_requested_column(analyzed_output_var, columns, excluded)
    if matched is not None:
        return {
            "status": "success",
            "output_var": matched,
            "source": "structured_analyzer",
            "requested": str(analyzed_output_var),
            "candidates": candidates,
        }

    if len(candidates) == 1:
        return {
            "status": "success",
            "output_var": candidates[0],
            "source": "sole_non_input_column",
            "requested": None,
            "candidates": candidates,
        }

    return {
        "status": "output_variable_ambiguous",
        "analyzer_requested": analyzed_output_var,
        "candidates": candidates,
        "columns": columns,
    }


def _execute_original_basis_tool_node(state: _schemas.AdaptiveWorkflowState):
    messages = state.get("basis_tool_messages", [])
    if not messages:
        return {"workflow_status": "basis_tool_not_called"}
    last = messages[-1]
    calls = getattr(last, "tool_calls", [])
    if not calls:
        return {"workflow_status": "basis_tool_not_called"}

    history = []
    for call in calls:
        if call.get("name") != "calc_basis_tool":
            continue
        args = dict(call.get("args", {}))
        result = calc_basis_tool.invoke(args)
        history.append({"args": args, "result": result})
        if result.get("status") != "success":
            return {"basis_tool_history": history, "workflow_status": "basis_failed"}

        input_vars = list(args.get("var_names", []))
        frame = pd.read_csv(state["csv_FilePath"], nrows=1)
        hp = dict(state.get("manual_hyperparams", {}))
        output_resolution = resolve_output_variable(
            columns=frame.columns.tolist(),
            input_vars=input_vars,
            analyzed_output_var=state.get("analyzed_output_var"),
            problem_prompt=state.get("prompt_problem", ""),
            manual_hyperparams=hp,
        )
        if output_resolution.get("status") != "success":
            return {
                "basis_tool_history": history,
                "workflow_status": str(output_resolution.get("status")),
                "output_resolution": output_resolution,
                "output_candidates": output_resolution.get("candidates", []),
            }
        output_var = str(output_resolution["output_var"])
        preprocessing = _workflow._resolve_dimensionless_input_preprocessing(
            hp.get("dimensionless_input_preprocessing"), input_vars, state.get("dimension_contract")
        )
        basis_formulas = result["pi_formulas"]
        if preprocessing:
            # A zero dimension column is a standalone free coordinate in the exact null-space basis.
            formula_terms = _workflow._format_preprocessed_input_terms(preprocessing)
            basis_formulas = {name: formula_terms.get(formula, formula) for name, formula in basis_formulas.items()}
        _presentation.display_basis_summary(
            input_vars=input_vars,
            output_var=output_var,
            pi_formulas=basis_formulas,
            output_resolution=output_resolution,
            hyperparameters=state.get("manual_hyperparams", {}),
        )
        return {
            "basis_tool_history": history,
            "basis_pi_formulas": basis_formulas,
            "basis_pi_matrices": result["basis_matrices"],
            "input_vars": input_vars,
            "output_var": output_var,
            "output_resolution": output_resolution,
            "workflow_status": "basis_ready",
        }
    return {"basis_tool_history": history, "workflow_status": "basis_failed"}


def _manual_input_override(state, hp):
    candidates = [value for value in (state.get("frontend_expected_inputs"), hp.get("frontend_expected_inputs")) if value is not None]
    for value in candidates:
        if not isinstance(value, (list, tuple)) or not value or any(not isinstance(name, str) or not name or name != name.strip() for name in value):
            raise _schemas.FrontendContractError("Explicit input override must be an ordered list of exact CSV names")
    if candidates and any(list(value) != list(candidates[0]) for value in candidates[1:]):
        raise _schemas.FrontendContractError("Conflicting explicit input-role/order overrides")
    return list(candidates[0]) if candidates else None


def _manual_output_override(state, hp):
    sources = []
    for label, value in [("state.frontend_expected_output", state.get("frontend_expected_output")),
                         ("manual.frontend_expected_output", hp.get("frontend_expected_output"))] + [
                             ("manual." + key, hp.get(key)) for key in ("output_var", "target_var", "dependent_var")]:
        if value is not None:
            if not isinstance(value, str) or not value or value != value.strip():
                raise _schemas.FrontendContractError("Explicit output override must be one exact CSV name: " + label)
            sources.append((label, value))
    if len({value for _, value in sources}) > 1:
        raise _schemas.FrontendContractError("Conflicting explicit output-role overrides: " + ", ".join(label for label, _ in sources))
    return sources[0][1] if sources else None


def _exact_csv_columns(csv_path):
    # pandas silently renames duplicate headers; never treat those invented names
    # as literal CSV bindings. The scientific reader remains unchanged.
    import csv
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    if not columns or len(columns) != len(set(columns)):
        raise _schemas.FrontendContractError("Raw CSV header must contain unique exact column names")
    parsed = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if parsed != columns:
        raise _schemas.FrontendContractError("Parsed CSV names differ from literal header; explicit column mapping is required")
    return columns


def _compile_state(state):
    columns = _exact_csv_columns(state["csv_FilePath"])
    hp = dict(state.get("manual_hyperparams") or {})
    compiled = compile_ledger(
        state.get("dimension_ledger"), columns,
        expected_inputs=_manual_input_override(state, hp),
        expected_output=_manual_output_override(state, hp),
    )
    prior = state.get("dimension_contract")
    if not isinstance(prior, dict) or "provisional_zero_coordinates" not in prior:
        return compiled
    coordinates = prior["provisional_zero_coordinates"]
    if prior.get("ledger_sha256") != compiled["ledger_sha256"]:
        raise _schemas.FrontendContractError("Provisional zeros belong to a different typed ledger")
    if not isinstance(coordinates, list) or not coordinates:
        raise _schemas.FrontendContractError("Invalid provisional zero coordinates")
    allowed = {(record["csv_name"], base) for record in compiled["ledger"]["variables"]
               for base in _config.BASE_DIMENSIONS if record["si_exponents"][base] == "0"}
    verified = []
    for item in coordinates:
        if (not isinstance(item, dict) or set(item) != {"csv_name", "base"}
                or not all(isinstance(value, str) for value in item.values())):
            raise _schemas.FrontendContractError("Invalid provisional zero coordinates")
        pair = (item["csv_name"], item["base"])
        if pair not in allowed:
            raise _schemas.FrontendContractError("Provisional zero coordinates must identify unique zero input exponents")
        allowed.remove(pair)
        verified.append(dict(item))
    compiled["contract"]["provisional_zero_coordinates"] = verified
    return compiled


def deterministic_basis_arguments(state):
    compiled = _compile_state(state)
    gate = state.get("dimension_audit") or {}
    if gate.get("gate_status") != "PASSED" or gate.get("ledger_sha256") != compiled["ledger_sha256"]:
        raise _schemas.FrontendContractError("Basis requires PASSED audit of the exact current typed ledger")
    if compiled["contract"]["status"] != "complete":
        raise _schemas.FrontendContractError("Unknown SI coordinates cannot reach the basis tool")
    return {"D_in": compiled["contract"]["D_in"], "var_names": compiled["contract"]["var_names"]}, compiled["ledger_sha256"]


def adaptive_basis_decider_node(state: dict):
    args, digest = deterministic_basis_arguments(state)
    message = AIMessage(content="Deterministic basis request from the exact audited typed ledger; no Builder LLM.",
        tool_calls=[{"name": "calc_basis_tool", "args": args, "id": "basis" + digest[:16], "type": "tool_call"}])
    return {"basis_tool_messages": [message], "basis_request_ledger_sha256": digest, "basis_request": args}


def adaptive_basis_executor_node(state: dict):
    args, digest = deterministic_basis_arguments(state)
    messages = state.get("basis_tool_messages") or []
    calls = getattr(messages[-1], "tool_calls", []) if messages else []
    if state.get("basis_request_ledger_sha256") != digest or len(calls) != 1 or calls[0].get("name") != "calc_basis_tool" or calls[0].get("args") != args:
        return {"workflow_status": "terminal_unresolved", "basis_tool_history": [{"status": "blocked", "reason": "Basis request differs from audited ledger"}]}
    compiled = _compile_state(state)
    # The retained original resolver supports legacy manual aliases. They have
    # already been validated, and display fields cannot become a second source.
    result = _execute_original_basis_tool_node({**state, **compiled["legacy_fields"], "dimension_contract": compiled["contract"]})
    if result.get("workflow_status") == "basis_ready" and result.get("output_var") != compiled["ledger"]["output_variable_name"]:
        return {**result, "workflow_status": "terminal_unresolved",
                "output_resolution": {"status": "blocked_output_rebinding", "expected": compiled["ledger"]["output_variable_name"], "observed": result.get("output_var")}}
    return result


def _blocked_audit(state, error, retrieval=None):
    """Record a terminal audit failure without changing the recovery policy."""
    message = "terminal_unresolved: " + str(error)
    audit = {
        "gate_status": "BLOCKED",
        "issues": [message],
        "ledger_sha256": state.get("dimension_ledger_sha256"),
    }
    _presentation.display_audit_result(audit, state.get("manual_hyperparams"))
    return {
        "hallucination_check": [message],
        "hallucination_decision": ["BLOCKED"],
        "dimension_audit": audit,
        "knowledge_retrieval": retrieval or {},
        "workflow_status": "terminal_unresolved",
    }


def analyzer_reply_payload(response):
    """Read raw Analyzer arguments when the structured parser retained them."""
    if not isinstance(response, dict) or set(response) != {"raw", "parsed", "parsing_error"}:
        return response
    raw = response["raw"]
    calls = getattr(raw, "tool_calls", ())
    if len(calls) != 1 or getattr(raw, "invalid_tool_calls", ()) or (
        calls[0].get("name") != _schemas.AnalyzerDimensionLedger.__name__
        or not isinstance(calls[0].get("args"), dict)
    ):
        raise _schemas.FrontendContractError("Analyzer must return exactly one valid ledger tool call")
    return calls[0]["args"]


def _output_binding_error(name, columns, expected_output):
    if not isinstance(name, str) or name not in columns:
        return f"Output variable {name!r} is not an exact CSV column"
    if expected_output is not None and name != expected_output:
        return f"Output variable {name!r} differs from the explicit output override {expected_output!r}"
    return None


def _record_output_binding(action, submitted, selected, status, duration_seconds=None):
    context = _provenance._ACTIVE_OBSERVATION.get()
    if context is None:
        return
    ledger, _, visit = context
    with ledger.lock:
        ledger.data.setdefault("output_binding_events", []).append({
            "action": action, "submitted": submitted, "selected": selected, "status": status,
            "duration_seconds": duration_seconds, "analyzer_visit": visit,
        })


def _verify_output_column(prompt, columns, submitted, expected_output):
    checker = _models._require_model_ready(_models.output_binding_checker_llm, "Output binding Checker")
    context = _provenance._ACTIVE_OBSERVATION.get()
    token = None
    if context is not None:
        ledger, _, visit = context
        token = _provenance._ACTIVE_OBSERVATION.set((ledger, "Output_Binding_Checker", visit))
    started = time.perf_counter()
    selected = None
    status = "failed"
    try:
        reply = checker.invoke([
            SystemMessage(content=(
                "Independently identify the dependent response requested in the full problem. "
                "Distinguish the requested response from inputs, metadata, and other derived columns. "
                "Return its zero-based index from the CSV columns, or null if ambiguous. "
                "Check column binding only."
            )),
            HumanMessage(content=json.dumps({
                "problem": prompt,
                "csv_columns_by_index": dict(enumerate(columns)),
            }, ensure_ascii=False)),
        ])
        index = _schemas.OutputColumnSelection.model_validate(_schemas.model_dict(reply)).column_index
        if index is None or index >= len(columns):
            raise _schemas.FrontendContractError("Output binding Checker did not select a valid CSV column")
        selected = columns[index]
        failure = _output_binding_error(selected, columns, expected_output)
        if failure:
            raise _schemas.FrontendContractError(failure)
        status = "confirmed" if selected == submitted else "repaired"
        return selected
    finally:
        if token is not None:
            _provenance._ACTIVE_OBSERVATION.reset(token)
        _record_output_binding("verify", submitted, selected, status, time.perf_counter() - started)


def _bind_analyzer_output(response, state, columns, expected_output):
    payload = _schemas.model_dict(response)
    if not isinstance(payload, dict):
        return response
    if expected_output is not None and expected_output not in columns:
        raise _schemas.FrontendContractError(f"Explicit output override {expected_output!r} is not an exact CSV column")
    submitted = payload.get("output_variable_name")
    decisions = state.get("hallucination_decision") or []

    if decisions and decisions[-1] == "RETRY":
        gate = state.get("dimension_audit") or {}
        prior = state.get("dimension_ledger")
        digest = state.get("dimension_ledger_sha256")
        if (gate.get("gate_status") != "RETRY" or not isinstance(prior, dict)
                or not digest or gate.get("ledger_sha256") != digest or ledger_sha256(prior) != digest):
            raise _schemas.FrontendContractError("Output binding on revision requires the ledger that was sent for revision")
        selected = prior.get("output_variable_name")
        failure = _output_binding_error(selected, columns, expected_output)
        if failure:
            raise _schemas.FrontendContractError(failure)
        if selected != submitted:
            _record_output_binding("preserve_on_revision", submitted, selected, "preserved")
    else:
        selected = _verify_output_column(state["prompt_problem"], columns, submitted, expected_output)

    if selected == submitted:
        return response
    return {**payload, "output_variable_name": selected}


def analyze_problem_node(state: dict):
    """Infer the physical-input ledger and display the actual Analyzer result."""
    hp = dict(state.get("manual_hyperparams") or {})
    previous = list(state.get("hallucination_check") or [])

    _presentation.display_frontend_stage(
        "Stage 1/7 · Problem and variable analysis",
        status="running",
        rows=[("Analysis round", len(previous) + 1)],
        hyperparameters=hp,
    )

    try:
        columns = _exact_csv_columns(state["csv_FilePath"])
        frame = pd.read_csv(state["csv_FilePath"])
        expected_output = _manual_output_override(state, hp)
        expected_inputs = _manual_input_override(state, hp)

        if not previous:
            try:
                _presentation.display_agent_run_header(
                    state["prompt_problem"],
                    state["csv_FilePath"],
                    len(frame),
                    columns,
                    hp,
                )
            except Exception as exc:
                print(f"[Display warning] Run header: {exc}")

        feedback = "\n\n".join(
            f"Audit round {i+1}:\n{text}"
            for i, text in enumerate(previous)
        ) or "First analysis; no prior audit."

        null_policy = hp.get("frontend_null_policy", "strict")
        provisional = uses_provisional_zero(null_policy)
        selected_analyzer = _models.analyzer_raw_llm if provisional else _models.analyzer_llm
        response = _models._require_model_ready(selected_analyzer, "Analyzer").invoke([
            SystemMessage(content=(
                "Infer one typed physical-input ledger. Exact CSV names and explicit input order are binding. "
                "Return each input exactly once with its physical quantity name, a plain-language physical_definition, "
                "and all seven named SI exponents M,L,T,I,Theta,N,J as rational strings. Use null for unknown "
                "coordinates. For a base quantity known not to participate, explicitly return '0'; this is known absence, "
                "not uncertainty. Use null only when the exponent is genuinely unknown. Never infer unknown as0. "
                "Exclude the output and metadata/unused/derived columns. "
                "quantity_name and physical_definition are retrieval meaning fields: do not put assigned SI units, "
                "dimension vectors or exponent formulas there. Do not output a separate dimensional matrix or n/r/k. "
                "They are generated deterministically from this ledger. A confident name is not proof of its SI vector."
            )),
            HumanMessage(content=(
                f"Problem statement:\n{state['prompt_problem']}\n\nCSV columns: {columns}\n"
                f"CSV preview:\n{frame.head().to_string()}\n\n"
                f"Explicit manual input-role/order override: {expected_inputs}\n"
                f"Explicit manual output override: {expected_output}\n\n"
                f"Previous physics-audit feedback (exact coordinate differences and cited original KB records):\n{feedback}"
            )),
        ])
        if provisional:
            response = analyzer_reply_payload(response)

        _presentation.display_frontend_details(
            "Analyzer structured response",
            _schemas.model_dict(response),
            hp,
        )

        raw_response = deepcopy(_schemas.model_dict(response))
        bound_response, input_order_binding = _bindings.bind_analyzer_order(
            _bind_analyzer_output(response, state, columns, expected_output),
            expected_inputs,
        )
        compiled = compile_ledger(
            bound_response,
            columns,
            expected_inputs=expected_inputs,
            expected_output=expected_output,
            null_policy=null_policy,
        )
    except Exception as exc:
        _presentation.display_frontend_stage(
            "Stage 1/7 · Problem and variable analysis",
            status="failed",
            summary=f"Analyzer or ledger validation failed: {exc}",
            hyperparameters=hp,
        )
        raise

    _presentation.display_analysis_result(compiled, hp)
    return {
        "problem_description": state["prompt_problem"],
        **compiled["legacy_fields"],
        "dimension_ledger": compiled["ledger"],
        "dimension_ledger_sha256": compiled["ledger_sha256"],
        "dimension_contract": compiled["contract"],
        "dimension_audit": None,
        "analyzer_raw_ledger": raw_response,
        "input_order_binding": input_order_binding,
    }


def physics_consistency_node(state: dict):
    """Retrieve evidence, audit the exact ledger and display the real feedback."""
    hp = dict(state.get("manual_hyperparams") or {})
    _presentation.display_frontend_stage(
        "Stage 2/7 · Physics consistency audit",
        status="running",
        rows=[
            ("Audit round", len(state.get("hallucination_decision") or []) + 1)
        ],
        hyperparameters=hp,
    )

    try:
        compiled = _compile_state(state)
    except Exception as exc:
        return _blocked_audit(
            state, "Invalid typed input ledger: " + str(exc)
        )

    enabled = state.get("frontend_rag_enabled")
    if enabled is None:
        enabled = hp.get("frontend_rag_enabled", True)
    if type(enabled) is not bool:
        return _blocked_audit(
            state, "frontend_rag_enabled must be a boolean"
        )

    ledger = compiled["ledger"]
    digest = compiled["ledger_sha256"]
    contexts = render_retrieval_interpretation(ledger)

    if enabled:
        retrieval = _knowledge.safe_retrieve_dimension_evidence(
            problem_description=state.get(
                "problem_description", state.get("prompt_problem", "")
            ),
            input_variable_names=ledger["input_variable_names"],
            dimension_detect="\n".join(contexts.values()),
            dimension_ledger=ledger,
        )
    else:
        retrieval = {
            "status": "disabled_for_ablation",
            "retrieval_modes": [],
            "n_retrieved_records": 0,
            "matches_by_variable": {},
            "queries_by_variable": {},
            "query_sha256_by_variable": {},
            "evidence_records_visible_by_variable": {},
            "evidence_markdown": (
                "External retrieval disabled. Use INTERNAL_REASONING or "
                "UNCERTAIN; no KB entry IDs are available."
            ),
        }

    _presentation.display_retrieval_result(retrieval, hp)

    system = (
        "Audit every named physical input in this exact typed ledger, including literal SI coordinate values. "
        "Do not substitute the correct definition or a KB vector for a wrong candidate and then claim PASSED. "
        "Return ledger_sha256 and each csv_name exactly once; before_si_exponents must copy the actual candidate. "
        "PASSED confirms the actual vector, CONFLICT requires a complete corrected suggested vector, and UNCERTAIN "
        "does not permit numerical execution. KB_RECORD requires actual entry_id citations from that variable's "
        "retrieved records. Canonical entity names are read by the program from the selected IDs; do not echo "
        "another quantity-name field. Your reason must explain why each cited record applies to this variable in "
        "the original problem, not merely why its vector matches. Cited vectors must support all known coordinates "
        "of the proposed/confirmed SI vector. INTERNAL_REASONING uses no IDs and must explain "
        "the physical meaning. Retrieval rank is relevance, not truth. Resolve symbol ambiguity from the original "
        "problem; do not choose an unrelated matching symbol. Known absent base dimensions must be explicit '0'; "
        "null means genuinely unknown, not unused. If any current coordinate is unknown, request revision or report "
        "UNCERTAIN; never call a partly known vector PASSED. Give a complete correction when physically justified. "
        "If a correction remains partly unknown, preserve null and explain what remains unresolved; the program "
        "will request revision rather than invent zeros. Lack of KB alone is not a physical contradiction. "
        "Matrix compilation and n/r/k are program tasks, not audit tasks."
    )

    provisional_context = ""
    if coordinates := compiled["contract"].get("provisional_zero_coordinates"):
        system += (
            " The listed provisional zero coordinates were null in the Analyzer response. "
            "Copy their current zero values in before_si_exponents and verify each from the physical definition "
            "or applicable retrieved evidence. Use PASSED only if you can affirm those zeros. "
            "If an exponent is nonzero, report CONFLICT with the complete corrected vector. "
            "If any provisional coordinate cannot be verified, report UNCERTAIN."
        )
        provisional_context = f"Provisional zero coordinates:\n{json.dumps(coordinates, ensure_ascii=False)}\n\n"

    try:
        answer = _models._require_model_ready(_models.checker_llm, "Checker").invoke([
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Problem:\n{state.get('problem_description', state.get('prompt_problem', ''))}\n\n"
                f"Typed ledger SHA256: {digest}\nTyped candidate ledger:\n"
                f"{json.dumps(ledger, ensure_ascii=False, sort_keys=True)}\n\n"
                f"{provisional_context}"
                f"Retrieved dimensional evidence:\n{retrieval.get('evidence_markdown', '')}"
            )),
        ])

        _presentation.display_frontend_details(
            "Checker structured response",
            _schemas.model_dict(answer),
            hp,
        )
        gate = gate_checker_audit(ledger, answer, retrieval)
    except Exception as exc:
        return _blocked_audit(
            state, "Checker contract invalid: " + str(exc), retrieval
        )

    decision = gate["gate_status"]
    limit = hp.get("frontend_max_revisions", 2)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        return _blocked_audit(
            state,
            "frontend_max_revisions must be a nonnegative integer",
            retrieval,
        )

    prior_retries = sum(
        item == "RETRY"
        for item in state.get("hallucination_decision", [])
    )
    if decision == "RETRY" and prior_retries >= limit:
        decision = "BLOCKED"
        gate = {
            **gate,
            "gate_status": "BLOCKED",
            "retry_budget_exhausted": True,
            "feedback": (
                gate["feedback"]
                + f"\nRevision limit {limit} exhausted; "
                "terminal_unresolved, no basis execution."
            ),
        }

    _presentation.display_audit_result(gate, hp)
    return {
        **compiled["legacy_fields"],
        "dimension_ledger": ledger,
        "dimension_ledger_sha256": digest,
        "dimension_contract": compiled["contract"],
        "dimension_audit": gate,
        "dimension_audit_history": [gate],
        "dimension_query": retrieval.get("queries_by_variable", {}),
        "knowledge_retrieval": retrieval,
        "hallucination_decision": [decision],
        "hallucination_check": [gate["feedback"]],
        "workflow_status": (
            "terminal_unresolved" if decision == "BLOCKED"
            else "frontend_verified" if decision == "PASSED"
            else "frontend_revision_requested"
        ),
    }
