"""Bind configured input order by exact names before dimensional compilation.

These adapters change neither SI assignments nor numerical algorithms.  Missing,
extra or duplicate inputs remain errors.  Analyses without an explicit order
retain the Analyzer's existing ordering behavior.
"""
from __future__ import annotations

from copy import deepcopy

from .schemas import FrontendContractError, model_dict


def _ordered_names(value):
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(name, str) or not name or name != name.strip() for name in value)
            or len(value) != len(set(value))):
        raise FrontendContractError("Explicit input order must contain unique exact CSV names")
    return list(value)


def frontend_bindings(config):
    """Expose optional case input/output metadata as explicit graph contracts."""
    bindings = {}
    if config.get("input_columns") is not None:
        bindings["frontend_expected_inputs"] = _ordered_names(config["input_columns"])
    if config.get("output_column") is not None:
        output = config["output_column"]
        if not isinstance(output, str) or not output or output != output.strip():
            raise FrontendContractError("Explicit output must be one exact CSV name")
        if output in bindings.get("frontend_expected_inputs", []):
            raise FrontendContractError("Explicit output cannot also be an input")
        bindings["frontend_expected_output"] = output
    return bindings


def bind_analyzer_order(response, expected_inputs=None):
    """Bind records by ``csv_name`` without editing the supplied response.

Only role names and order are handled here.  SI coordinates, including unknown
or provisional values, reach the existing dimensional validator unchanged.
    """
    payload = deepcopy(model_dict(response))
    observed_names = payload.get("input_variable_names") if isinstance(payload, dict) else None
    observed_records = payload.get("variables") if isinstance(payload, dict) else None
    record_names = ([record.get("csv_name") if isinstance(record, dict) else None
                     for record in observed_records] if isinstance(observed_records, list) else None)
    evidence = {"authority": "analyzer" if expected_inputs is None else "explicit_case_contract",
                "observed_input_order": deepcopy(observed_names),
                "observed_record_order": record_names,
                "bound_order": deepcopy(observed_names), "reordered": False,
                "matching": "exact_csv_name"}
    if expected_inputs is None:
        return payload, evidence

    expected = _ordered_names(expected_inputs)
    if not isinstance(payload, dict) or not isinstance(observed_records, list):
        raise FrontendContractError("Analyzer response must contain named variable records")
    names = _ordered_names(observed_names)
    _ordered_names(record_names)
    # Preserve the compiler's existing cleanup of the separately declared output.
    output = payload.get("output_variable_name")
    names = [name for name in names if name != output]
    records = [record for record in observed_records if record["csv_name"] != output]
    cleaned_record_names = [record["csv_name"] for record in records]
    if set(names) != set(cleaned_record_names):
        raise FrontendContractError("Variable records must cover exactly the selected input names")
    if set(names) != set(expected):
        raise FrontendContractError("Input set contradicts the explicit case contract")
    by_name = {record["csv_name"]: record for record in records}
    payload["input_variable_names"] = expected
    payload["variables"] = [by_name[name] for name in expected]
    evidence.update(bound_order=expected,
                    reordered=observed_names != expected or record_names != expected)
    return payload, evidence
