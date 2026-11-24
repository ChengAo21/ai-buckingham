"""Validate published scientific inputs independently of live agent execution."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("M", "L", "T", "I", "Theta", "N", "J")
CASE_ROWS = {
    "mhd": 483,
    "pressure_gradient": 144,
    "rayleigh_benard": 182,
    "laser_keyhole": 90,
    "blasius": 200,
    "sandgrain": 104,
    "velocity_scaling": 1109,
    "goldenfeld": 414,
    "cess_channel": 1000,
    "rough_pipe": 1200,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(case_id: str) -> dict:
    return read_json(ROOT / "configs" / "cases" / f"{case_id}.json")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def test_case_catalogue_has_ten_portable_unambiguous_entries():
    entries = read_json(ROOT / "configs" / "cases.json")["cases"]
    assert len(entries) == len(CASE_ROWS) == 10
    assert {entry["case_id"] for entry in entries} == set(CASE_ROWS)
    assert {path.stem for path in (ROOT / "configs" / "cases").glob("*.json")} == set(CASE_ROWS)
    for entry in entries:
        case_id = entry["case_id"]
        assert entry["config"] == f"configs/cases/{case_id}.json"
        assert entry["dataset"] == f"data/datasets/{case_id}.csv"
        assert entry["title"] == config(case_id)["title"]
        assert "\ufffd" not in entry["title"]
    assert config("rayleigh_benard")["title"] == "Rayleigh-Benard convection"


@pytest.mark.parametrize("case_id", CASE_ROWS)
def test_case_bytes_schema_rows_and_numeric_values_agree_with_publication(case_id):
    case = config(case_id)
    data = ROOT / case["dataset"]
    assert data.resolve().is_relative_to((ROOT / "data" / "datasets").resolve())
    assert data.name == f"{case_id}.csv"
    metadata = case["data_metadata"]
    record = next(item for item in read_json(ROOT / "data" / "manifest.json")["datasets"]
                  if item["case_id"] == case_id)
    assert sha256(data) == metadata["sha256"] == record["sha256"]
    header, rows = read_csv(data)
    assert len(header) == len(set(header))
    assert header == metadata["columns"] == record["columns"]
    assert len(rows) == CASE_ROWS[case_id] == metadata["rows"] == record["rows"]
    assert metadata["original_filename"] == record["original_filename"]
    assert metadata["original_column_order_preserved"] is True
    inputs, output = case["input_columns"], case["output_column"]
    assert inputs and len(inputs) == len(set(inputs)) and output not in inputs
    assert [column for column in header if column in inputs] == inputs
    assert set(header) == set(inputs + [output] + metadata["auxiliary_columns"])
    numerical = np.asarray([[float(row[column]) for column in inputs + [output]] for row in rows])
    assert np.isfinite(numerical).all()
    positive_response_rows = int(np.count_nonzero(numerical[:, -1] > 0))
    assert positive_response_rows == metadata["response_log_eligible_rows"]
    assert positive_response_rows == {"blasius": 196, "velocity_scaling": 1103}.get(case_id, CASE_ROWS[case_id])


@pytest.mark.parametrize("case_id", CASE_ROWS)
def test_case_dimensions_are_traceable_to_archived_reference(case_id):
    case = config(case_id)
    provenance = case["provenance"]
    reference = ROOT / provenance["reference_dimensions_source"]
    assert reference.resolve().is_relative_to(ROOT.resolve())
    assert sha256(reference) == provenance["reference_dimensions_sha256"]
    expected = read_json(reference)[provenance["reference_case_id"]]
    dimensions = case["expected_dimensions"]
    assert tuple(dimensions["dimension_order"]) == DIMENSIONS
    assert dimensions["inputs"] == expected["inputs"]
    assert set(case["input_columns"]) == set(expected["inputs"])
    assert dimensions["output"] == expected["output_dimensions"] == dict.fromkeys(DIMENSIONS, 0)
    assert case["output_column"] == expected["output"]
    for vector in dimensions["inputs"].values():
        assert set(vector) == set(DIMENSIONS)
        assert all(isinstance(value, (int, float)) and np.isfinite(value) for value in vector.values())
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_sha256"])
    assert provenance["original_prompt_symbol"] and provenance["original_hyperparams_symbol"]
    assert case["prompt"] and case["output_column"] in case["prompt"]


def test_sandgrain_shift_is_explicit_dimensionless_and_does_not_rewrite_input():
    case = config("sandgrain")
    assert case["manual_hyperparams"]["dimensionless_input_preprocessing"] == {
        "Sk": {"transform": "shift", "offset": 2.0}
    }
    assert not any(case["expected_dimensions"]["inputs"]["Sk"].values())
    _, rows = read_csv(ROOT / case["dataset"])
    skewness = np.asarray([float(row["Sk"]) for row in rows])
    assert np.any(skewness < 0), "The supplied raw skewness values must remain unshifted on disk"
    assert np.all(skewness + 2.0 > 0), "Declared shift must make the logarithmic input admissible"
    assert case["manual_hyperparams"]["gpr_stratify_column"] == "case"
    assert case["manual_hyperparams"]["gpr_log_features"] is False
    assert case["manual_hyperparams"]["gpr_log_target"] is False


def test_prompt_formula_constraints_remain_explicit_for_the_two_scaling_cases():
    velocity = config("velocity_scaling")
    assert velocity["formula_constraints"] == {"input_wall_units_power": 1, "output_wall_units_power": 0}
    assert "constrained to 1 for the INPUT and 0 for the OUTPUT" in velocity["prompt"]
    goldenfeld = config("goldenfeld")
    assert goldenfeld["formula_constraints"] == {"output_relative_roughness_power": 0}
    assert "constrained to 0 for the OUTPUT" in goldenfeld["prompt"]


@pytest.mark.parametrize("filename,count,prefix", [
    ("physical_quantity_index.npz", 247, "pq_"),
    ("dimensionless_relation_index.npz", 251, "dg_"),
])
def test_embedding_index_identity_record_count_finiteness_and_normalization(filename, count, prefix):
    directory = ROOT / "knowledge" / "indexes"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or not (directory / filename).is_file():
        pytest.skip("Embedding indexes have not been built in this checkout")
    manifest = read_json(manifest_path)
    item = next(entry for entry in manifest["indexes"] if entry["file"] == filename)
    kb = ROOT / item["knowledge_base"]
    record_ids = [line.split("|", 2)[1].strip() for line in kb.read_text(encoding="utf-8").splitlines()
                  if line.lstrip().startswith(f"| {prefix}")]
    assert len(record_ids) == len(set(record_ids)) == count
    assert sha256(kb) == item["kb_sha256"]
    assert sha256(directory / filename) == item["sha256"]
    with np.load(directory / filename, allow_pickle=False) as archive:
        embeddings = archive["embeddings"]
        assert archive["kb_sha256"].item() == item["kb_sha256"]
        assert archive["embedding_model"].item() == item["embedding_model"] == "GLM-Embedding-3"
        assert archive["embedding_provider"].item() == manifest["provider"]
        assert int(archive["record_count"]) == count
        assert int(archive["embedding_dimension"]) == 2048
        assert list(embeddings.shape) == item["shape"] == [count, 2048]
        assert np.issubdtype(embeddings.dtype, np.floating) and np.isfinite(embeddings).all()
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-10, atol=1e-12)
