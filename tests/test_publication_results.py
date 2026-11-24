"""Acceptance tests compare real saved evidence with reported author results."""
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = runpy.run_path(str(ROOT / "scripts" / "check_publication.py"))["check_cases"]


@pytest.fixture(scope="module")
def publication_report():
    return {case["case_id"]: case for case in CHECK(ROOT)["cases"]}


@pytest.mark.parametrize("case_id", [
    "mhd", "pressure_gradient", "rayleigh_benard", "laser_keyhole", "blasius",
    "sandgrain", "velocity_scaling", "goldenfeld", "cess_channel", "rough_pipe",
])
def test_saved_result_matches_reported_numerical_evidence(case_id, publication_report):
    case = publication_report[case_id]
    assert case["status"] == "passed", case["failures"]
