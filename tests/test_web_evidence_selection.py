"""Views must show measured evidence rather than similarly named plan fields."""
from pathlib import Path
import json
import pytest

from ai_buckingham.web.data import first

ROOT = Path(__file__).resolve().parents[1]


def test_measured_regime_evidence_takes_precedence_over_plan():
    actual = {"surrogate_gradients": [[1.0, 0.0]], "feature_coordinates": [[0.2, 0.4]]}
    state = {"workflow_plan": {"regime_detection": {"candidate_cluster_counts": [2,3,4]}},
             "compiled_workflow": {"regime_detection": {"min_cluster_samples":60}},
             "workflow_result": {"regime_detection": actual}}
    assert first(state, "regime_detection") is actual


@pytest.mark.parametrize("case_id", ["blasius","mhd","cess_channel","rough_pipe","sandgrain","velocity_scaling"])
def test_included_state_exposes_recorded_gradients_and_current_local_results(case_id):
    paths = sorted((ROOT / "results" / "cases" / case_id).glob("*/manifest.json"), reverse=True)
    manifest = next(p for p in paths if json.loads(p.read_text(encoding="utf-8"))["status"] == "completed")
    state = json.loads((manifest.parent / "state.json").read_text(encoding="utf-8"))
    detection = first(state, "regime_detection")
    assert detection is state["workflow_result"]["regime_detection"]
    assert len(detection["surrogate_gradients"]) == len(detection["feature_coordinates"]) > 0
    assert first(state, "local_workflows") is state["workflow_result"]["local_workflows"]
