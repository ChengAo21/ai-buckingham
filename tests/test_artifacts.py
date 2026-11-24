"""Scientific evidence remains portable, finite and immutable across consumers."""
from pathlib import Path
import json

import numpy as np
import pytest

from ai_buckingham.artifacts import RunRecorder, load_run, portable


def test_state_export_preserves_arrays_and_redacts_credentials(tmp_path):
    recorder = RunRecorder(tmp_path / "run", "custom", {"dataset": "missing.csv", "prompt": "example"})
    update = {"basis": np.array([[1.0, -1.0]]), "mi": np.float64(2.5), "api_key": "never-distribute-this"}
    recorder.record_event("NumericalInference", update)
    state = {"workflow_status": "success", "workflow_result": update,
             "final_report": "A recorded result", "array": np.array([1, 2, 3])}
    recorder.finalize(state)
    loaded = load_run(recorder.run_dir)
    assert loaded["state"]["array"] == [1, 2, 3]
    assert loaded["state"]["workflow_result"]["mi"] == 2.5
    assert loaded["events"][0]["update"]["basis"] == [[1.0, -1.0]]
    assert loaded["state"]["workflow_result"]["api_key"] == "[REDACTED]"
    assert "never-distribute-this" not in (recorder.run_dir / "state.json").read_text()
    assert (recorder.run_dir / "report.html").is_file()
    with pytest.raises(FileExistsError):
        RunRecorder(recorder.run_dir, "custom", {})


def test_nonfinite_diagnostics_are_explicit_in_standard_json():
    exported = portable({"mi": np.nan, "ratio": np.inf, "loss": -np.inf})
    assert json.loads(json.dumps(exported, allow_nan=False))["ratio"] == {"nonfinite": "inf"}


def test_rich_cards_and_original_image_are_stored(tmp_path):
    recorder = RunRecorder(tmp_path / "run", "custom", {})
    original_html = '<section data-stage="physical-consistency">CONFLICT then RETRY</section>'
    recorder.save_rich_outputs([{"data": {"text/html": original_html}}])
    recorder.finalize({"workflow_status": "terminal_unresolved"}, status="blocked")
    assert original_html in (recorder.run_dir / "report.html").read_text(encoding="utf-8")
    assert load_run(recorder.run_dir)["manifest"]["status"] == "blocked"
