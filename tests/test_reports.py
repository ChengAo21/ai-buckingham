"""Reader reports preserve selected runs and recorded terminal stage cards."""
import base64
import hashlib
import json
from pathlib import Path

import pytest

from ai_buckingham.reports import _CardParser, export_report, render_report, report_figures, report_sections


ROOT = Path(__file__).resolve().parents[1]


def _card(title, status, body):
    return {"text/html": '<section class="daa-card"><h3 class="daa-card-title">'
            + title + '</h3><span class="daa-status">' + status + "</span>" + body + "</section>"}


def _run(tmp_path, cards):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/display_bundles.json").write_text(json.dumps(cards), encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"case_id":"custom", "status":"failed"}', encoding="utf-8")
    return tmp_path


def test_last_stage_snapshot_retains_a_failure_and_flattens_scientific_details(tmp_path):
    run = _run(tmp_path, [
        _card("Problem formulation", "Running", "old snapshot"),
        _card("Problem formulation", "Blocked", '<details><summary>Input ledger</summary>kept</details>'
              '<details><summary>Raw Analyzer and compiled contract</summary>raw dump</details>'),
        _card("Run provenance and recorded costs", "completed", "HTTP record"),
    ])
    sections = report_sections(run)
    assert len(sections) == 1
    body = sections[0]["html"]
    assert "Blocked" in body and "kept" in body
    assert "old snapshot" not in body and "raw dump" not in body
    assert "HTTP record" not in body and "<details" not in body


def test_render_report_reads_only_selected_run_and_preserves_source_files(tmp_path):
    run = _run(tmp_path, [_card("Selected dimensionless relations", "success", "custom formula")])
    (run / "plots").mkdir()
    image_bytes = b"selected-run-image"
    (run / "plots/selected.png").write_bytes(image_bytes)
    before = {path.relative_to(run): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in run.rglob("*") if path.is_file()}
    page = render_report(run)
    assert "custom formula" in page
    assert base64.b64encode(image_bytes).decode() in page
    assert "Magnetohydrodynamic" not in page
    after = {path.relative_to(run): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in run.rglob("*") if path.is_file()}
    assert after == before


def test_all_cards_in_a_bundle_are_available_and_outer_folds_do_not_nest(tmp_path):
    one = _card("Problem formulation", "Complete", "problem")
    two = _card("Analysis protocol", "Complete", '<details><summary>Method</summary>settings</details>')
    run = _run(tmp_path, [{"text/html": one["text/html"] + two["text/html"]}])
    sections = report_sections(run)
    assert [section["id"] for section in sections] == ["formulation", "planning"]
    page = render_report(run)
    assert page.count('<details class="daa-report-stage"') == 2
    assert page.count("</details>") == 2
    assert "settings" in page


def test_partial_run_retains_figures_from_legacy_directory(tmp_path):
    run = _run(tmp_path, [])
    (run / "figures").mkdir()
    image_bytes = b"partial-legacy-image"
    (run / "figures/partial.png").write_bytes(image_bytes)
    page = render_report(run)
    assert base64.b64encode(image_bytes).decode() in page


def test_inline_captures_are_retained_but_not_repeated_beside_saved_plots(tmp_path):
    run = _run(tmp_path, [])
    folder = run / "plots"
    folder.mkdir()
    original = folder / "custom_global_evidence.png"
    capture = folder / "display_007.png"
    original.write_bytes(b"saved-plot")
    capture.write_bytes(b"inline-copy")
    assert report_figures(run) == [original]
    page = render_report(run)
    assert base64.b64encode(original.read_bytes()).decode() in page
    assert base64.b64encode(capture.read_bytes()).decode() not in page
    assert capture.read_bytes() == b"inline-copy"


def test_inline_only_run_still_shows_its_captured_plot(tmp_path):
    run = _run(tmp_path, [])
    (run / "plots").mkdir()
    capture = run / "plots/display_003.png"
    capture.write_bytes(b"only-inline-plot")
    assert report_figures(run) == [capture]
    assert base64.b64encode(capture.read_bytes()).decode() in render_report(run)


def test_derived_report_is_repeatable_for_partial_runs_and_protects_raw_report(tmp_path):
    run = _run(tmp_path, [])
    original = run / "report.html"
    original.write_text("original record", encoding="utf-8")
    destination = run / "result.html"
    export_report(run, destination)
    first = destination.read_bytes()
    export_report(run, destination)
    assert destination.read_bytes() == first
    assert "unavailable" in destination.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        export_report(run, original)
    assert original.read_text(encoding="utf-8") == "original record"
    assert not list(run.glob("*.tmp"))


def test_terminal_failure_is_visible_above_previously_completed_cards(tmp_path):
    run = _run(tmp_path, [_card("Selected dimensionless relations", "success", "completed earlier")])
    page = render_report(run)
    assert "Run status: failed" in page
    assert page.index("Run status: failed") < page.index("completed earlier")
    (run / "manifest.json").write_text('{"case_id":"custom", "status":"completed"}', encoding="utf-8")
    assert "Run status:" not in render_report(run)


@pytest.mark.parametrize("case", json.loads((ROOT / "docs/paper_cases.json").read_text(encoding="utf-8"))["cases"],
                         ids=lambda case: case["id"])
def test_published_reader_report_preserves_final_card_values_for_all_cases(case):
    index = json.loads((ROOT / "results/index.json").read_text(encoding="utf-8"))
    entry = next(item for item in index["cases"] if item["case_id"] == case["id"])
    selected = next(run for run in entry["runs"] if run["run_id"] == entry["selected_run"])
    run_dir = ROOT / selected["directory"]
    sections = report_sections(run_dir)
    assert [section["id"] for section in sections] == ["results", "formulation", "dimensions", "planning", "regimes", "learning"]
    content = "".join(section["html"] for section in sections)
    assert "Run provenance and recorded costs" not in content
    assert "<details" not in content
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    for local in state["workflow_result"]["local_workflows"]:
        for formula in local["final_solution"]["input_formulas"]:
            assert formula.replace("&", "&amp;").replace('"', "&quot;") in sections[0]["html"]

    parsed = _CardParser(content).root
    planning = next(card for card in parsed.find_all("daa-card")
                    if card.find("daa-card-title").text() == "Analysis protocol")
    plan_metrics = planning.find("daa-metrics").find_all("daa-metric")
    assert [metric.find("daa-metric-key").text() for metric in plan_metrics] == [
        "Protocol", "Similarity", "Output optimization", "Similarity decision · relative MI threshold",
        "Input constraints", "Output constraints", "Manual overrides",
    ]
    screens = [card for card in parsed.find_all("daa-card")
               if card.find("daa-card-title").text().startswith("Similarity-class screening")]
    assert len(screens) == sum(local.get("similarity_screening", {}).get("status") == "success"
                               for local in state["workflow_result"]["local_workflows"])
    for card in screens:
        assert len(card.find("daa-metrics").find_all("daa-metric")) == 10
    for unwanted in (
        "Physical exponents are display-normalized", "Restart stability",
        "A candidate representation was optimized", "Termination:",
        "Authoritative manual", "pending numerical screen", "Replanning budget",
        "Type I was selected because", "Type II was selected because",
        "MI-only decision", "Screening is diagnostic only", "Auto screening ignores",
        "Count selection, parsimony checks", "scale-equivalent representations",
        "Final relations are copied directly",
        "Program-compiled dimensional matrix", "Counts are computed from this matrix", "structured_analyzer",
        "Checker decisions are model judgments",
    ):
        assert unwanted not in content
    assert "Dimensional matrix" in content


def test_archived_metric_promotion_preserves_values_and_formulas(tmp_path):
    metrics = ''.join('<div class="daa-metric"><div class="daa-metric-key">' + key
                      + '</div><div class="daa-metric-value">' + value + '</div></div>'
                      for key, value in [("Protocol", "custom"), ("Input constraints", "{x: 2}")])
    body = '<div class="daa-metrics">' + metrics + '</div><div class="daa-formula">x**2</div>'
    body += '<div class="daa-note">Authoritative manual override</div>'
    body += ('<details class="daa-details"><summary>Methods and diagnostics</summary><table>'
             '<tr><td>Output constraints</td><td>{y: -1}</td></tr>'
             '<tr><td>Manual overrides</td><td>4</td></tr>'
             '<tr><td>Replanning budget</td><td>99</td></tr></table></details>')
    run = _run(tmp_path, [_card("Analysis protocol", "Complete", body)])
    content = report_sections(run)[0]["html"]
    metrics = _CardParser(content).root.find_all("daa-metric")
    values = {metric.find("daa-metric-key").text(): metric.find("daa-metric-value").text()
              for metric in metrics}
    assert values == {"Protocol": "custom", "Input constraints": "{x: 2}",
                      "Output constraints": "{y: -1}", "Manual overrides": "4"}
    assert "x**2" in content and "Replanning budget" not in content
    assert "Authoritative manual" not in content and "Methods and diagnostics" not in content


def test_restart_stability_removal_keeps_other_diagnostic_rows(tmp_path):
    body = ('<details><summary>Methods and diagnostics</summary><table>'
            '<tr><td>Restart stability</td><td>1</td></tr>'
            '<tr><td>Additional result</td><td>2.718</td></tr></table></details>')
    run = _run(tmp_path, [_card("global · candidate k=1", "Complete", body)])
    content = report_sections(run)[0]["html"]
    assert "Restart stability" not in content and "Additional result" in content and "2.718" in content


def test_future_cards_promote_the_retained_settings(monkeypatch):
    from ai_buckingham import presentation

    captured = []
    monkeypatch.setattr(presentation, "_inject_agent_editorial_styles", lambda: None)
    monkeypatch.setattr(presentation, "_workflow_ui_enabled", lambda _: True)
    monkeypatch.setattr(presentation, "display_workflow_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation, "display", lambda output: captured.append(output.data))
    presentation.display_compiled_plan(
        {"rationale": "old rationale"},
        {"template": "auto_mechanism_adaptive", "local_workflow": {"similarity_type": "auto",
         "similarity_screening_max_relative_mi_change": .15, "fixed_in_power": {"x": 2},
         "fixed_out_power": {"y": -1}}, "max_replans": 2},
        {"manual_overrides": {"one": 1, "two": 2}},
    )
    planning = _CardParser(captured.pop()).root
    assert len(planning.find_all("daa-metric")) == 7
    values = {metric.find("daa-metric-key").text(): metric.find("daa-metric-value").text()
              for metric in planning.find_all("daa-metric")}
    assert json.loads(values["Input constraints"]) == {"x": 2}
    assert json.loads(values["Output constraints"]) == {"y": -1}
    assert "old rationale" not in planning.text() and "Replanning budget" not in planning.text()

    presentation.display_similarity_screening_summary({
        "status": "success", "scope_id": "global", "selected_similarity_type": "first_kind",
        "fixed_output_mutual_information": 3.14, "joint_output_mutual_information": 3.16,
        "relative_mi_loss_without_output_scaling": .01, "max_relative_mi_change": .15,
        "screening_max_cosine_similarity": .7, "reason": "old explanatory reason",
    })
    screening_html = captured.pop()
    screening = _CardParser(screening_html).root
    assert len(screening.find_all("daa-metric")) == 10
    assert "3.14" in screening.text() and "3.16" in screening.text()
    assert "old explanatory reason" not in screening_html and "Methods and diagnostics" not in screening_html
