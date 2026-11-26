"""Exercise the six evidence-explorer pages without model or network access.

Run from any working directory with the app dependencies installed. The checks
read the included publication snapshot, change real Streamlit controls, and
forbid both HTTPX requests and starting a live-analysis subprocess. They never
modify scientific results. The default report is an ignored runtime artifact;
pass --report docs/validation/web_smoke.json to refresh the release QA record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "generated" / "web_smoke.json")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    from streamlit.testing.v1 import AppTest
    import pandas as pd
    from ai_buckingham.web.data import read_json, reproduction, runs

    checks = []
    failure = None
    app = None
    network_calls = 0
    live_job_starts = 0

    def record(name, condition=True, details=None):
        errors = [item.message for item in app.exception] if app is not None else []
        passed = bool(condition) and not errors
        item = {"check": name, "passed": passed, "errors": errors}
        if details is not None:
            item["details"] = details
        checks.append(item)
        print(("PASS " if passed else "FAIL ") + name, flush=True)
        if not passed:
            raise AssertionError(name + (": " + "; ".join(errors) if errors else ""))

    def navigate(page):
        app.radio(key="page_navigation").set_value(page).run(timeout=args.timeout)

    def frames_with(*columns):
        return [element.value for element in app.dataframe if set(columns).issubset(element.value.columns)]

    def select_by_label(label):
        return next(widget for widget in app.selectbox if widget.label == label)

    def case_run_path(case_id):
        candidates = [p for p in runs(case_id) if read_json(p / "manifest.json").get("status") == "completed"]
        if len(candidates) != 1:
            raise AssertionError(f"The publication snapshot needs exactly one completed run for {case_id}; found {len(candidates)}")
        return candidates[0]

    try:
        with patch("httpx.Client.send", side_effect=AssertionError("HTTP is forbidden during offline web QA")) as sync_send, \
             patch("httpx.AsyncClient.send", side_effect=AssertionError("HTTP is forbidden during offline web QA")) as async_send, \
             patch("ai_buckingham.web.run_page.start_job", side_effect=AssertionError("Live analysis is forbidden during offline web QA")) as start_job:
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=args.timeout).run()
            metrics = {m.label: str(m.value) for m in app.metric}
            record("Overview renders the final publication snapshot")
            record("Overview emphasizes paper coverage and observations",
                   metrics.get("Physical systems") == "10"
                   and metrics.get("Observations") == "4,926"
                   and metrics.get("Paper case coverage") == "10/10"
                   and metrics.get("Dimensional-checking trials") == "600"
                   and not {"Completed runs", "Running runs", "Failed / stopped"}.intersection(metrics),
                   metrics)
            inventory = frames_with("Physical system", "Observations")
            record("Overview presents research results and a case inventory without run status",
                   any(element.value == "### Selected results" for element in app.markdown)
                   and len(inventory) == 1 and len(inventory[0]) == 10
                   and not {"Completed runs", "Running runs", "Failed / stopped runs"}.intersection(inventory[0].columns))

            navigate("Case workflow")
            case_ids = sorted(p.stem for p in (ROOT / "configs" / "cases").glob("*.json"))
            for case_id in case_ids:
                app.selectbox(key="workflow_case").set_value(case_id).run()
                collapse_path = Path(select_by_label("Scope").value)
                expected = pd.read_csv(collapse_path)
                point_tables = frames_with("csv_row", "response_transformed")
                record(f"Case workflow and displayed collapse belong to the active case/run: {case_id}",
                       collapse_path.is_relative_to(case_run_path(case_id))
                       and any(frame.equals(expected) for frame in point_tables),
                       {"selected_collapse": str(collapse_path.relative_to(ROOT)), "rows": len(expected)})
                if case_id == "velocity_scaling":
                    saved = read_json(case_run_path(case_id) / "state.json")
                    output = saved["workflow_result"]["local_workflows"][0]["final_solution"]["output_dimensionless_formula"]
                    record("Type II response relation retains the complete recorded output rescaling",
                           any(element.value == output for element in app.code)
                           and any("Π_out ≈ F(Π1)" in element.value for element in app.caption))
            app.selectbox(key="workflow_case").set_value("blasius").run()
            app.toggle(key="show_complete_card").set_value(True).run()
            record("Saved complete HTML card is displayed on explicit toggle", len(app.get("iframe")) == 1)
            app.toggle(key="show_complete_card").set_value(False).run()

            app.selectbox(key="workflow_case").set_value("sandgrain").run()
            app.multiselect(key="prediction_split").set_value(["test"]).run()
            predictions = frames_with("reference", "original_seven", "learned_groups", "split")
            record("Sandgrain prediction interaction selects the same 21 test observations",
                   len(predictions) == 1 and len(predictions[0]) == 21 and set(predictions[0]["split"]) == {"test"})
            saved = read_json(case_run_path("sandgrain") / "state.json")
            commentary = next(element for element in app.expander if element.label == "Archived model commentary")
            record("Sandgrain shows a two-input response and preserves model commentary in its closed archive",
                   any("Π_out ≈ F(Π1, Π2)" in element.value for element in app.caption)
                   and any(element.value == saved["final_report"] for element in commentary.markdown)
                   and not commentary.proto.expanded)

            app.selectbox(key="workflow_case").set_value("rough_pipe").run()
            mi = [float(m.value) for m in app.metric if m.label == "Mutual information"]
            r2 = [float(m.value) for m in app.metric if m.label == "Conditional held-out R²"]
            record("Corrected rough-pipe results reach the case metrics",
                   sorted(round(v, 3) for v in mi) == [2.198, 3.085]
                   and sorted(round(v, 3) for v in r2) == [0.974, 0.984],
                   {"mutual_information": mi, "conditional_test_r2": r2})

            app.selectbox(key="workflow_case").set_value("pressure_gradient").run()
            baseline = frames_with("Re_delta", "pressure_gradient_class")
            record("Pressure-gradient baseline exposes 144 original observations",
                   len(baseline) == 1 and len(baseline[0]) == 144)
            app.multiselect(key="pressure_baseline_classes").set_value(["ZPG", "APG"]).run()
            baseline = frames_with("Re_delta", "pressure_gradient_class")
            record("Pressure-gradient class interaction selects 120 ZPG/APG observations",
                   len(baseline) == 1 and len(baseline[0]) == 120 and set(baseline[0]["pressure_gradient_class"]) == {"ZPG", "APG"})

            navigate("Retrieval evidence")
            record("Archived variable-level retrieval evidence renders",
                   bool(frames_with("entry_id", "lexical_score", "semantic_score", "hybrid_score")))
            app.selectbox(key="rag_model").set_value("GPT-4o-mini").run()
            record("Changing the Analyzer model also changes the actual archived trial source",
                   Path(select_by_label("Trial identifier").value).parent.name == "GPT-4o-mini")
            app.selectbox(key="rag_case").set_value("MHD").run()
            archive = reproduction("rag_ablation") / "trials" / "GPT-4o-mini"
            revised_trial = next(p for p in sorted(archive.glob("MHD_*"))
                                 if len(read_json(p / "on.json").get("checker_rounds", [])) > 1)
            select_by_label("Trial identifier").set_value(revised_trial).run()
            app.selectbox(key="rag_round").set_value(1).run()
            record("Archived revision trial exposes its actual second checking round",
                   app.selectbox(key="rag_round").value == 1,
                   {"trial": revised_trial.name, "checker_round": 2})
            if "eta" in app.selectbox(key="rag_variable").options:
                app.selectbox(key="rag_variable").set_value("eta").run()
            record("Archived quantity switch retains ranked evidence and score components",
                   bool(frames_with("entry_id", "lexical_score", "semantic_score", "hybrid_score")))
            app.radio(key="rag_source").set_value("Included case run").run()
            for case_id in ["mhd", "sandgrain", "cess_channel", "rough_pipe", "velocity_scaling"]:
                app.selectbox(key="rag_case_run").set_value(case_run_path(case_id)).run()
                record(f"Included case Checker event pairs expose exact ranked evidence: {case_id}",
                       bool(frames_with("entry_id", "lexical_score", "semantic_score", "hybrid_score")))

            navigate("Numerical diagnostics")
            for case_id in ["mhd", "pressure_gradient", "rayleigh_benard", "laser_keyhole", "blasius"]:
                app.selectbox(key="diagnostic_case").set_value(case_id).run()
                record(f"Exported MI grid and genuine generation history render: {case_id}",
                       len(app.select_slider(key="landscape_generation").options) > 1)
                generation = int(app.select_slider(key="landscape_generation").options[0])
                app.select_slider(key="landscape_generation").set_value(generation).run()
                record(f"MI history slider reveals the first recorded generation: {case_id}",
                       int(app.select_slider(key="landscape_generation").value) == generation)
            app.radio(key="landscape_rendering").set_value("3D surface").run()
            record("Exported MI grid supports the interactive 3D surface")
            app.radio(key="landscape_rendering").set_value("Contour").run()

            for case_id in ["blasius", "cess_channel", "rough_pipe"]:
                app.selectbox(key="diagnostic_case").set_value(case_id).run()
                candidate_path = Path(select_by_label("Candidate partition").value)
                source_path = Path(select_by_label("Pointwise partition evidence").value)
                expected_rows = {"blasius":196,"cess_channel":1000,"rough_pipe":1200}[case_id]
                candidates = frames_with("z_x", "z_y", "Regime", "csv_row")
                record(f"Switching cases keeps candidate K arrays in the active context: {case_id}",
                       candidate_path.parent == source_path.parent
                       and len(candidates) == 1 and len(candidates[0]) == expected_rows,
                       {"candidate_path": str(candidate_path.relative_to(ROOT)), "rows": expected_rows})
            memberships = frames_with("z_x", "z_y", "Regime", "csv_row")
            member_frame = memberships[0].set_index("csv_row")
            row_labels = {row: int(member_frame.loc[row, "Regime"]) for row in [544, 1181]}
            record("Corrected rough-pipe pointwise labels reach the K=2 view", row_labels == {544: 1, 1181: 0},
                   {"csv_row_index_base": 0, "labels": row_labels})
            selected_candidate = Path(select_by_label("Candidate partition").value)
            select_by_label("Candidate partition").set_value(selected_candidate.with_name("k_4.npz")).run()
            memberships = frames_with("z_x", "z_y", "Regime", "csv_row")
            record("Rough-pipe candidate selector exposes all 1200 K=4 point assignments",
                   len(memberships) == 1 and len(memberships[0]) == 1200 and memberships[0]["Regime"].nunique() == 4)

            app.selectbox(key="diagnostic_case").set_value("goldenfeld").run()
            memberships = frames_with("z_x", "z_y", "Regime", "csv_row")
            record("Goldenfeld candidate partition evidence renders independently",
                   len(memberships) == 1 and len(memberships[0]) == 414)

            app.selectbox(key="diagnostic_case").set_value("mhd").run()
            generation = int(app.select_slider(key="trace_generation").options[0])
            app.select_slider(key="trace_generation").set_value(generation).run()
            record("Actual CMA trace slider exposes a recorded first generation",
                   int(app.select_slider(key="trace_generation").value) == generation)
            searches = app.selectbox(key="trace_search").options
            app.selectbox(key="trace_search").set_value(1).run()
            record("CMA search selector switches between separately recorded searches", len(searches) > 1)

            navigate("Comparisons")
            trials = frames_with("pair_id", "validated_result")
            rounds = frames_with("pair_id", "checker_round", "replayed_gate")
            record("Benchmark view exposes all 600 scored trials and 730 checking rounds",
                   len(trials) == 1 and len(trials[0]) == 600 and len(rounds) == 1 and len(rounds[0]) == 730)
            app.multiselect(key="benchmark_cases").set_value(["MHD"]).run()
            trials = frames_with("pair_id", "validated_result")
            record("Benchmark physical-system filter selects 60 scheduled MHD trials",
                   len(trials) == 1 and len(trials[0]) == 60)
            app.multiselect(key="benchmark_variants").set_value(["original"]).run()
            trials = frames_with("pair_id", "validated_result")
            record("Benchmark wording filter selects 30 original-prompt MHD trials",
                   len(trials) == 1 and len(trials[0]) == 30)

            app.radio(key="comparison_page").set_value("Cess regime comparison").run()
            record("Historical Cess relevance/gradient comparison renders")
            app.selectbox(key="cess_m").set_value(5).run()
            regional = frames_with("conditioning", "M", "seed", "category", "n")
            record("Cess M=5 selects its five recorded regional scores",
                   len(regional) == 1 and len(regional[0]) == 5 and regional[0]["M"].eq(5).all())
            app.selectbox(key="cess_seed").set_value(46).run()
            app.selectbox(key="cess_condition").set_value("raw").run()
            regional = frames_with("conditioning", "M", "seed", "category", "n")
            record("Cess seed and conditioning controls select the matching recorded calculation",
                   len(regional) == 1 and regional[0]["seed"].eq(46).all() and regional[0]["conditioning"].eq("raw").all())

            app.radio(key="comparison_page").set_value("Velocity transformations").run()
            record("Historical velocity profiles and RMS comparison render")
            app.multiselect(key="velocity_profiles").set_value([1, 6]).run()
            profiles = frames_with("case", "y_plus", "U_plus", "x_TL", "U_TL", "x_AI", "U_AI")
            record("Velocity profile selection exposes only the selected native observations",
                   len(profiles) == 1 and set(profiles[0]["case"]) == {1, 6})
            app.radio(key="velocity_interval").set_value("30–200").run()
            record("Velocity RMS interval control selects the outer interval",
                   app.radio(key="velocity_interval").value == "30–200")

            navigate("Run analysis")
            record("Supplied-case analysis form renders without starting execution")
            metrics = {m.label: str(m.value) for m in app.metric}
            status_panel = next(element for element in app.expander if element.label == "Saved runs and status")
            record("Run analysis retains saved-run status and access in a closed panel",
                   metrics.get("Completed runs") == "10"
                   and metrics.get("Running runs") == "0" and metrics.get("Failed / stopped") == "0"
                   and len(app.selectbox(key="saved_run_selection").options) == 10
                   and not status_panel.proto.expanded,
                   metrics)
            for case_id in ["blasius", "rough_pipe"]:
                app.selectbox(key="live_case").set_value(case_id).run()
                expected = read_json(ROOT / "configs" / "cases" / f"{case_id}.json")
                prompt = next(widget.value for widget in app.text_area if widget.label == "Problem description")
                previews = [json.loads(widget.value) for widget in app.json]
                preview = next(value for value in previews if isinstance(value, dict)
                               and value.get("case_id") == case_id and "prompt" in value)
                record(f"Live case switch keeps the problem and submitted configuration consistent: {case_id}",
                       prompt == expected["prompt"] and preview.get("prompt") == expected["prompt"]
                       and preview.get("numerical_threads") == expected["numerical_threads"]
                       and (case_id != "rough_pipe" or preview.get("numerical_threads") == 20),
                       {"case_id": case_id, "prompt_matches_selected_case": prompt == expected["prompt"],
                        "submitted_numerical_threads": preview.get("numerical_threads")})
            app.radio(key="live_input_mode").set_value("My CSV and problem").run()
            record("Custom CSV analysis form renders without starting execution")
            submit = next(button for button in app.button if button.label == "Start LangGraph analysis")
            submit.click().run()
            record("Incomplete analysis submission reports validation and starts no job",
                   len(app.error) >= 1 and start_job.call_count == 0)
            network_calls = sync_send.call_count + async_send.call_count
            live_job_starts = start_job.call_count
            record("All page interactions issued zero HTTP requests and zero live jobs",
                   network_calls == 0 and live_job_starts == 0)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if not checks or checks[-1]["passed"]:
            checks.append({"check": "Smoke execution completed", "passed": False,
                           "errors": [str(error)]})
        print(f"FAIL {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "network_requests_blocked": True,
        "live_analysis_start_blocked": True,
        "observed_http_requests": network_calls,
        "observed_live_job_starts": live_job_starts,
        "script": "scripts/check_web.py", "script_sha256": sha256(Path(__file__)),
        "app_sha256": sha256(ROOT / "streamlit_app.py"),
        "result_index_sha256": sha256(ROOT / "results" / "index.json"),
        "expected_publication_snapshot": {"completed_cases": 10, "running_runs": 0, "failed_runs": 0},
        "total_checks": len(checks),
        "all_passed": failure is None and all(item["passed"] for item in checks),
        "checks": checks,
    }
    if failure is not None:
        report["failure"] = failure
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "all_passed": report["all_passed"],
                      "report": str(args.report.resolve())}), flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
