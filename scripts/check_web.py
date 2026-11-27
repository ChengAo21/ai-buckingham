"""Exercise the five result-explorer pages without model or network access.

Run from any working directory with the app dependencies installed. The checks
read the included publication snapshot, change real Streamlit controls, and
forbid both HTTPX requests and starting a live-analysis subprocess. They never
modify scientific results. Use --report to save a local verification report.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import re
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(tempfile.gettempdir()) / "ai-buckingham-checks" / "web_smoke.json")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    from streamlit.testing.v1 import AppTest
    import numpy as np
    import pandas as pd
    from ai_buckingham.web.data import read_json, reproduction, runs
    from ai_buckingham.web import run_page

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

    def plot_spec(key):
        plot = next(item for item in app.get("plotly_chart") if item.proto.id.endswith("-" + key))
        return json.loads(plot.proto.spec)

    def plot_array(value):
        if isinstance(value, dict) and "bdata" in value:
            values = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
            shape = tuple(int(part) for part in value.get("shape", str(len(values))).split(","))
            return values.reshape(shape)
        return np.asarray(value)

    def displayed_memberships():
        return {int(row[0]): int(trace["name"])
                for trace in plot_spec("candidate_partition_scatter")["data"]
                for row in plot_array(trace["customdata"])}

    def axis_checkbox(axis):
        return next(widget for widget in app.checkbox if widget.label == f"Logarithmic {axis}-axis")

    def result_html():
        return html.unescape("".join(item.proto.body for item in app.get("html")
                                     if "Selected dimensionless relations" in item.proto.body))

    def verify_collapse_axes(name, expected):
        case_id = app.selectbox(key="workflow_case").value
        path = Path(select_by_label("Scope").value)
        panel = path.stem.removesuffix("_collapse")
        key = f"case_collapse_{case_id}_{path.parents[1].name}_{panel}"
        plot = next(element for element in app.get("plotly_chart") if element.proto.id.endswith("-" + key))
        layout = json.loads(plot.proto.spec)["layout"]
        actual = (layout["xaxis"]["type"], layout["yaxis"]["type"])
        controls = (axis_checkbox("x").value, axis_checkbox("y").value)
        record(name, actual == tuple(expected) and controls == tuple(axis == "log" for axis in expected),
               {"case": case_id, "panel": panel, "axes": actual, "controls": controls})

    def case_run_path(case_id):
        candidates = [p for p in runs(case_id) if read_json(p / "manifest.json").get("status") == "completed"]
        if len(candidates) != 1:
            raise AssertionError(f"The publication snapshot needs exactly one completed run for {case_id}; found {len(candidates)}")
        return candidates[0]

    try:
        with patch("httpx.Client.send", side_effect=AssertionError("HTTP is forbidden during offline web QA")) as sync_send, \
             patch("httpx.AsyncClient.send", side_effect=AssertionError("HTTP is forbidden during offline web QA")) as async_send, \
             patch("ai_buckingham.web.run_page.start_job", side_effect=AssertionError("Live analysis is forbidden during offline web QA")) as start_job, \
             patch("ai_buckingham.web.run_page.numerical_settings", wraps=run_page.numerical_settings) as numerical_form:
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=args.timeout).run()
            record("Overview renders the final publication snapshot")
            metadata = read_json(ROOT / "docs/paper_cases.json")
            paper_cases = {case["id"]: case for case in metadata["cases"]}
            gallery = app.main.container(key="overview-gallery")
            headings = [element.value.removeprefix("### ") for element in gallery.markdown
                        if element.value.startswith("### ")]
            categories = [element.value.removeprefix("## ") for element in gallery.markdown
                          if element.value.startswith("## ")]
            record("Overview shows all ten figures and paper case names in order",
                   headings == [case["title"] for case in metadata["cases"]]
                   and len(gallery.get("image")) == 10)
            rows = [block for block in gallery.get("flex_container")
                    if any(child.type == "column" for child in block.children.values())]
            record("Overview arranges ten case figures in five pairs",
                   len(rows) == 5 and all(len(row.columns) == 2 for row in rows)
                   and all(len(column.get("image")) == 1 for column in gallery.columns))
            record("Overview uses the paper's three similarity categories without a count dashboard",
                   categories == [group["title"] for group in metadata["groups"]]
                   and not app.metric and not app.dataframe and not app.get("plotly_chart"))

            navigate("Case results")
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
                panel = collapse_path.stem.removesuffix("_collapse")
                paper_case = paper_cases[case_id]
                axes = paper_case.get("collapse_axes_by_panel", {}).get(panel, paper_case["collapse_axes"])
                verify_collapse_axes(f"Collapse axes match the paper defaults independently: {case_id}/{panel}", axes)
                if case_id == "velocity_scaling":
                    saved = read_json(case_run_path(case_id) / "state.json")
                    output = saved["workflow_result"]["local_workflows"][0]["final_solution"]["output_dimensionless_formula"]
                    record("Type II response relation retains the complete recorded output rescaling",
                           output in result_html())
                saved = read_json(case_run_path(case_id) / "state.json")
                record(f"Paper case does not display unreviewed model interpretation: {case_id}",
                       not any(element.value == saved.get("final_report") for element in app.markdown)
                       and not app.get("iframe"))
                cards = html.unescape(" ".join(item.proto.body for item in app.get("html")))
                removed_copy = ("Restart stability", "Termination: fixed_partition",
                                "A candidate representation was optimized and checked",
                                "Physical exponents are display-normalized",
                                "Authoritative manual similarity_type",
                                "Screening is diagnostic only",
                                "Count selection, parsimony checks",
                                "scale-equivalent representations", "Program-compiled dimensional matrix",
                                "Counts are computed from this matrix", "Checker decisions are model judgments")
                record(f"Obsolete process text is absent from all visible case cards: {case_id}",
                       not any(text in cards for text in removed_copy)
                       and all(widget.label != "CMA-ES search" for widget in app.selectbox))

            app.selectbox(key="workflow_case").set_value("mhd").run()
            axis_checkbox("x").set_value(False).run()
            verify_collapse_axes("The x-axis control changes independently of the y-axis", ("linear", "log"))
            app.selectbox(key="workflow_case").set_value("velocity_scaling").run()
            verify_collapse_axes("Case switching applies both Type II paper defaults", paper_cases["velocity_scaling"]["collapse_axes"])
            axis_checkbox("y").set_value(True).run()
            verify_collapse_axes("The y-axis control remains usable after switching cases", ("log", "log"))
            app.selectbox(key="workflow_case").set_value("goldenfeld").run()
            verify_collapse_axes("Goldenfeld does not inherit logarithmic controls from another case", paper_cases["goldenfeld"]["collapse_axes"])
            app.selectbox(key="workflow_case").set_value("velocity_scaling").run()
            verify_collapse_axes("Returning to a case restores its own paper defaults", paper_cases["velocity_scaling"]["collapse_axes"])
            axis_checkbox("x").set_value(False).run()
            verify_collapse_axes("Manual x-axis changes remain available after returning", ("linear", "linear"))

            app.selectbox(key="workflow_case").set_value("rough_pipe").run()
            for panel in ["regime_0", "regime_1"]:
                select_by_label("Scope").set_value(case_run_path("rough_pipe") / "tables" / f"{panel}_collapse.csv").run()
                verify_collapse_axes(f"Rough-pipe panel uses its own paper axes: {panel}",
                                     paper_cases["rough_pipe"]["collapse_axes_by_panel"][panel])
                if panel == "regime_0":
                    axis_checkbox("x").set_value(False).run()
                    verify_collapse_axes("Rough-pipe scope controls can be changed independently", ("linear", "linear"))

            app.selectbox(key="workflow_case").set_value("sandgrain").run()
            app.multiselect(key="prediction_split").set_value(["test"]).run()
            predictions = frames_with("reference", "original_seven", "learned_groups", "split")
            record("Sandgrain prediction interaction selects the same 21 test observations",
                   len(predictions) == 1 and len(predictions[0]) == 21 and set(predictions[0]["split"]) == {"test"})
            saved = read_json(case_run_path("sandgrain") / "state.json")
            inputs = saved["workflow_result"]["local_workflows"][0]["final_solution"]["input_formulas"]
            record("Sandgrain displays both selected groups without a model-generated response law",
                   len(inputs) == 2 and all(value in result_html() for value in inputs)
                   and not any(element.value == saved["final_report"] for element in app.markdown))

            app.selectbox(key="workflow_case").set_value("rough_pipe").run()
            mi = [float(value) for value in re.findall(r"MI / information score:</strong> ([\d.]+)", result_html())]
            r2 = [float(value) for value in re.findall(r"Held-out R²:</strong> ([\d.]+)", result_html())]
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
            navigate("Case results")
            for case_id in ["mhd", "sandgrain", "cess_channel", "rough_pipe", "velocity_scaling"]:
                app.selectbox(key="workflow_case").set_value(case_id).run()
                record(f"Included case Checker event pairs expose exact ranked evidence: {case_id}",
                       bool(frames_with("entry_id", "lexical_score", "semantic_score", "hybrid_score")))

            navigate("Case results")
            for case_id in ["mhd", "pressure_gradient", "rayleigh_benard", "laser_keyhole", "blasius"]:
                app.selectbox(key="workflow_case").set_value(case_id).run()
                landscape = plot_spec("landscape_3d_surface")
                record(f"The paper MI landscape has one 3D surface and its full trajectory: {case_id}",
                       sum(trace["type"] == "surface" for trace in landscape["data"]) == 1
                       and any(trace["type"] == "scatter3d" for trace in landscape["data"])
                       and not app.select_slider)
                for key in ("mi_path_0", "epsilon_path_0"):
                    stages = plot_spec(key)["data"]
                    record(f"Stage results are labeled bars: {case_id}/{key}",
                           all(trace["type"] == "bar" and trace.get("text") for trace in stages))
                    record(f"Stage bars display saved uncertainty: {case_id}/{key}",
                           all(trace.get("error_y", {}).get("type", "data") == "data"
                               and len(plot_array(trace["error_y"]["array"])) == len(plot_array(trace["y"]))
                               for trace in stages))

            for case_id in ["blasius", "cess_channel", "rough_pipe"]:
                app.selectbox(key="workflow_case").set_value(case_id).run()
                candidate_path = Path(select_by_label("Candidate partition").value)
                expected_rows = {"blasius":196,"cess_channel":1000,"rough_pipe":1200}[case_id]
                actual = displayed_memberships()
                with np.load(candidate_path.parent / "coordinates.npz", allow_pickle=False) as arrays:
                    indices = arrays["valid_indices"]
                with np.load(candidate_path, allow_pickle=False) as arrays:
                    expected = dict(zip(indices.tolist(), arrays["labels"].tolist()))
                record(f"Switching cases keeps candidate K arrays in the active context: {case_id}",
                       len(actual) == expected_rows and actual == expected,
                       {"candidate_path": str(candidate_path.relative_to(ROOT)), "rows": expected_rows})
            row_labels = {row: displayed_memberships()[row] for row in [544, 1181]}
            record("Corrected rough-pipe pointwise labels reach the K=2 view", row_labels == {544: 1, 1181: 0},
                   {"csv_row_index_base": 0, "labels": row_labels})
            selected_candidate = Path(select_by_label("Candidate partition").value)
            select_by_label("Candidate partition").set_value(selected_candidate.with_name("k_4.npz")).run()
            memberships = displayed_memberships()
            record("Rough-pipe candidate selector exposes all 1200 K=4 point assignments",
                   len(memberships) == 1200 and len(set(memberships.values())) == 4)

            app.selectbox(key="workflow_case").set_value("goldenfeld").run()
            memberships = displayed_memberships()
            record("Goldenfeld candidate partition evidence renders independently",
                   len(memberships) == 414)

            navigate("Comparisons")
            record("Dimensional checking shows the paper's 600 trial summary",
                   next(item for item in app.metric if item.label == "Selected scheduled trials").value == "600"
                   and not app.dataframe and not app.json)
            app.multiselect(key="benchmark_cases").set_value(["MHD"]).run()
            record("Benchmark physical-system filter selects 60 scheduled MHD trials",
                   next(item for item in app.metric if item.label == "Selected scheduled trials").value == "60")
            app.multiselect(key="benchmark_variants").set_value(["original"]).run()
            record("Benchmark wording filter selects 30 original-prompt MHD trials",
                   next(item for item in app.metric if item.label == "Selected scheduled trials").value == "30")

            app.radio(key="comparison_page").set_value("Cess regime comparison").run()
            record("Cess comparison retains the paper figure and interactive partitions",
                   len(app.get("image")) == 1 and app.selectbox(key="cess_m").value == 3
                   and len(app.get("plotly_chart")) == 2 and not app.dataframe and not app.json)
            before = [trace["name"] for trace in plot_spec("cess_regional")["data"]]
            app.selectbox(key="cess_m").set_value(5).run()
            after = [trace["name"] for trace in plot_spec("cess_regional")["data"]]
            record("Cess region selection updates its saved M=5 classification",
                   plot_spec("cess_regional")["layout"]["title"]["text"] == "IT–π, M = 5"
                   and before != after and "K = 3" in plot_spec("cess_gradient")["layout"]["title"]["text"])

            app.radio(key="comparison_page").set_value("Velocity transformations").run()
            record("Velocity profiles, RMS comparison and paper figure render", len(app.get("image")) == 1)
            app.multiselect(key="velocity_profiles").set_value([1, 6]).run()
            profiles = plot_spec("velocity_profiles_plot")["data"]
            record("Velocity profile selection shows the selected curves",
                   len(profiles) == 9 and all(trace["line"]["width"] >= 2.8 for trace in profiles)
                   and not app.dataframe)
            app.radio(key="velocity_interval").set_value("30–200").run()
            record("Velocity RMS interval control selects the outer interval",
                   app.radio(key="velocity_interval").value == "30–200")

            navigate("Run analysis")
            record("Run analysis defaults to the ten saved physical systems",
                   app.radio(key="analysis_action").value == "View saved results"
                   and len(app.selectbox(key="saved_result_case").options) == 10)
            app.selectbox(key="saved_result_case").set_value("mhd").run()
            app.button(key="open_saved_results").click().run()
            record("Saved-result access opens the existing MHD cards without execution",
                   app.radio(key="page_navigation").value == "Case results"
                   and app.selectbox(key="workflow_case").value == "mhd" and start_job.call_count == 0)
            navigate("Run analysis")
            app.radio(key="analysis_action").set_value("Run new analysis").run()
            record("Supplied-case analysis form renders without starting execution")
            for case_id in ["blasius", "rough_pipe"]:
                app.selectbox(key="live_case").set_value(case_id).run()
                expected = read_json(ROOT / "configs" / "cases" / f"{case_id}.json")
                prompt = next(widget.value for widget in app.text_area if widget.label == "Problem description")
                preview = numerical_form.call_args.args[0]
                record(f"Live case switch keeps the problem and submitted configuration consistent: {case_id}",
                       prompt == expected["prompt"] and preview == expected
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
