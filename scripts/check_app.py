"""Check measured gradient views offline; HTTP requests are forbidden in this QA."""
from pathlib import Path
import argparse
import json
import re
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(tempfile.gettempdir()) / "ai-buckingham-checks" / "web_checks.json")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    from streamlit.testing.v1 import AppTest
    checks = []
    def record(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"check": name, "passed": True})
    with patch("httpx.Client.send", side_effect=AssertionError("Network is forbidden in offline UI checks")), patch("httpx.AsyncClient.send", side_effect=AssertionError("Network is forbidden in offline UI checks")):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=args.timeout).run()
        record("Overview has no exception", not app.exception)
        metadata = json.loads((ROOT / "docs/paper_cases.json").read_text(encoding="utf-8"))
        gallery = app.main.container(key="overview-gallery")
        headings = [item.value.removeprefix("### ") for item in gallery.markdown
                    if item.value.startswith("### ")]
        record("Overview shows all ten cases in paper order",
               headings == [case["title"] for case in metadata["cases"]]
               and len(gallery.get("image")) == 10)
        paper_figures = app.main.expander[0]
        record("Figures 1 and 2 are available above the gallery in a collapsed panel",
               paper_figures.label == "Figures 1–2 · Similarity settings and workflow" and not paper_figures.proto.expanded
               and [tab.label for tab in paper_figures.tabs] == ["Similarity settings", "AI Buckingham workflow"]
               and all(len(tab.get("image")) == 1 for tab in paper_figures.tabs)
               and list(app.main).index(paper_figures) < list(app.main).index(gallery))
        record("Overview has no count dashboard or sample-count chart",
               not app.metric and not app.dataframe and not app.get("plotly_chart"))
        app.sidebar.radio[0].set_value("Case results").run()
        for case_id in ["blasius", "mhd", "cess_channel", "rough_pipe", "sandgrain", "velocity_scaling"]:
            app.selectbox(key="workflow_case").set_value(case_id).run()
            record(f"{case_id}: recorded gradient coordinates reach the view", not app.exception and len(app.selectbox(key="gradient_x").options) >= 2 and not any("Gradient coordinates and vectors were not recorded" in item.value for item in app.info))
        app.sidebar.radio[0].set_value("Case results").run()
        app.selectbox(key="workflow_case").set_value("pressure_gradient").run()
        baseline = [item.value for item in app.dataframe if "pressure_gradient_class" in item.value.columns]
        record("Pressure-gradient baseline exposes all 144 source rows", not app.exception and len(baseline) == 1 and len(baseline[0]) == 144)
        app.multiselect(key="pressure_baseline_classes").set_value(["ZPG", "APG"]).run()
        baseline = [item.value for item in app.dataframe if "pressure_gradient_class" in item.value.columns]
        record("Pressure-gradient baseline category filter retains 120 actual rows", not app.exception and len(baseline) == 1 and len(baseline[0]) == 120)
        app.selectbox(key="workflow_case").set_value("rough_pipe").run()
        result_html = "".join(item.proto.body for item in app.get("html")
                              if "Selected dimensionless relations" in item.proto.body)
        mi = [float(value) for value in re.findall(r"MI / information score:</strong> ([\d.]+)", result_html)]
        r2 = [float(value) for value in re.findall(r"Held-out R²:</strong> ([\d.]+)", result_html)]
        record("Rough-pipe page displays the verified final mutual information", not app.exception and sorted(round(value, 3) for value in mi) == [2.198, 3.085])
        record("Rough-pipe page displays the verified final validation scores", sorted(round(value, 3) for value in r2) == [0.974, 0.984])
        record("Case results uses completed cards with one level of expandable content",
               result_html and not app.json and not any("<details" in item.proto.body for item in app.get("html"))
               and not any(frame.value.shape[1] > 50 for frame in app.dataframe))
        app.sidebar.radio[0].set_value("Run analysis").run()
        record("Run analysis opens the ten saved cases by default",
               not app.exception and app.radio(key="analysis_action").value == "View saved results"
               and len(app.selectbox(key="saved_result_case").options) == 10)
        app.selectbox(key="saved_result_case").set_value("rough_pipe").run()
        app.button(key="open_saved_results").click().run()
        record("Opening a saved result navigates to its existing cards",
               not app.exception and app.radio(key="page_navigation").value == "Case results"
               and app.selectbox(key="workflow_case").value == "rough_pipe")
    report = {"offline_only": True, "network_requests_blocked": True, "total_checks": len(checks), "all_passed": True, "checks": checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "all_passed": True}))


if __name__ == "__main__":
    main()
