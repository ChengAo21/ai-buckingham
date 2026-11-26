"""Check measured gradient views offline; HTTP requests are forbidden in this QA."""
from pathlib import Path
import argparse
import json
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
        headings = [item.value.removeprefix("### ") for item in app.main
                    if item.type == "markdown" and item.value.startswith("### ")]
        record("Overview shows all ten cases in paper order",
               headings == [case["title"] for case in metadata["cases"]]
               and len(app.get("image")) == 10)
        record("Overview has no count dashboard or sample-count chart",
               not app.metric and not app.dataframe and not app.get("plotly_chart"))
        app.sidebar.radio[0].set_value("Numerical diagnostics").run()
        for case_id in ["blasius", "mhd", "cess_channel", "rough_pipe", "sandgrain", "velocity_scaling"]:
            app.selectbox(key="diagnostic_case").set_value(case_id).run()
            record(f"{case_id}: recorded gradient coordinates reach the view", not app.exception and len(app.selectbox(key="gradient_x").options) >= 2 and not any("Gradient coordinates and vectors were not recorded" in item.value for item in app.info))
        app.sidebar.radio[0].set_value("Case workflow").run()
        app.selectbox(key="workflow_case").set_value("pressure_gradient").run()
        baseline = [item.value for item in app.dataframe if "pressure_gradient_class" in item.value.columns]
        record("Pressure-gradient baseline exposes all 144 source rows", not app.exception and len(baseline) == 1 and len(baseline[0]) == 144)
        app.multiselect(key="pressure_baseline_classes").set_value(["ZPG", "APG"]).run()
        baseline = [item.value for item in app.dataframe if "pressure_gradient_class" in item.value.columns]
        record("Pressure-gradient baseline category filter retains 120 actual rows", not app.exception and len(baseline) == 1 and len(baseline[0]) == 120)
        app.selectbox(key="workflow_case").set_value("rough_pipe").run()
        mi = [float(metric.value) for metric in app.metric if metric.label == "Mutual information"]
        r2 = [float(metric.value) for metric in app.metric if metric.label == "Test R²"]
        record("Rough-pipe page displays the verified final mutual information", not app.exception and sorted(round(value, 3) for value in mi) == [2.198, 3.085])
        record("Rough-pipe page displays the verified final validation scores", sorted(round(value, 3) for value in r2) == [0.974, 0.984])
        app.sidebar.radio[0].set_value("Run analysis").run()
        metrics = {metric.label: str(metric.value) for metric in app.metric}
        record("Run analysis retains saved-run status and access",
               not app.exception and metrics.get("Completed runs") == "10"
               and metrics.get("Running runs") == "0" and metrics.get("Failed / stopped") == "0"
               and len(app.selectbox(key="saved_run_selection").options) == 10)
    report = {"offline_only": True, "network_requests_blocked": True, "total_checks": len(checks), "all_passed": True, "checks": checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "all_passed": True}))


if __name__ == "__main__":
    main()
