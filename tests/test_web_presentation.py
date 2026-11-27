"""Presentation must keep saved MI values and separate titles from legends."""
from unittest.mock import patch
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from ai_buckingham.web.components import chart
from ai_buckingham.web.diagnostics import landscape_figure, mi_view, stage_comparison


def test_3d_landscape_preserves_signed_grid_values_and_missing_cells():
    frame = pd.DataFrame({"coefficient_x": [0, 2, 0, 2], "coefficient_y": [-1, -1, 1, 1],
                          "mutual_information": [-.2, .4, np.nan, 1.8]})
    figure = landscape_figure("mhd", frame, pd.DataFrame(), {})
    trace = figure.data[0]
    assert trace.type == "surface"
    np.testing.assert_equal(np.asarray(trace.z), [[-.2, .4], [np.nan, 1.8]])
    np.testing.assert_equal(np.asarray(trace.x), [0, 2])
    np.testing.assert_equal(np.asarray(trace.y), [-1, 1])
    assert trace.colorscale[0][1] == "#440154"
    assert trace.colorscale[-1][1] == "#fde725"
    assert figure.layout.scene.aspectmode == "manual"


def test_trajectory_uses_matching_coefficients_and_conditional_surface_heights():
    frame = pd.DataFrame({"coefficient_x": [0, 2, 0, 2], "coefficient_y": [0, 0, 1, 1],
                          "mutual_information": [0, 2, 2, 4]})
    history = pd.DataFrame({"generation": [1, 2, 3], "mean_c_3": [3, 1, 2],
                            "mean_c_2": [0, .5, 1], "global_best_mi": [100, 200, 300]})
    result = {"plot_info": {"selected_coefficients": ["c_4", "c_3"]}}
    figure = landscape_figure("pressure_gradient", frame, history, result)
    trajectory = figure.data[1]
    np.testing.assert_equal(np.asarray(trajectory.x), [3, 1, 2])
    np.testing.assert_allclose(np.asarray(trajectory.z)[1:], [2.06, 4.06])
    assert np.isnan(trajectory.z[0])
    assert trajectory.connectgaps is False
    np.testing.assert_equal(np.asarray(trajectory.customdata)[1:, 1], [2, 4])
    assert "Start" not in [trace.name for trace in figure.data]
    assert figure.data[-2].name == "End"
    assert figure.data[-2].textposition == "middle center"
    assert figure.data[-2].showlegend is False
    assert figure.data[-1].name == "★ End"
    assert figure.data[-1].mode == "markers"
    assert figure.layout.scene.xaxis.title.text == "c<sub>4</sub>"
    assert figure.layout.scene.yaxis.title.text == "c<sub>3</sub>"


@pytest.mark.parametrize("case_id", ["mhd", "blasius", "pressure_gradient", "rayleigh_benard", "laser_keyhole"])
def test_paper_landscape_uses_all_exported_trajectory_points_and_true_endpoints(case_id):
    folder = Path(__file__).resolve().parents[1] / "results/mi_landscapes" / case_id
    frame = pd.read_csv(folder / "response_surface.csv")
    history = pd.read_csv(next(folder.glob("*generation_history.csv")))
    trajectory = pd.read_csv(folder / "trajectory_surface.csv")
    result = json.loads((folder / "optimization_result.json").read_text(encoding="utf-8"))
    figure = landscape_figure(case_id, frame, history, result, trajectory)
    path = figure.data[1]
    offset = np.nanmax(frame.mutual_information) - np.nanmin(frame.mutual_information)
    np.testing.assert_equal(np.asarray(path.x), trajectory.coefficient_x.to_numpy())
    np.testing.assert_equal(np.asarray(path.y), trajectory.coefficient_y.to_numpy())
    np.testing.assert_allclose(np.asarray(path.z), trajectory.mutual_information + .015 * offset)
    assert np.isfinite(path.z).all()
    assert [trace.name for trace in figure.data[2:]] == ["Start", "End", "★ End"]
    np.testing.assert_equal(path.customdata[:, 0], history.generation.to_numpy())
    assert figure.layout.scene.xaxis.range[0] <= min(path.x)
    assert figure.layout.scene.xaxis.range[1] >= max(path.x)
    assert figure.layout.scene.yaxis.range[0] <= min(path.y)
    assert figure.layout.scene.yaxis.range[1] >= max(path.y)


def test_stage_bars_read_final_candidate_values_without_summing_duplicate_counts():
    local = {"dimension_path": [
        {"pi_count": 1, "mutual_information": 3, "normalized_information_score": .1},
        {"pi_count": 2, "mutual_information": 4, "normalized_information_score": .2},
        {"pi_count": 1, "mutual_information": 3.5, "normalized_information_score": .15}],
        "sparsification": {"candidate": {"mutual_information": 3.6, "normalized_information_score": .16}},
        "rationalization": {"candidate": {"mutual_information": 99, "normalized_information_score": 99},
                            "final": {"mutual_information": 3.7, "normalized_information_score": .17}}}
    frame = stage_comparison(local)
    assert frame.Stage.tolist() == ["k=1", "k=2", "Sparse", "Rationalized"]
    assert frame.MI.tolist() == [3.5, 4, 3.6, 3.7]
    assert frame["ε"].tolist() == [.15, .2, .16, .17]
    assert frame[["MI SD", "ε SD"]].isna().all().all()


def test_stage_uncertainty_stays_with_each_candidate_and_preserves_missing_components():
    local = {"dimension_path": [
        {"pi_count": 1, "mutual_information": 2, "normalized_information_score": .2,
         "restart_mutual_information_std": .03, "restart_information_score_std": 0,
         "mi_score_subsampling_uq": {"subsample_mi_std": .04, "subsample_score_std": .02}},
        {"pi_count": 2, "mutual_information": 3, "normalized_information_score": .1}],
        "sparsification": {"candidate": {"mutual_information": 2.1, "normalized_information_score": .18,
            "restart_mutual_information_std": .05,
            "mi_score_subsampling_uq": {"subsample_mi_std": .12, "subsample_score_std": .06}}},
        "rationalization": {
            "candidate": {"mutual_information": 99, "restart_mutual_information_std": 99},
            "final": {"mutual_information": 2.2, "normalized_information_score": .17,
                      "restart_mutual_information_std": .06, "restart_information_score_std": .03,
                      "mi_score_subsampling_uq": {"subsample_mi_std": .08}}}}
    original = deepcopy(local)
    frame = stage_comparison(local)
    np.testing.assert_allclose(frame["MI SD"], [.05, np.nan, .13, .1], equal_nan=True)
    np.testing.assert_allclose(frame["ε SD"], [.02, np.nan, .06, .03], equal_nan=True)
    assert frame.loc[0, "ε restart SD"] == 0
    assert pd.isna(frame.loc[2, "ε restart SD"])
    assert pd.isna(frame.loc[3, "ε subsampling SD"])
    assert local == original
    with (patch("ai_buckingham.web.diagnostics.is_publication_run", return_value=False),
          patch("ai_buckingham.web.diagnostics.st.markdown"),
          patch("ai_buckingham.web.diagnostics.st.caption") as caption,
          patch("ai_buckingham.web.diagnostics.st.columns", return_value=[nullcontext(), nullcontext()]),
          patch("ai_buckingham.web.diagnostics.st.download_button"),
          patch("ai_buckingham.web.diagnostics.chart") as display):
        mi_view({"local_workflows": [local]}, None, "example")
    caption.assert_called_once_with("Error bars: ±1 SD.")
    assert [call.args[1] for call in display.call_args_list] == ["mi_path_0", "epsilon_path_0"]
    for metric, call in zip(("MI", "ε"), display.call_args_list):
        figure = call.args[0]
        trace = figure.data[0]
        np.testing.assert_allclose(trace.y, frame[metric])
        np.testing.assert_allclose(trace.error_y.array, frame[f"{metric} SD"], equal_nan=True)
        np.testing.assert_allclose(trace.text, frame[metric])
        assert trace.textposition == "none"
        for label, (_, row) in zip(figure.layout.annotations, frame.iterrows()):
            error = row[f"{metric} SD"] if pd.notna(row[f"{metric} SD"]) else 0
            assert label.x == row.Stage
            assert label.y == row[metric] + error
            assert label.yshift == 8
            assert label.yanchor == "bottom"
            assert figure.layout.yaxis.range[1] > label.y
            assert figure.layout.yaxis.range[0] <= row[metric] - error
        assert trace.customdata[1][0] == ""
        assert "Subsampling SD" in trace.customdata[0][0]


def test_published_stage_errors_match_saved_restart_and_subsampling_values():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "results/index.json").read_text(encoding="utf-8"))["cases"]
    checked = 0
    for case in cases:
        path = root / "results/cases" / case["case_id"] / case["selected_run"] / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        for local in state["workflow_result"]["local_workflows"]:
            candidates = {item["pi_count"]: item for item in local["dimension_path"]}
            sources = [candidate for _, candidate in sorted(candidates.items())]
            for field in ("sparsification", "rationalization"):
                result = local.get(field) or {}
                solution = result.get("final") or result.get("candidate") or {}
                if solution:
                    sources.append(solution)
            frame = stage_comparison(local)
            assert len(frame) == len(sources)
            for (_, row), source in zip(frame.iterrows(), sources):
                uq = source["mi_score_subsampling_uq"]
                assert row["MI"] == source["mutual_information"]
                assert row["ε"] == source["normalized_information_score"]
                assert row["MI SD"] == np.hypot(source["restart_mutual_information_std"], uq["subsample_mi_std"])
                assert row["ε SD"] == np.hypot(source["restart_information_score_std"], uq["subsample_score_std"])
                checked += 1
    assert checked == 46


def test_titled_comparison_has_separate_legend_strip_and_unchanged_points():
    frame = pd.DataFrame({"Re_tau":[1000, 3000, 10000],"eta":[.001,.1,.8],
                          "regime":["Inner","Mixed","Outer"]})
    figure = px.scatter(frame, x="Re_tau", y="eta", color="regime", log_x=True, log_y=True,
                        title="AI Buckingham<br><sup>Recorded gradient partition</sup>")
    points = [(list(trace.x), list(trace.y)) for trace in figure.data]
    with patch("ai_buckingham.web.components.st.plotly_chart") as display:
        chart(figure, "cess_layout_regression", compact_title=True)
    assert [(list(trace.x), list(trace.y)) for trace in figure.data] == points
    assert figure.layout.title.y > 0.9
    assert figure.layout.title.yref == "paper"
    assert figure.layout.title.yanchor == "bottom"
    assert figure.layout.title.font.family.startswith("Times New Roman")
    assert figure.layout.legend.font.family.startswith("Times New Roman")
    assert figure.layout.legend.y < 0
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.margin.b >= 100
    assert figure.layout.xaxis.type == figure.layout.yaxis.type == "log"
    assert display.call_count == 1
