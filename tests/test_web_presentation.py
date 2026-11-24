"""Presentation must keep saved MI values and separate titles from legends."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.express as px

from ai_buckingham.web.components import chart
from ai_buckingham.web.diagnostics import mi_contour


def test_dense_mi_contours_preserve_saved_grid_and_full_signed_range():
    grid = pd.DataFrame([[-0.2, 0.4], [np.nan, 1.8]], index=[-1, 1], columns=[0, 2])
    trace = mi_contour(grid)
    np.testing.assert_equal(np.asarray(trace.z), grid.to_numpy())
    np.testing.assert_equal(np.asarray(trace.x), grid.columns.to_numpy())
    np.testing.assert_equal(np.asarray(trace.y), grid.index.to_numpy())
    assert trace.contours.start == -0.2
    assert trace.contours.end == 1.8
    assert (trace.contours.end-trace.contours.start)/trace.contours.size >= 79


def test_constant_mi_grid_remains_constant_without_zero_contour_step():
    grid = pd.DataFrame(np.full((2, 2), 0.3))
    trace = mi_contour(grid)
    np.testing.assert_equal(np.asarray(trace.z), grid.to_numpy())
    assert trace.contours.size is None


def test_titled_comparison_has_separate_legend_strip_and_unchanged_points():
    frame = pd.DataFrame({"Re_tau":[1000, 3000, 10000],"eta":[.001,.1,.8],
                          "regime":["Inner","Mixed","Outer"]})
    figure = px.scatter(frame, x="Re_tau", y="eta", color="regime", log_x=True, log_y=True,
                        title="AI Buckingham<br><sup>Recorded gradient partition</sup>")
    points = [(list(trace.x), list(trace.y)) for trace in figure.data]
    with patch("ai_buckingham.web.components.st.plotly_chart") as display:
        chart(figure, "cess_layout_regression")
    assert [(list(trace.x), list(trace.y)) for trace in figure.data] == points
    assert figure.layout.title.y > 0.9
    assert figure.layout.legend.y < 0
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.margin.b >= 100
    assert figure.layout.xaxis.type == figure.layout.yaxis.type == "log"
    assert display.call_count == 1
