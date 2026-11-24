"""Consistent, restrained academic presentation components."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import scrub

COLORS = ["#245B78", "#209B88", "#C17837", "#8B6BA5", "#D25264", "#4D7B9A"]


def style() -> None:
    st.markdown("""<style>
    .stApp {background:#f7f9fc;color:#172431;}
    [data-testid="stSidebar"] {background:#edf2f7;border-right:1px solid #d8e1eb;}
    .block-container {max-width:1450px;padding-top:2.4rem;padding-bottom:3rem;}
    h1,h2,h3 {font-family:Georgia,'Times New Roman',serif;letter-spacing:-.025em;color:#142a3a;}
    h1 {font-size:2.8rem;} h2 {font-size:1.9rem;} h3 {font-size:1.45rem;}
    .ab-masthead {position:relative;overflow:hidden;border-top:4px solid #17374c;border-bottom:1px solid #cdd8e4;padding:1rem 235px 1.4rem 0;margin-bottom:1.5rem;}
    .ab-motif {position:absolute;right:8px;top:22px;width:210px;height:125px;opacity:.3;pointer-events:none;}
    @media (max-width:900px) {.ab-masthead {padding-right:0;}.ab-motif {display:none;}}
    .ab-kicker {font-size:.85rem;letter-spacing:.15em;font-weight:700;color:#245b78;text-transform:uppercase;}
    .ab-deck {color:#55677a;font-size:1.05rem;line-height:1.6;max-width:1100px;}
    .ab-tag {display:inline-block;border:1px solid #bdd2e0;background:#edf5fa;color:#245b78;border-radius:3px;padding:.2rem .6rem;font-size:.85rem;margin:.2rem .35rem .2rem 0;}
    [data-testid="stMetric"] {background:white;border:1px solid #d8e1eb;border-top:3px solid #245b78;padding:1rem;border-radius:4px;}
    [data-testid="stMetricLabel"] {font-size:.9rem;}
    [data-testid="stMetricLabel"] p {white-space:normal;overflow:visible;text-overflow:clip;}
    [data-testid="stExpander"] {background:white;border-color:#d8e1eb;}
    .stButton>button,.stDownloadButton>button {border-radius:4px;}
    .stCaption {color:#5a6b7c;}
    </style>""", unsafe_allow_html=True)


def masthead(title: str, description: str, tag: str = "Evidence explorer") -> None:
    motif = ('<svg class="ab-motif" viewBox="0 0 210 125" aria-hidden="true" focusable="false">'
             '<g fill="none" stroke="#209b88" stroke-width="1.4">'
             '<path d="M6 105C44 107 40 33 78 42S119 109 151 66S180 28 204 17"/>'
             '<path d="M6 115C43 117 43 51 81 57S120 119 154 81S183 45 204 36" opacity=".45"/>'
             '<path d="M6 95C41 97 37 16 75 27S116 96 148 51S177 11 204 3" opacity=".45"/>'
             '</g><g fill="#245b78"><circle cx="78" cy="42" r="3.5"/>'
             '<circle cx="151" cy="66" r="3.5"/><circle cx="188" cy="27" r="3.5"/></g>'
             '<path d="M5 121H207M10 121V8" fill="none" stroke="#245b78" opacity=".35"/></svg>')
    st.markdown(f'<div class="ab-masthead">{motif}<div class="ab-kicker">AI Buckingham · {html.escape(tag)}</div>'
                f'<h1>{html.escape(title)}</h1><div class="ab-deck">{html.escape(description)}</div></div>', unsafe_allow_html=True)


def chart(fig: go.Figure, key: str | None = None, *, compact_title: bool = False) -> None:
    # Keep the title above the plotting area and the legend below the axes.
    # A shared top strip made long titles collide with Plotly Express legend
    # headings, especially when two comparisons share a Streamlit row.
    legend_traces = [trace for trace in fig.data if hasattr(trace, "showlegend") and trace.showlegend is not False]
    has_legend = (fig.layout.showlegend is not False
                  and (fig.layout.showlegend is True or len(legend_traces)>1
                       or any(trace.showlegend is True for trace in legend_traces)))
    fig.update_layout(template="plotly_white", colorway=COLORS, paper_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Arial, sans-serif", "size": 14, "color": "#20384b"},
                      height=fig.layout.height or 520,
                      title={"x":0,"xanchor":"left","y":.98,"yanchor":"top","automargin":True,
                             "font":{"size":18}},
                      margin={"l": 45, "r": 35, "t": 85 if fig.layout.title.text else 35, "b": 125 if has_legend else 60,
                              "autoexpand":True},
                      legend={"orientation":"h","yanchor":"top","y":-.22,"x":0,"xanchor":"left",
                              "title":{"side":"top"},"font":{"size":12},"maxheight":100})
    if compact_title and fig.layout.title.text:
        # Keep comparison headings close to the axes, with room for two lines
        # below the preceding controls, independent of the figure height.
        fig.update_layout(title={"yref":"paper","y":1,"yanchor":"bottom",
                                 "pad":{"b":30},"automargin":False}, margin={"t":85})
    fig.update_xaxes(automargin=True, title_standoff=16)
    fig.update_yaxes(automargin=True, title_standoff=12)
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}})


def table(frame: pd.DataFrame, name: str, key: str | None = None) -> None:
    if frame.empty:
        st.info("This table has not been recorded for the selected evidence source.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("Download table · CSV", frame.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv", key=key or f"download_{name}")


def json_panel(title: str, value, key: str | None = None, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.json(scrub(value), expanded=False)
        st.download_button("Download · JSON", json.dumps(scrub(value), ensure_ascii=False, indent=2, default=str),
                           file_name=title.lower().replace(" ", "_") + ".json", mime="application/json",
                           key=key or "json_" + title)


def figures(folder: Path, key: str = "figures") -> None:
    images = sorted(folder.rglob("*.png")) if folder.is_dir() else []
    vectors = sorted(folder.rglob("*.pdf")) + sorted(folder.rglob("*.svg")) if folder.is_dir() else []
    if not images and not vectors:
        st.info("No rendered figure is present in this run. Numerical evidence is shown separately.")
        return
    for i, path in enumerate(images):
        st.image(str(path), caption=path.relative_to(folder).as_posix(), width="stretch")
        st.download_button("Download full-resolution PNG", path.read_bytes(), file_name=path.name,
                           mime="image/png", key=f"{key}_png_{i}")
    for i, path in enumerate(vectors):
        st.download_button("Download publication figure · " + path.name, path.read_bytes(), file_name=path.name,
                           mime="application/pdf" if path.suffix == ".pdf" else "image/svg+xml", key=f"{key}_vector_{i}")


def selected_run(case_id: str, *, key: str = "case_run"):
    from .data import runs, load_run
    available = runs(case_id)
    if not available:
        st.info("No saved case run is available. Input data and archived benchmark evidence remain accessible.")
        return None
    picked = st.selectbox("Evidence run", available, format_func=lambda p: p.name, key=key)
    result = load_run(picked)
    manifest = result["manifest"]
    st.caption("Source: " + str(manifest.get("source_kind", manifest.get("origin", manifest.get("evidence_type", manifest.get("mode", "saved run")))))
               + " · Status: " + str(manifest.get("status", manifest.get("scientific_status", "not recorded"))))
    return result
