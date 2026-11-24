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
    .ab-masthead {border-top:4px solid #17374c;border-bottom:1px solid #cdd8e4;padding:1rem 0 1.4rem;margin-bottom:1.5rem;}
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
    st.markdown(f'<div class="ab-masthead"><div class="ab-kicker">AI Buckingham · {html.escape(tag)}</div>'
                f'<h1>{html.escape(title)}</h1><div class="ab-deck">{html.escape(description)}</div></div>', unsafe_allow_html=True)


def chart(fig: go.Figure, key: str | None = None) -> None:
    fig.update_layout(template="plotly_white", colorway=COLORS, paper_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Arial, sans-serif", "size": 14, "color": "#20384b"},
                      margin={"l": 30, "r": 25, "t": 50, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0})
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
