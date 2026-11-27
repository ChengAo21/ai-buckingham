"""Consistent, restrained academic presentation components."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import scrub
from .publication import collapse_axes

COLORS = ["#245B78", "#209B88", "#C17837", "#8B6BA5", "#D25264", "#4D7B9A"]


def style() -> None:
    st.html("""<style>
    .stApp {background:#f7f9fc;color:#172431;}
    [data-testid="stSidebar"] {
        background:linear-gradient(170deg,#e3eeed,#f0f5f4 65%);
        border-right:1px solid #cbdcda;
    }
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
    .block-container:has(.ab-paper-heading) {padding-top:4.5rem;}
    .ab-paper-heading {
        display:grid; grid-template-columns:minmax(0,1fr) minmax(6rem,18%);
        align-items:center; gap:clamp(1rem,3vw,2.5rem);
        padding:clamp(1.1rem,3vw,2.35rem); border:1px solid #ccdddb;
        border-radius:.75rem; background:#e4f0ef;
    }
    .ab-paper-illustration {display:block; width:100%; max-width:11rem; height:auto; justify-self:center;}
    @media (max-width:54rem) {
        .ab-paper-heading {grid-template-columns:1fr;}
        .ab-paper-illustration {display:none;}
    }
    .ab-paper-heading h1 {
        max-width:32ch; margin:0; padding:0; color:#143d43;
        font:600 clamp(1.7rem,3.25vw,3rem)/1.12 Georgia,'Times New Roman',serif;
        letter-spacing:-.025em; text-wrap:balance;
    }
    .ab-paper-heading .ab-authors {margin:.85rem 0 .65rem;color:#354e52;line-height:1.5;}
    .ab-paper-heading nav {display:flex;flex-wrap:wrap;gap:.5rem 1.35rem;}
    .ab-paper-heading nav a {color:#075564;text-underline-offset:.2em;font-weight:600;}
    .st-key-overview-gallery {container-type:inline-size;}
    .st-key-overview-gallery h2 {margin-top:.5rem;color:#173e43;}
    .st-key-overview-gallery [data-testid="stColumn"] {
        padding:.9rem; border:1px solid #d5e2e0; border-radius:.65rem; background:#fff;
    }
    .st-key-overview-gallery h3 {
        margin:0; padding:0; min-height:2.6em; color:#193f44;
        font:600 clamp(1.05rem,1.6vw,1.3rem)/1.3 Georgia,'Times New Roman',serif;
    }
    @container (max-width:44rem) {
        .st-key-overview-gallery [data-testid="stHorizontalBlock"] {flex-direction:column;}
        .st-key-overview-gallery [data-testid="stColumn"] {width:100%;flex:auto;min-width:0;}
        .st-key-overview-gallery h3 {min-height:0;}
    }
    </style>""")


def masthead(title: str, description: str = "", tag: str = "") -> None:
    kicker = "AI Buckingham" + (" · " + tag if tag else "")
    deck = f'<div class="ab-deck">{html.escape(description)}</div>' if description else ""
    st.markdown(f'<div class="ab-masthead"><div class="ab-kicker">{html.escape(kicker)}</div>'
                f'<h1>{html.escape(title)}</h1>{deck}</div>', unsafe_allow_html=True)


def chart(fig: go.Figure, key: str | None = None, *, compact_title: bool = False) -> None:
    # Keep the title above the plotting area and the legend below the axes.
    # A shared top strip made long titles collide with Plotly Express legend
    # headings, especially when two comparisons share a Streamlit row.
    legend_traces = [trace for trace in fig.data if hasattr(trace, "showlegend") and trace.showlegend is not False]
    has_legend = (fig.layout.showlegend is not False
                  and (fig.layout.showlegend is True or len(legend_traces)>1
                       or any(trace.showlegend is True for trace in legend_traces)))
    fig.update_layout(template="plotly_white", colorway=COLORS, paper_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Times New Roman, Times, serif", "size": 14, "color": "#20384b"},
                      height=fig.layout.height or 520,
                      title={"x":0,"xanchor":"left","y":.98,"yanchor":"top","automargin":True,
                             "font":{"family":"Times New Roman, Times, serif","size":18}},
                      margin={"l": 45, "r": 35, "t": 85 if fig.layout.title.text else 35, "b": 125 if has_legend else 60,
                              "autoexpand":True},
                      legend={"orientation":"h","yanchor":"top","y":-.22,"x":0,"xanchor":"left",
                              "title":{"side":"top"},"font":{"family":"Times New Roman, Times, serif","size":12},"maxheight":100})
    if compact_title and fig.layout.title.text:
        # Keep comparison headings close to the axes, with room for two lines
        # below the preceding controls, independent of the figure height.
        fig.update_layout(title={"yref":"paper","y":1,"yanchor":"bottom",
                                 "pad":{"b":30},"automargin":False}, margin={"t":85})
    fig.update_xaxes(automargin=True, title_standoff=16)
    fig.update_yaxes(automargin=True, title_standoff=12)
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}})


def collapse_axis_controls(case_id: str, panel: str, *, key: str) -> tuple[bool, bool]:
    x_scale, y_scale = collapse_axes(case_id, panel)
    left, right = st.columns(2)
    log_x = left.checkbox("Logarithmic x-axis", value=x_scale == "log", key=f"{key}_log_x")
    log_y = right.checkbox("Logarithmic y-axis", value=y_scale == "log", key=f"{key}_log_y")
    return log_x, log_y


def table(frame: pd.DataFrame, name: str, key: str | None = None) -> None:
    if frame.empty:
        st.info("This table has not been recorded for the selected evidence source.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("Download table · CSV", frame.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv", key=key or f"download_{name}")


def interpretation_commentary(text: str | None) -> None:
    if not text:
        return
    with st.expander("Model output", expanded=False):
        st.markdown(text)


def json_panel(title: str, value, key: str | None = None, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.json(scrub(value), expanded=True)
        st.download_button("Download · JSON", json.dumps(scrub(value), ensure_ascii=False, indent=2, default=str),
                           file_name=title.lower().replace(" ", "_") + ".json", mime="application/json",
                           key=key or "json_" + title)


def json_download(title: str, value, key: str) -> None:
    st.download_button("Download " + title + " · JSON",
                       json.dumps(scrub(value), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                       file_name=key + ".json", mime="application/json", key=key)


def figures(folder: Path, key: str = "figures", *, display_images: bool = True) -> None:
    images = sorted(folder.rglob("*.png")) if folder.is_dir() else []
    vectors = sorted(folder.rglob("*.pdf")) + sorted(folder.rglob("*.svg")) if folder.is_dir() else []
    if not images and not vectors:
        st.info("No rendered figure is present in this run. Numerical evidence is shown separately.")
        return
    for i, path in enumerate(images):
        if display_images:
            st.image(str(path), caption=path.relative_to(folder).as_posix(), width="stretch")
        st.download_button("Download full-resolution PNG", path.read_bytes(), file_name=path.name,
                           mime="image/png", key=f"{key}_png_{i}")
    for i, path in enumerate(vectors):
        st.download_button("Download figure · " + path.name, path.read_bytes(), file_name=path.name,
                           mime="application/pdf" if path.suffix == ".pdf" else "image/svg+xml", key=f"{key}_vector_{i}")


def selected_run(case_id: str, *, key: str = "case_run"):
    from .data import runs, load_run
    available = runs(case_id)
    if not available:
        st.info("No saved case run is available. Input data and archived benchmark evidence remain accessible.")
        return None
    picked = available[0] if len(available) == 1 else st.selectbox("Run", available, format_func=lambda p: p.name, key=key)
    result = load_run(picked)
    manifest = result["manifest"]
    status = str(manifest.get("status", manifest.get("scientific_status", "not recorded")))
    if len(available) > 1 or status != "completed":
        st.caption("Run " + picked.name + " · " + status)
    return result
