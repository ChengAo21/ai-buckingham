"""The paper's physical systems and result figures."""
from html import escape

import streamlit as st

from .data import ROOT
from .publication import publication


def render():
    metadata = publication()
    paper = metadata["paper"]
    authors = " · ".join(escape(author) for author in paper["authors"])
    st.html(f"""
        <header class="ab-paper-heading">
          <div>
            <h1>{escape(paper["title"])}</h1>
            <p class="ab-authors">{authors}</p>
            <nav aria-label="Paper and code">
              <a href="{escape(paper["url"], quote=True)}" target="_blank" rel="noopener noreferrer">arXiv:{escape(paper["version"])}</a>
              <a href="https://github.com/ChengAo21/ai-buckingham" target="_blank" rel="noopener noreferrer">GitHub</a>
            </nav>
          </div>
          <img class="ab-paper-illustration" src="app/static/buckingham-robot.png" alt="">
        </header>
    """)

    with st.expander("Figures 1–2 · Similarity settings and workflow", expanded=False):
        similarity, workflow = st.tabs(["Similarity settings", "AI Buckingham workflow"])
        with similarity:
            st.image(str(ROOT / "docs/assets/similarity-settings.png"), width="stretch",
                     caption="Figure 1. Overview of the physical similarity settings considered in this work.")
        with workflow:
            st.image(str(ROOT / "docs/assets/workflow.png"), width="stretch",
                     caption="Figure 2. " + paper["workflow_caption"])

    with st.container(key="overview-gallery"):
        for group in metadata["groups"]:
            st.markdown("## " + group["title"])
            cases = [case for case in metadata["cases"] if case["group"] == group["id"]]
            for start in range(0, len(cases), 2):
                pair = cases[start:start + 2]
                for column, case in zip(st.columns(len(pair), gap="medium", border=True), pair):
                    with column:
                        st.markdown("### " + case["title"])
                        image = ROOT / case["image"]
                        if image.is_file():
                            st.image(str(image), width="stretch")
                        else:
                            st.info("Case figure is unavailable.")
