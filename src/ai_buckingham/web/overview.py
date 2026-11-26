"""The paper's physical systems and result figures."""
import streamlit as st

from .components import masthead
from .data import ROOT
from .publication import publication


def render():
    metadata = publication()
    paper = metadata["paper"]
    masthead(paper["title"])
    st.markdown(f"[arXiv:{paper['version']}]({paper['url']})")
    for group in metadata["groups"]:
        st.markdown("## " + group["title"])
        for case in metadata["cases"]:
            if case["group"] != group["id"]:
                continue
            st.markdown("### " + case["title"])
            image = ROOT / case["image"]
            if image.is_file():
                st.image(str(image), width="stretch")
            else:
                st.info("Case figure is unavailable.")
