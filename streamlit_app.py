"""Launch with: streamlit run streamlit_app.py"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import streamlit as st
from ai_buckingham.web.components import style
from ai_buckingham.web import overview, cases, rag, diagnostics, comparisons, run_page

st.set_page_config(page_title="AI Buckingham · Evidence Explorer",page_icon="Π",layout="wide",initial_sidebar_state="expanded")
style()
with st.sidebar:
    st.markdown("## AI Buckingham")
    st.caption("Agentic discovery of dimensionless scaling laws across physical regimes")
    selection=st.radio("Explore",["Overview","Case workflow","Retrieval evidence","Numerical diagnostics","Comparisons","Run analysis"],key="page_navigation")
    st.divider()
    st.caption("Browse saved results without API credentials.")

PAGES={"Overview":overview.render,"Case workflow":cases.render,"Retrieval evidence":rag.render,
       "Numerical diagnostics":diagnostics.render,"Comparisons":comparisons.render,"Run analysis":run_page.render}
PAGES[selection]()
