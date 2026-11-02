import streamlit as st
from . import overview,methods,explorer,run
from .theme import CSS

def main():
    st.set_page_config(page_title='AI Buckingham',page_icon='Π',layout='wide',initial_sidebar_state='expanded')
    st.markdown('<style>'+CSS+'</style>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><span>Π</span><strong>AI<br>Buckingham</strong></div>',unsafe_allow_html=True)
        if st.session_state.get('navigation','Overview') not in ('Overview','Methods','Explorer','Run'):st.session_state['navigation']='Explorer'
        page=st.radio('Navigate',['Overview','Methods','Explorer','Run'],key='navigation',label_visibility='collapsed')
        st.markdown('<div class="sidebar-note">Physical reasoning<br>Dimensionless learning</div>',unsafe_allow_html=True)
    {'Overview':overview.render,'Methods':methods.render,'Explorer':explorer.render,'Run':run.render}[page]()
