from html import escape
import streamlit as st
from ..catalog import CASES,frame
from ..showcase import CASE_STORIES
from .theme import heading,image
from .methods import TYPE_I,TYPE_II

def open_case(key):
    st.session_state['selected_case']=key
    st.session_state['navigation']='Explorer'

def render():
    st.markdown('<div class="hero"><div class="hero-copy"><div class="eyebrow">Agentic data-driven physics</div><h1>AI Buckingham</h1><p>Discover physical scaling laws through LLM-based orchestration, gradient-based physical-regime identification, and mutual-information-based dimensionless learning.</p><div class="hero-capabilities"><div><span>Π</span><b>Type I & II similarity<small>Autonomous identification</small></b></div><div><span>↔</span><b>Global / multiple physical regimes<small>Autonomous regime assessment</small></b></div></div></div><img class="hero-robot" alt="AI agent exploring physical scaling laws" src="'+image('docs/assets/agent-robot.png')+'"></div>',unsafe_allow_html=True)
    heading('AI Buckingham workflow')
    st.markdown('<div class="scheme"><img alt="Problem classes and agent workflow" src="'+image('docs/assets/discovery-map.svg')+'"></div>',unsafe_allow_html=True)
    heading('Explore the physical systems','Nine cases across complete, incomplete and regime-dependent similarity')
    family=st.segmented_control('Case family',['All cases','Type I','Type II','Multiple regimes'],default='All cases',key='case_family',label_visibility='collapsed')
    for group,title in [('Type I','Type I · complete similarity'),('Type II','Type II · incomplete similarity'),('Multi-regime','Multiple physical regimes')]:
        if family not in (None,'All cases',group) and not(family=='Multiple regimes' and group=='Multi-regime'):continue
        st.markdown('### '+title)
        if group=='Type I':st.latex(TYPE_I)
        elif group=='Type II':st.latex(TYPE_II)
        else:
            st.latex(r'\operatorname{Sim}(\nabla g,\Omega_I)=\sum_{j=1}^{n}\left(\frac{\lambda_j^{(I)}}{\|\boldsymbol\lambda_I\|_2}\right)^2\left[\nabla g\cdot\mathbf v_j^{(I)}\right]^2')
            st.caption('Partition by gradient similarity, then learn Type I or Type II scaling within each physical regime.')
        with st.container(horizontal=True,wrap=True,gap='small',horizontal_alignment='left',key='case_gallery_'+group.replace(' ','_')):
            for key,c in CASES.items():
                if c['category']!=group:continue
                with st.container(width=280,border=True,key='case_card_'+key):
                    st.markdown('<div class="case-heading">'+escape(CASE_STORIES[key]['short'])+'</div>',unsafe_allow_html=True)
                    st.image(image(f'docs/assets/schematics/{key}.png'),width='stretch')
                    st.caption(f'{len(frame(key)):,} observations')
                    st.button('Explore results ↗',key='overview_open_'+key,on_click=open_case,args=(key,),width='stretch')
