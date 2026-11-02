"""Regime-wise objective exploration and reported selection/validation evidence."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from ..catalog import ROOT,CASES,reference
from ..scopes import REGIMES,scope_data,regime_label
from ..experiments import landscape,optimize_slice,display_trace,jsonable
from ..charts import surface,convergence,validation_panel
from .theme import heading,plot_title

@st.cache_data(show_spinner=False)
def read_file(path,revision):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def load(path):return read_file(path,(ROOT/path).stat().st_mtime_ns)

def optimization(key,scope='global',partition=None):
    suffix=key if scope=='global' else key+'_'+scope
    current=scope_data(key,scope,partition)
    state_key=suffix+'_'+current['fingerprint'][:12]
    with st.container(border=True,key='search_controls_'+suffix):
        plot_title('Surface and search controls')
        a,b,c,d=st.columns(4,gap='medium')
        with a:i=st.selectbox('Horizontal coefficient',range(len(CASES[key]['basis'])),format_func=lambda n:f'c{n+1}',key='axis_i_'+suffix)
        with b:j=st.selectbox('Vertical coefficient',range(len(CASES[key]['basis'])),index=1,format_func=lambda n:f'c{n+1}',key='axis_j_'+suffix)
        with c:resolution=st.selectbox('Grid per axis',[45,65,81,101],index=2,key='grid_resolution_'+suffix)
        with d:span=st.number_input('Coefficient half-span',min_value=.25,max_value=3.,value=1.,step=.25,key='grid_span_'+suffix)
        a,b,c,d,e=st.columns([1,1,1,1.05,1.05],gap='medium',vertical_alignment='bottom')
        with a:gens=st.number_input('Generations',min_value=5,max_value=30000,value=40,step=5,key='slice_generations_'+suffix)
        with b:pop=st.number_input('Population',min_value=6,max_value=300,value=24,step=6,key='slice_population_'+suffix)
        with c:sigma=st.number_input('Initial scale',min_value=.01,max_value=2.,value=.25,step=.05,key='slice_sigma_'+suffix)
        with d:update=st.button('Update surface',disabled=i==j,key='update_surface_'+suffix,width='stretch')
        with e:search=st.button('Run search',disabled=i==j,key='run_slice_'+suffix,width='stretch',type='primary')
    if update or search:
        from threadpoolctl import threadpool_limits
        with st.spinner('Evaluating the selected coefficient plane…'):
            with threadpool_limits(limits=1):
                st.session_state['grid_'+state_key]=landscape(key,resolution,i,j,span,False,scope,partition)
                if search:st.session_state['trace_'+state_key]=optimize_slice(key,int(gens),int(pop),float(sigma),i,j,scope,partition,span)
    p=st.session_state.get('grid_'+state_key)
    run=st.session_state.get('trace_'+state_key)
    if p is None:
        stored=load(f'results/previews/{suffix}_landscape.json')
        if scope=='global' or stored.get('scope_fingerprint')==current['fingerprint']:p=stored
        else:
            st.info('The current partition changed. Update the surface to evaluate its new observations.');return
    if run is None:
        stored=load(f'results/previews/{suffix}_trajectory.json')
        if scope=='global' or stored.get('scope_fingerprint')==current['fingerprint']:run=stored
    axes=run.get('settings',{}).get('basis_axes',[0,1]) if run else [0,1]
    compatible=run and {p['axis_i'],p['axis_j']}==set(axes)
    trace=display_trace(run) if compatible else None
    with st.container(horizontal=True,wrap=True,gap='medium',key='learning_triptych_'+suffix):
        with st.container(width=330,key='learning_plot_contour_'+suffix):
            plot_title('Mutual-information contours')
            st.plotly_chart(surface(p,False,trace,axes,False,390),width='stretch',key='contour_'+suffix)
        with st.container(width=330,key='learning_plot_surface_'+suffix):
            plot_title('Objective surface')
            st.plotly_chart(surface(p,True,trace,axes,False,390),width='stretch',key='surface_'+suffix)
        with st.container(width=330,key='learning_plot_convergence_'+suffix):
            plot_title('Best-so-far mutual information')
            if compatible:st.plotly_chart(convergence(run['trace'],390),width='stretch',key='convergence_'+suffix)
            else:st.info('Run a search in this coefficient plane to display its convergence.')
    st.markdown('<div class="optimization-key"><span><i class="key-line"></i>CMA mean</span><span><i class="key-start"></i>Start</span><span><i class="key-end"></i>End</span><span><i class="key-best"></i>Best MI</span></div>',unsafe_allow_html=True)
    st.caption(f"{p['rows']:,} samples · "+('KSG' if p['method']=='kraskov' else 'Histogram')+f" ({p['param']}) · Two coefficients vary; all others are fixed. Paths are new searches"+(' within this physical regime.' if scope!='global' else '.'))
    data=pd.DataFrame([(x,y,-p['z'][b][a]) for b,y in enumerate(p['y']) for a,x in enumerate(p['x'])],columns=[f"c{p['axis_i']+1}",f"c{p['axis_j']+1}",'MI'])
    with st.container(horizontal=True):
        st.download_button('Surface CSV',data.to_csv(index=False),file_name=suffix+'_surface.csv',key='download_surface_'+suffix)
        if run:st.download_button('Search JSON',json.dumps(jsonable(run),indent=2),file_name=suffix+'_search.json',key='download_search_'+suffix)

def validation(key):
    heading('Selection and validation')
    rows=[r for r in reference()['stages'] if r['case']==key]
    st.plotly_chart(validation_panel(rows),width='stretch',key='validation_'+key)
    if 'screening' in CASES[key]:
        x,y,loss=CASES[key]['screening'];a,b,c=st.columns(3)
        a.metric('Type I · MI',f'{x:.3f}');b.metric('Type II · MI',f'{y:.3f}');c.metric('Relative MI loss',f'{loss:.3f}')
    plot_title('Stage values and predictive validation')
    df=pd.DataFrame(rows)[['regime','stage','mi','mi_sd','score','score_sd','r2']]
    df['regime']=df['regime'].map(regime_label)
    st.dataframe(df.rename(columns={'regime':'Regime','stage':'Stage','mi':'MI','mi_sd':'σ(MI)','score':'ε','score_sd':'σ(ε)','r2':'Validation R²'}),hide_index=True,width='stretch',
        column_config={name:st.column_config.NumberColumn(format='%.3f') for name in ['MI','σ(MI)','ε','σ(ε)','Validation R²']})
    st.caption('Reported stage values and uncertainty. Held-out R² evaluates the selected representation.')
    if key=='pipe':st.warning('Regime B · Viscous: the rationalized Reynolds representation fails predictive validation (R² = −3.079).')
    st.download_button('Results CSV',df.to_csv(index=False),file_name=key+'_validation.csv')

def render(key,partition=None):
    if key in REGIMES:
        scopes=REGIMES[key]
        panes=st.tabs(['Regime '+s+' · '+item['name'] for s,item in scopes.items()])
        for pane,(scope,item) in zip(panes,scopes.items()):
            with pane:
                st.latex(item['formula'])
                optimization(key,scope,partition)
    else:optimization(key)
    validation(key)
