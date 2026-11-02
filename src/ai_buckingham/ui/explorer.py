import json
from html import escape
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ..catalog import ROOT,CASES,frame,prepare_data,coordinates,reference
from ..settings import paper_settings
from ..showcase import CASE_STORIES
from ..charts import scatter,style,regime_names,category_style,COLORS
from ..experiments import regime_diagnostic,jsonable
from ..representations import scaling_data,axis_defaults
from ..scopes import REGIMES
from .theme import heading,image,formula,plot_title
from .dimensions import view as dimension_view
from .knowledge import case_view
from .learning import render as learning_view
from .eigenpairs import render as eigen_details

@st.cache_data(show_spinner=False)
def cached_file(path,revision):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def load(path):return cached_file(path,(ROOT/path).stat().st_mtime_ns)

def draw(fig,**kwargs):
    with st.container(horizontal=True,horizontal_alignment="center"):
        with st.container(width=550):st.plotly_chart(fig,width="stretch")

def groups(df):
    return next((df[n].astype(str).to_numpy() for n in ('source','case') if n in df and df[n].nunique()<20),None)

def active_partition(key):
    return st.session_state.get('regime_'+key)


def scaling(key):
    c=CASES[key]
    dimension_view(c)
    scope='global'
    if key in REGIMES:
        scope=st.segmented_control('Physical regime',list(REGIMES[key]),default='A',key='scaling_regime_'+key,
            format_func=lambda s:'Regime '+s+' · '+REGIMES[key][s]['name']) or 'A'
    data=scaling_data(key,scope,active_partition(key))
    with st.container(border=True,key='scaling_formula_'+key):
        st.latex(data['formula'])
        if data['output_formula']:st.latex(data['output_formula'])
    defaults=axis_defaults(key,scope)
    colors=np.repeat('Regime '+scope+' · '+REGIMES[key][scope]['name'],len(data['rows'])) if key in REGIMES else groups(data['frame'])
    left,right=st.columns(2,gap='large')
    for index,(pane,title,x,y,xlabel,ylabel) in enumerate([
        (left,'Response in the Buckingham–Π basis',data['raw_x'],data['raw_y'],data['raw_label'],c['output']),
        (right,'Response in the discovered representation',data['x'],data['y'],'Π₁','Rescaled response Π̃y' if data['output_formula'] else c['output'])]):
        with pane:
            plot_title(title)
            with st.container(horizontal=True,gap='medium'):
                logx=st.toggle('Log x',value=defaults[index][0],key=f'log_x_{key}_{scope}_{index}')
                logy=st.toggle('Log y',value=defaults[index][1],key=f'log_y_{key}_{scope}_{index}')
            draw(scatter(x,y,xlabel,ylabel,colors,logx,logy,data['rows']))
    if c.get('exponent_note'):st.caption(c['exponent_note'])
    with st.container(horizontal=True):
        st.download_button('Data CSV',frame(key).to_csv(index=False),file_name=c['file'])
        st.download_button('Scaling CSV',pd.DataFrame({'row':data['rows'],'input_pi':data['x'],'response':data['y']}).to_csv(index=False),file_name=key+'_'+scope+'_scaling.csv')


def criteria(key):
    cl=paper_settings(key)['gradient_regime_options']['clustering']
    items=[(str(cl['min_cluster_samples']),'Samples per cluster','Minimum support'),(f"{cl['min_cluster_fraction']:.0%}",'Cluster fraction','Minimum share'),(f"{cl['eigenvalue_ratio_threshold']:g}",'Eigenvalue ratio','Minimum anisotropy'),(f"< {cl['max_intercluster_cosine']:.2f}",'Direction cosine','K = 2 separation')]
    st.markdown('<div class="criteria">'+''.join(f'<div class="criterion"><b>{v}</b><span>{label}</span><small>{sub}</small></div>' for v,label,sub in items)+'</div>',unsafe_allow_html=True)

def recompute(key):
    from threadpoolctl import threadpool_limits
    with st.spinner('Fitting the surrogate and assessing candidate partitions…'):
        with threadpool_limits(limits=1):st.session_state['regime_'+key]=jsonable(regime_diagnostic(key,False))

def selected_labels(result,choice='Selected partition'):
    """A global decision is one group, even when no cluster-label array is stored."""
    count=len(result.get('feature_coordinates',[]))
    partitions=result.get('candidate_partitions',{})
    if choice not in (None,'Selected partition'):
        candidate=partitions[choice.split()[-1]]
        values=candidate['labels'] if isinstance(candidate,dict) else candidate
    elif result.get('best_k',1)==1:
        values=np.zeros(count,dtype=int)
    else:
        candidate=partitions.get(str(result.get('best_k')), {})
        values=result.get('labels',candidate.get('labels') if isinstance(candidate,dict) else candidate)
    if values is None or len(values)!=count:raise ValueError('The selected partition has no compatible row labels.')
    return np.asarray(values,int)

def regimes(key):
    r=reference()['regime_summary'][key]
    run=st.session_state.get('regime_'+key)
    if run is None and (ROOT/f'results/computed/{key}_regimes.json').exists():run=load(f'results/computed/{key}_regimes.json')
    if run:
        result=run['result'];evidence=result.get('decision_evidence',{})
        r=dict(selected_k=result.get('best_k',1),global_E=result.get('single_regime_baseline',{}).get('leading_fraction',r['global_E']),direction_cosine=evidence.get('k2_absolute_cosine',r['direction_cosine']))
    a,b,d=st.columns(3);a.metric('Selected regimes',r['selected_k']);b.metric('Global eigenvalue fraction E',f"{r['global_E']:.4g}");d.metric('K = 2 · |cos θ|',f"{r['direction_cosine']:.3f}" if r['direction_cosine'] is not None else '—')
    criteria(key)
    st.caption('Choose the smallest converged partition that satisfies all criteria.')
    st.button('Recompute partition',key='recompute_regimes',on_click=recompute,args=(key,))
    if not run:
        st.markdown('**One global scaling**' if r['selected_k']==1 else '**Local scaling**')
        st.caption('Recompute the partition to inspect gradient directions and candidate clusters.');return
    det=run['result'];parts=det.get('candidate_partitions',{})
    options=['Selected partition']+[f'K = {n}' for n in parts]
    chosen=st.segmented_control('Partition',options,default=options[0],key='partition_'+key)
    features=np.asarray(det.get('feature_coordinates',[]));grads=np.asarray(det.get('surrogate_gradients',[]))
    if features.ndim!=2 or not len(features):st.info(det.get('message','No valid partition.'));return
    try:labels=selected_labels(det,chosen)
    except ValueError as exc:st.info(str(exc));return
    valid=np.asarray(det.get('valid_indices',range(len(features))),int);df,X,Y,rows=prepare_data(key);phys=df.iloc[valid]
    names,mapping=regime_names(key,labels,phys)
    left,right=st.columns(2,gap='large')
    with left:
        plot_title('Gradient-based partition')
        i=0;j=min(1,features.shape[1]-1)
        fig=scatter(features[:,i],features[:,j],'log Π(S)₁','log Π(S)₂',names,rows=rows[valid])
        for n in range(0,len(features),max(1,len(features)//25)):
            if len(grads):
                v=grads[n,[i,j]];v=v/max(np.linalg.norm(v),1e-12)*.045*np.linalg.norm(np.ptp(features[:,[i,j]],axis=0))
                fig.add_annotation(x=features[n,i]+v[0],y=features[n,j]+v[1],ax=features[n,i],ay=features[n,j],xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=2,arrowsize=.8,arrowwidth=1,arrowcolor='#374e48',opacity=.55)
        draw(fig)
    with right:
        plot_title('Physical coordinates')
        if key=='cess':x=phys['u_tau']*phys['h']/phys['nu'];y=phys['y']/phys['h'];xl='Reτ';yl='y / h'
        elif key=='pipe':x=phys['rho']*phys['U']*phys['D']/phys['mu'];y=phys['f_D'];xl='Re';yl='f_D'
        else:x=np.exp(features[:,0]);y=Y[valid,0];xl='Π⁽¹⁾';yl=CASES[key]['output']
        draw(scatter(x,y,xl,yl,names,True,key!='pipe',rows[valid]),width='stretch')
    left,right=st.columns(2,gap='large')
    with left:
        plot_title('Gradient direction distribution')
        fig=go.Figure(go.Histogram(x=np.degrees(np.arctan2(grads[:,j],grads[:,i])),nbinsx=36,marker_color='#6f9d88'))
        fig.update_layout(xaxis_title='Angle / degrees',yaxis_title='Observations')
        draw(style(fig,360).update_layout(margin=dict(l=65,r=20,t=15,b=65)),width='stretch')
    with right:
        plot_title('Gradient covariance spectrum')
        k=str(det.get('best_k',1)) if chosen in (None,options[0]) else chosen.split()[-1]
        candidate=next((v for v in det.get('candidate_summary',[]) if str(v['k'])==k),None)
        fig=go.Figure()
        if candidate:
            for cluster in candidate.get('clusters',[]):
                name=mapping.get(cluster['cluster'],f"Regime {chr(65+cluster['cluster'])}")
                fig.add_trace(go.Bar(x=[f'λ{n+1}' for n in range(len(cluster['eigenvalues']))],y=cluster['eigenvalues'],name=name,marker_color=category_style(name,cluster['cluster'])[0]))
        elif k=='1':
            eigenvalues=det.get('single_regime_baseline',{}).get('eigenvalues',[])
            fig.add_trace(go.Bar(x=[f'λ{n+1}' for n in range(len(eigenvalues))],y=eigenvalues,name='Global regime',marker_color='#6f9d88',width=.4))
        fig.update_layout(yaxis_type='log',yaxis_title='Eigenvalue',barmode='group')
        draw(style(fig,360),width='stretch')
    candidates=[{'K':v['k'],'Cluster sizes':str(v['sample_counts']),'Min. ratio':v.get('minimum_eigenvalue_ratio'),'Min. fraction':v.get('minimum_cluster_fraction'),'Converged':v.get('converged'),'Accepted':v.get('meets_regime_criteria')} for v in det.get('candidate_summary',[])]
    if candidates:st.dataframe(pd.DataFrame(candidates),hide_index=True,width='stretch')
    st.download_button('Partition JSON',json.dumps(jsonable(run),indent=2),file_name=key+'_partition.json')
    eigen_details(key,run,selected_k=int(k))

def render():
    key=st.selectbox('Physical system',list(CASES),key='selected_case',format_func=lambda k:CASE_STORIES[k]['short'])
    c=CASES[key];story=CASE_STORIES[key]
    st.markdown(f'<div class="case-banner"><div><span class="tag {c["category"].lower().replace(" ","-")}">{c["category"]}</span><h1>{escape(story["short"])}</h1><p>{escape(story["system"])} · {len(frame(key)):,} observations</p></div><img alt="{escape(story["short"])} schematic" src="{image(f"docs/assets/schematics/{key}.png")}"></div>',unsafe_allow_html=True)
    a,b,c,d=st.tabs(['Scaling','Regimes','Dimensionless learning','Retrieval'])
    with a:scaling(key)
    with b:regimes(key)
    with c:learning_view(key,active_partition(key))
    with d:case_view(key)
