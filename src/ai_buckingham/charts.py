"""A consistent visual grammar for the Streamlit and static research pages."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Profile colors sampled from the supplied velocity-comparison artwork.
COLORS=['#7b8cde','#e96a73','#eaa05d','#aa7bcb','#59bf78','#af805e']
SYMBOLS=['circle','square','triangle-up','diamond','triangle-down','pentagon']
# Regime colors follow the supplied physical-regime comparison.
REGIME_STYLES={
    'Outer':('#d55e00','triangle-up'),
    'Intermediate':('#009e73','diamond'),
    'Near-wall':('#0072b2','circle'),
    'Rough turbulent':('#d55e00','triangle-up'),
    'Viscous':('#0072b2','square'),
}

def regime_names(case_id, labels, physical_frame):
    """Order newly computed labels by physical location; preserve IDs in exports."""
    from .scopes import physical_cluster_mapping
    labels=np.asarray(labels);mapping=physical_cluster_mapping(case_id,labels,physical_frame)
    return np.array([mapping[k] for k in labels]),mapping

def category_style(name,index=0):
    return REGIME_STYLES.get(str(name).split(' · ')[-1],(COLORS[index%len(COLORS)],SYMBOLS[index%len(SYMBOLS)]))

def style(fig, height=530):
    fig.update_layout(template='plotly_white',height=height,font=dict(family='Arial, sans-serif',color='#263d46',size=15),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#fff',margin=dict(l=65,r=25,t=18,b=108),
        colorway=COLORS,legend=dict(orientation='h',y=-.22,yanchor='top',x=.5,xanchor='center',font=dict(size=13),title_text='',itemsizing='constant'))
    fig.update_xaxes(gridcolor='#edf1f1',zeroline=False,ticks='outside',tickfont=dict(size=14),title_font=dict(size=17),automargin=True)
    fig.update_yaxes(gridcolor='#edf1f1',zeroline=False,ticks='outside',tickfont=dict(size=14),title_font=dict(size=17),automargin=True)
    return fig

def scatter(x,y,xlabel,ylabel,color=None,logx=False,logy=False,rows=None):
    d=pd.DataFrame({'x':x,'y':y,'group': ['Observations']*len(x) if color is None else np.asarray(color).astype(str),'CSV row':np.arange(len(x)) if rows is None else rows})
    groups=d['group'].unique()
    color_map={g:category_style(g,i)[0] for i,g in enumerate(groups)}
    symbol_map={g:category_style(g,i)[1] for i,g in enumerate(groups)}
    ordered=sorted(groups) if all(g.startswith('Regime ') for g in groups) else [g for g in REGIME_STYLES if g in groups]+[g for g in groups if g not in REGIME_STYLES]
    f=px.scatter(d,x='x',y='y',color='group',symbol='group',custom_data=['CSV row'],color_discrete_map=color_map,symbol_map=symbol_map,category_orders={'group':ordered},log_x=logx,log_y=logy,render_mode='svg')
    f.update_traces(marker=dict(size=10,opacity=.86,line=dict(width=.6,color='white')),hovertemplate=xlabel+': %{x:.5g}<br>'+ylabel+': %{y:.5g}<br>CSV row: %{customdata[0]}<extra>%{fullData.name}</extra>')
    f.update_layout(xaxis_title=xlabel,yaxis_title=ylabel)
    return style(f)

def evidence_chart(rows,metric='score'):
    d=pd.DataFrame(rows);err='score_sd' if metric=='score' else 'mi_sd'
    f=px.bar(d,x='stage',y=metric,color='regime',barmode='group',error_y=err,color_discrete_sequence=COLORS,category_orders={'stage':['k = 1','k = 2','Sparsification','Rationalization']})
    f.update_layout(xaxis_title=None,yaxis_title='ε = exp(−MI) · smaller is better' if metric=='score' else 'Mutual information · nats',bargap=.32)
    return style(f)

STAGE_COLORS={'k = 1':'#70b7eb','k = 2':'#ac7be1','Sparsification':'#76bea5','Rationalization':'#e7b56d'}
def validation_panel(rows):
    from plotly.subplots import make_subplots
    from .scopes import regime_label
    figure=make_subplots(rows=1,cols=2,subplot_titles=('ε = exp[−max(MI, 0)]','Mutual information'),horizontal_spacing=.15)
    regimes=list(dict.fromkeys(r['regime'] for r in rows));multiple=len(regimes)>1
    labels={'k = 1':'1 group','k = 2':'2 groups','Sparsification':'Sparsified','Rationalization':'Rationalized'}
    for stage,color in STAGE_COLORS.items():
        selected=[r for r in rows if r['stage']==stage]
        if not selected:continue
        x=[regime_label(r['regime']) if multiple else labels[stage] for r in selected]
        for column,metric in [(1,'score'),(2,'mi')]:
            figure.add_trace(go.Bar(x=x,y=[r[metric] for r in selected],name=labels[stage],legendgroup=stage,
                showlegend=multiple and column==1,marker_color=color,width=.16 if multiple else .46,
                error_y=dict(type='data',array=[r.get(metric+'_sd',0) for r in selected],visible=True,thickness=1,width=3),
                hovertemplate='%{x}<br>%{y:.4f}<extra>'+labels[stage]+'</extra>'),row=1,col=column)
    figure.update_layout(template='plotly_white',height=350,barmode='group',bargap=.25,bargroupgap=.08,
        paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Arial',size=13,color='#36555b'),
        margin=dict(l=50,r=20,t=45,b=85 if multiple else 65),legend=dict(orientation='h',y=-.22,x=.5,xanchor='center',font=dict(size=12)))
    figure.update_xaxes(tickfont=dict(size=12),showgrid=False)
    figure.update_yaxes(gridcolor='#edf0eb',zeroline=False,tickfont=dict(size=12))
    figure.update_yaxes(title_text='ε',row=1,col=1);figure.update_yaxes(title_text='MI / nats',row=1,col=2)
    return figure

def surface(payload,three_d=True,trace=None,trace_axes=(0,1),show_legend=True,height=500):
    mi=-np.asarray(payload['z'],dtype=float)
    lo=float(np.nanmin(mi));hi=float(np.nanmax(mi))
    bar=dict(title=dict(text='MI',font=dict(size=16)),tickfont=dict(size=13),thickness=12,len=.75,x=1.02,outlinewidth=0)
    if three_d:
        f=go.Figure(go.Surface(x=payload['x'],y=payload['y'],z=mi,colorscale='Viridis',colorbar=bar,lighting=dict(ambient=.75,diffuse=.75,roughness=.8,specular=.15)))
        axis=lambda text:dict(title=dict(text=text,font=dict(size=17)),tickfont=dict(size=13),nticks=5,showbackground=True,backgroundcolor='#fafbf9',gridcolor='#dce4df',zeroline=False)
        f.update_layout(scene=dict(xaxis=axis(f"c{payload['axis_i']+1}"),yaxis=axis(f"c{payload['axis_j']+1}"),zaxis=axis('MI'),aspectmode='manual',aspectratio=dict(x=1,y=1,z=.82),camera=dict(eye=dict(x=1.55,y=1.55,z=1.18))))
    else:
        f=go.Figure(go.Contour(x=payload['x'],y=payload['y'],z=mi,colorscale='Viridis',colorbar=bar,autocontour=False,contours=dict(start=lo,end=hi,size=max((hi-lo)/45,1e-8),coloring='fill',showlines=False,showlabels=False)))
        f.update_layout(xaxis_title=f"c{payload['axis_i']+1}",yaxis_title=f"c{payload['axis_j']+1}")
        f.update_xaxes(type='linear',constrain='domain',nticks=5)
        f.update_yaxes(type='linear',scaleanchor='x',scaleratio=1,constrain='domain',nticks=5)
    if trace and {payload['axis_i'],payload['axis_j']}==set(trace_axes) and not payload.get('log_mi'):
        means=all('mean_parameters' in v for v in trace)
        key='mean_parameters' if means else 'best_parameters'
        def coordinate(v,axis):return v[key][axis] if len(v[key])>max(trace_axes) else v[key][list(trace_axes).index(axis)]
        x=np.array([coordinate(v,payload['axis_i']) for v in trace]);y=np.array([coordinate(v,payload['axis_j']) for v in trace])
        z=np.array([v['mean_mi'] if means else -v['best_objective'] for v in trace])
        if three_d:
            f.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',name='CMA mean' if means else 'Best-so-far',line=dict(color='#E63946',width=6)))
            f.add_trace(go.Scatter3d(x=x[[0]],y=y[[0]],z=z[[0]],mode='markers',name='Start',marker=dict(size=3.5,color='white',symbol='circle',line=dict(color='#19344f',width=1))))
            f.add_trace(go.Scatter3d(x=x[[-1]],y=y[[-1]],z=z[[-1]],mode='markers',name='End',marker=dict(size=4.5,color='#f4c95d',symbol='diamond',line=dict(color='#19344f',width=1))))
        else:
            f.add_trace(go.Scatter(x=x,y=y,mode='lines',name='CMA mean' if means else 'Best-so-far',line=dict(color='#E63946',width=3)))
            f.add_trace(go.Scatter(x=x[[0]],y=y[[0]],mode='markers',name='Start',marker=dict(size=8,color='white',symbol='circle',line=dict(color='#19344f',width=1.2))))
            f.add_trace(go.Scatter(x=x[[-1]],y=y[[-1]],mode='markers',name='End',marker=dict(size=10,color='#f4c95d',symbol='diamond',line=dict(color='#19344f',width=1.2))))
    style(f,height).update_layout(showlegend=show_legend,margin=dict(l=56 if not three_d else 3,r=55,t=16,b=110 if show_legend else 64),legend=dict(y=-.30))
    f.update_xaxes(title_standoff=14);f.update_yaxes(title_standoff=12)
    return f

def convergence(trace,height=270):
    f=go.Figure(go.Scatter(x=[v['generation'] for v in trace],y=[-v['best_objective'] for v in trace],mode='lines',line=dict(color='#507e74',width=2),showlegend=False))
    f.update_layout(xaxis_title='CMA-ES generation',yaxis_title='Best MI / nats')
    return style(f,height).update_layout(margin=dict(l=62,r=15,t=18,b=64))
