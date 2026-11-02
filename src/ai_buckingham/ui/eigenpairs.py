"""Calculated eigenpair details, without repeating partition-level summaries."""
import numpy as np
import pandas as pd
import streamlit as st
from ..catalog import CASES
from .theme import plot_title


def matrix_tex(value):
    a=np.asarray(value,float)
    return r'\begin{bmatrix}'+r'\\'.join(' & '.join(f'{v:.4f}' for v in row) for row in a)+r'\end{bmatrix}'


def eigenpair_records(result,k):
    """Return stored spectra and column-oriented directions without mutating a run."""
    if int(k)==1:
        baseline=result.get('single_regime_baseline',{})
        clusters=[dict(cluster=0,eigenvalues=baseline.get('eigenvalues',[]),
                       eigenvectors_in_log_base_pi=baseline.get('eigenvectors_in_log_base_pi',[]))]
    else:
        candidate=next((r for r in result.get('candidate_summary',[]) if int(r['k'])==int(k)),{})
        clusters=candidate.get('clusters',[])
    basis=np.asarray(result.get('schmidt_basis',[]),float)
    records=[]
    for i,cluster in enumerate(clusters):
        vectors=np.asarray(cluster.get('eigenvectors_in_log_base_pi',[]),float)
        physical=np.asarray(cluster.get('dimensionless_exponent_directions_in_physical_variables',[]),float)
        if physical.size:physical=physical.T
        elif basis.ndim==2 and vectors.ndim==2 and basis.shape[0]==vectors.shape[0]:
            physical=basis.T@vectors
        records.append(dict(name='Global' if int(k)==1 else 'Regime '+chr(65+i),
                            eigenvalues=np.asarray(cluster.get('eigenvalues',[]),float),
                            screening=vectors,physical=physical))
    return records


def render(key,run=None,selected_k=None):
    if not run:return
    result=run['result'];counts=[1]+[int(r['k']) for r in result.get('candidate_summary',[])]
    default=selected_k if selected_k in counts else result.get('best_k')
    if default not in counts:default=1
    with st.expander('Gradient eigenpair details'):
        k=st.segmented_control('Partition size',counts,default=default,
            format_func=lambda n:'Global' if n==1 else f'K = {n}',key=f'eigen_details_k_{key}_{default}') or default
        records=eigenpair_records(result,k)
        spectra=[{'Regime':r['name'],**{f'λ{j+1}':v for j,v in enumerate(r['eigenvalues'])}} for r in records]
        if not records or not any(len(r['eigenvalues']) for r in records):
            st.info('No eigenpair results are available for this partition.');return
        st.dataframe(pd.DataFrame(spectra),hide_index=True,width='stretch',
            column_config={f'λ{j+1}':st.column_config.NumberColumn(format='%.6g') for j in range(7)})
        st.caption('Columns follow decreasing eigenvalue order. V uses Schmidt-orthonormal log-Π coordinates; W gives the corresponding physical-variable directions.')
        for r in records:
            with st.container(border=True,key=f"eigen_matrix_{key}_{k}_{r['name'].replace(' ','_')}"):
                plot_title(r['name'])
                left,right=st.columns([1,1.25],gap='medium')
                with left:
                    st.caption('Screening eigenvectors')
                    if r['screening'].size:st.latex(r'\mathbf V='+matrix_tex(r['screening']))
                with right:
                    st.caption('Physical exponent directions')
                    if r['physical'].size:st.latex(r'\mathbf W='+matrix_tex(r['physical']))
        st.caption('Rows of W: '+', '.join(CASES[key]['inputs'])+'.')
        fit=result.get('gpr',{});surrogate=run.get('settings',{}).get('gradient_regime_options',{}).get('surrogate',{})
        if fit and surrogate:
            st.caption(f"GPR: {fit.get('n_train_samples','—')} training samples · {surrogate.get('n_restarts_optimizer','—')} restarts · maximum {surrogate.get('optimizer_maxiter','—')} optimizer iterations.")
        if fit.get('optimizer_success') is False:
            st.caption('The surrogate optimizer retained its best finite fit without reaching convergence.')
