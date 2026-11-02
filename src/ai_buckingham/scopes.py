"""Physical names and row provenance for displayed regime-wise experiments."""
import hashlib
import json
import numpy as np
from .catalog import ROOT,CASES,coordinates

REGIMES={
    'cess':{
        'A':dict(name='Outer',coefficients=[0.,-1.],formula=r'\Pi_1=Re_\tau=\frac{hu_\tau}{\nu}'),
        'B':dict(name='Intermediate',coefficients=[.57,-1.],formula=r'\Pi_1=Re_\tau\eta^{0.57},\qquad\eta=y/h'),
        'C':dict(name='Near-wall',coefficients=[1.,-1.],formula=r'\Pi_1=y^+=\frac{yu_\tau}{\nu}'),
    },
    'pipe':{
        'A':dict(name='Rough turbulent',coefficients=[0.,1.],formula=r'\Pi_1=\epsilon/D'),
        'B':dict(name='Viscous',coefficients=[-1.,0.],formula=r'\Pi_1=Re^{-1}=\frac{\mu}{\rho DU}'),
    },
}

def regime_label(value):
    if str(value)=='global':return 'Global'
    value=str(value)
    letter=chr(65+int(value)) if value.isdigit() else value
    return 'Regime '+letter

def partition_result(case_id,partition=None):
    if partition is not None:return partition.get('result',partition)
    return json.loads((ROOT/f'results/computed/{case_id}_regimes.json').read_text(encoding='utf-8'))['result']

def physical_cluster_mapping(case_id,labels,physical_frame):
    """Map numerical IDs to physical names, independent of cluster-ID permutations."""
    labels=np.asarray(labels);ids=np.unique(labels)
    if len(ids)==1:return {int(ids[0]):'Global'}
    if case_id=='cess' and len(ids)==3:
        position=(physical_frame['y']/physical_frame['h']).to_numpy()
        order=sorted(ids,key=lambda k:np.median(position[labels==k]),reverse=True)
        names=['A · Outer','B · Intermediate','C · Near-wall']
    elif case_id=='pipe' and len(ids)==2:
        reynolds=(physical_frame['rho']*physical_frame['U']*physical_frame['D']/physical_frame['mu']).to_numpy()
        order=sorted(ids,key=lambda k:np.median(reynolds[labels==k]),reverse=True)
        names=['A · Rough turbulent','B · Viscous']
    else:order=ids;names=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(ids)]
    return {int(k):'Regime '+name for k,name in zip(order,names)}

def scope_data(case_id,scope='global',partition=None):
    c=CASES[case_id];df,lp,x,y,rows=coordinates(case_id)
    base=np.asarray(c['coefficients'],float);mask=np.arange(len(rows))
    name='Global';scope=str(scope)
    if scope!='global':
        if case_id not in REGIMES or scope not in REGIMES[case_id]:raise ValueError('Unknown physical regime.')
        result=partition_result(case_id,partition)
        if result.get('best_k')!=len(REGIMES[case_id]):raise ValueError('The current partition does not support these physical-regime definitions.')
        labels=np.asarray(result['labels'],int);valid=np.asarray(result['valid_indices'],int)
        names=physical_cluster_mapping(case_id,labels,df.iloc[valid])
        selected=[k for k,v in names.items() if v.startswith('Regime '+scope+' ·')]
        if len(selected)!=1:raise ValueError('No compatible physical regime was identified.')
        mask=valid[labels==selected[0]];base=np.asarray(REGIMES[case_id][scope]['coefficients'],float)
        name='Regime '+scope+' · '+REGIMES[case_id][scope]['name']
    rows=rows[mask];lp=lp[mask];y=y[mask]
    method=c['mi_method'];param=c['mi_param']
    if method=='auto':
        method='kraskov' if len(rows)<800 else 'binning'
        param=5 if method=='kraskov' else (20 if case_id=='pipe' and scope!='global' else 40)
    fingerprint=hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()
    return dict(case=case_id,scope=scope,name=name,frame=df.iloc[mask],logpi=lp,response=y,rows=rows,
                coefficients=base,method=method,param=param,fingerprint=fingerprint)
