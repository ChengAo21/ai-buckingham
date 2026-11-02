"""Reproducible local experiments and JSON logging. No network calls here."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import numpy as np
from .catalog import CASES, ROOT, prepare_data, coordinates
from .settings import paper_settings

def jsonable(obj):
    if isinstance(obj, dict): return {str(k):jsonable(v) for k,v in obj.items() if not any(x in str(k).upper() for x in ('API_KEY','TOKEN','PASSWORD','SECRET'))}
    if isinstance(obj, (list,tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj,np.ndarray): return jsonable(obj.tolist())
    if isinstance(obj,np.generic): return jsonable(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return None
    if isinstance(obj,Path): return str(obj)
    if isinstance(obj,str):return re.sub(r'\b(?:sk-|lsv2_)[A-Za-z0-9_-]{18,}','[REDACTED]',obj)
    if obj is None or isinstance(obj,(int,float,bool)): return obj
    return jsonable(str(obj))

def write_result(result, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(result),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def provenance(case_id, mode, settings, dataset_path=None):
    return dict(case=case_id, mode=mode, timestamp=datetime.now(timezone.utc).isoformat(), seed=42,
                dataset_sha256=hashlib.sha256((Path(dataset_path) if dataset_path else ROOT/'data'/CASES[case_id]['file']).read_bytes()).hexdigest(), settings=settings,
                source='computed_in_this_repository', manuscript_reference=False)

def landscape(case_id, resolution=31, axis_i=0, axis_j=1, span=1.0, log_mi=False, scope='global', partition=None):
    """An actual MI objective slice, fixing all other coefficients at paper values.

    This is not an original optimization trajectory or a search over all coefficients.
    """
    from .engine.information import global_calc_MI
    from .scopes import scope_data
    d=scope_data(case_id,scope,partition);logpi=d['logpi'];ys=d['response'];rows=d['rows']
    base=d['coefficients'];axis_i=int(axis_i);axis_j=int(axis_j)
    if axis_i==axis_j or min(axis_i,axis_j)<0 or max(axis_i,axis_j)>=len(base):raise ValueError('Choose two distinct valid coefficients.')
    method=d['method'];param=d['param']
    a=np.linspace(base[axis_i]-span,base[axis_i]+span,resolution); b=np.linspace(base[axis_j]-span,base[axis_j]+span,resolution)
    z=np.zeros((resolution,resolution)); target=np.log(ys) if log_mi else ys
    for j,y in enumerate(b):
        for i,x in enumerate(a):
            coeff=base.copy();coeff[axis_i]=x;coeff[axis_j]=y
            values=logpi@coeff if log_mi else np.exp(np.clip(logpi@coeff,-700,700))
            z[j,i]=-float(global_calc_MI(values[:,None],target[:,None],method,param))
    return dict(x=a.tolist(),y=b.tolist(),z=z.tolist(),axis_i=axis_i,axis_j=axis_j,method=method,param=param,
                base=base.tolist(),log_mi=log_mi,rows=len(rows),scope=scope,scope_name=d['name'],scope_fingerprint=d['fingerprint'],
                csv_row_indices=rows.tolist(),provenance='recomputed_objective_slice')

def regime_diagnostic(case_id, preview=False):
    from .engine.regimes import detect_gradient_regimes
    c=CASES[case_id];df,X,Y,rows=prepare_data(case_id);hp=paper_settings(case_id)
    if preview:
        hp['gradient_regime_options']['surrogate'].update(max_train_samples=220,n_restarts_optimizer=0,optimizer_maxiter=100)
        hp['gradient_regime_options']['clustering'].update(n_init=8)
    random.seed(42);np.random.seed(42)
    result=detect_gradient_regimes(X,Y,c['basis'],gradient_regime_options=hp['gradient_regime_options'],workflow_template=hp['workflow_template'],min_cluster_samples=60)
    return dict(**provenance(case_id,'preview_regime_diagnostic' if preview else 'paper_regime_diagnostic',hp),csv_row_indices=rows.tolist(),result=result)

def numerical_workflow(case_id, quick=False, *, csv_path=None, settings=None, problem_prompt=None, definition=None):
    """Run the original adaptive inference with a verified, supplied dimension basis.

    quick is an explicitly different budget and never labeled paper reproduction.
    """
    from .engine.agent import _fallback_workflow_plan, compile_workflow_plan, execute_compiled_workflow
    from .settings import prompt
    c=CASES[case_id];hp=deepcopy(settings) if settings is not None else paper_settings(case_id)
    data_path=Path(csv_path) if csv_path else ROOT/'data'/c['file']
    import pandas as pd
    if definition is not None:
        from .definitions import validate_definition
        definition=validate_definition(definition,pd.read_csv(data_path,nrows=1).columns)
        c={**c,'inputs':definition['inputs'],'output':definition['output'],'basis':definition['basis']}
        hp.pop('_benchmark_case_id',None)
        hp['output_var']=c['output']
    missing=set(c['inputs']+[c['output']])-set(pd.read_csv(data_path,nrows=1).columns)
    if missing:raise ValueError('Dataset does not match the selected dimensional basis: '+', '.join(sorted(missing)))
    if quick:
        hp.update(cma_popsize=24,cma_maxiter=30,cma_restarts=1,regime_cma_restarts=1,enable_gpr_validation=False,enable_score_uq=False)
        hp['gradient_regime_options']['surrogate'].update(max_train_samples=180,n_restarts_optimizer=0,optimizer_maxiter=80)
        hp['gradient_regime_options']['clustering'].update(n_init=5)
    random.seed(42);np.random.seed(42)
    plan=_fallback_workflow_plan(problem_prompt or prompt(case_id),hp)
    compiled=compile_workflow_plan(plan,len(c['basis']),hp)
    result=execute_compiled_workflow(compiled,str(data_path),c['inputs'],c['output'],c['basis'],hp)
    return dict(**provenance('custom' if definition else case_id,'quick_numerical' if quick else 'numerical',hp,data_path),definition=definition,plan=jsonable(plan.model_dump()),compiled=jsonable(compiled),result=jsonable(result))

def optimize_slice(case_id, generations=30, population=24, sigma=.35, axis_i=0, axis_j=1, scope='global', partition=None, span=None):
    """A new CMA-ES run on the same two-dimensional slice shown by landscape()."""
    from cma import CMAEvolutionStrategy
    from .engine.information import global_calc_MI
    from .scopes import scope_data
    c=CASES[case_id];d=scope_data(case_id,scope,partition);lp=d['logpi'];y=d['response'];rows=d['rows'];base=d['coefficients']
    axis_i=int(axis_i);axis_j=int(axis_j)
    if axis_i==axis_j or min(axis_i,axis_j)<0 or max(axis_i,axis_j)>=len(base):raise ValueError('Choose two distinct valid coefficients.')
    axes=[axis_i,axis_j]
    method=d['method'];param=d['param']
    def objective(p):
        co=base.copy();co[axes]=p;x=np.exp(np.clip(lp@co,-700,700))
        return -float(global_calc_MI(x[:,None],y[:,None],method,param))
    def full(p):
        co=base.copy();co[axes]=p;return co.tolist()
    np.random.seed(42)
    initial=base[axes]+.12
    initial_mi=-objective(initial)
    options=dict(seed=42,popsize=int(population),maxiter=int(generations),verbose=-9,verb_log=0)
    if span is not None:options['bounds']=[(base[axes]-float(span)).tolist(),(base[axes]+float(span)).tolist()]
    es=CMAEvolutionStrategy(initial,sigma,options)
    trace=[]
    while not es.stop():
        solutions=es.ask();values=[objective(p) for p in solutions];es.tell(solutions,values)
        # Mean MI is a separate display diagnostic, not an optimizer evaluation.
        trace.append(dict(generation=es.countiter,evaluations=es.countevals,best_objective=float(es.result.fbest),best_parameters=full(es.result.xbest),mean_parameters=full(es.mean),mean_mi=-objective(es.mean),sigma=float(es.sigma)))
    settings=dict(generations=generations,population=population,sigma=sigma,initial_coefficients=initial.tolist(),fixed_input_coefficients=base.tolist(),output_coefficients=c.get('output_coefficients',np.zeros(len(base)).tolist()),basis_axes=axes,mi_method=method,mi_parameter=param,log_mi=False,coefficient_half_span=span)
    return dict(**provenance(case_id,'interactive_CMA_slice',settings),scope=scope,scope_name=d['name'],scope_fingerprint=d['fingerprint'],
                csv_row_indices=rows.tolist(),initial=dict(mean_parameters=full(initial),mean_mi=initial_mi),trace=trace,diagnostic_evaluations=len(trace)+1)

def display_trace(run):
    if not run:return None
    return ([run['initial']] if run.get('initial') else [])+run.get('trace',[])

def api_workflow(case_id=None, *, csv_path=None, problem_prompt=None, settings=None):
    """Explicit API-backed run; settings and prompt may be sent to configured providers."""
    from dotenv import load_dotenv
    from .settings import prompt
    load_dotenv(ROOT/'.env',override=False)
    from .engine.agent import build_adaptive_agent_graph
    import os
    hp=deepcopy(settings) if settings is not None else paper_settings(case_id)
    data_path=Path(csv_path) if csv_path else ROOT/'data'/CASES[case_id]['file']
    problem=problem_prompt or prompt(case_id)
    if case_id is None:hp.pop('_benchmark_case_id',None)
    graph=build_adaptive_agent_graph()
    result=graph.invoke({'messages':[], 'prompt_problem':problem, 'csv_FilePath':str(data_path), 'manual_hyperparams':hp},config={'recursion_limit':60})
    # Export the numerical evidence only, never messages containing provider metadata.
    reported_hp=deepcopy(hp)
    reported_hp['api_configuration']={name:os.getenv(name,default) for name,default in [('ANALYZER_MODEL','gpt-5-mini'),('PLANNER_MODEL','gpt-5-mini'),('OPENAI_BASE_URL',os.getenv('OPENAI_API_BASE','https://api.chatanywhere.tech/v1')),('PARATERA_BASE_URL','https://llmapi.paratera.com/v1')]}
    reported_hp['api_configuration']['requested_temperature']=0.0
    return dict(**provenance(case_id or 'custom','API_agent',reported_hp,data_path),result=jsonable({k:v for k,v in result.items() if k not in ('messages',)}))
