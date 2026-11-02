"""Rebuild local, self-contained website data and labeled diagnostic previews."""
from pathlib import Path
import sys,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from threadpoolctl import threadpool_limits
from ai_buckingham.catalog import CASES,frame,coordinates,reference
from ai_buckingham.settings import paper_settings
from ai_buckingham.experiments import jsonable,landscape,regime_diagnostic,write_result
from ai_buckingham.engine.agent import _fallback_workflow_plan,compile_workflow_plan
from ai_buckingham.settings import prompt
from ai_buckingham.showcase import CASE_STORIES,WORKFLOW
from ai_buckingham.representations import scaling_data,axis_defaults
from ai_buckingham.scopes import REGIMES

def display_payload(key,scope='global'):
    from matplotlib.mathtext import math_to_image
    import matplotlib as mpl
    d=scaling_data(key,scope);folder=ROOT/'docs/assets/formulas';folder.mkdir(exist_ok=True)
    stem=key if scope=='global' else key+'_'+scope
    assets=[]
    mpl.rcParams['mathtext.fontset']='stix'
    for index,formula in enumerate([d['formula'],d['output_formula']]):
        if not formula:continue
        filename=stem+'_'+str(index)+'.svg'
        math_to_image('$'+formula.replace(r'\widetilde\Pi',r'\widetilde{\Pi}')+'$',folder/filename,format='svg',dpi=200)
        assets.append('assets/formulas/'+filename)
    df=d['frame'];groups=next((df[n].astype(str).tolist() for n in ('case','source') if n in df and df[n].nunique()<20),['Observations']*len(df))
    if scope!='global':groups=['Regime '+scope+' · '+REGIMES[key][scope]['name']]*len(df)
    return dict(x=d['x'].tolist(),y=d['y'].tolist(),raw_x=d['raw_x'].tolist(),raw_y=d['raw_y'].tolist(),
        raw_label=d['raw_label'],group=groups,rows=d['rows'].tolist(),axis_defaults=axis_defaults(key,scope),formula_assets=assets,n_scope=len(d['rows']))

def main():
    (ROOT/'results/previews').mkdir(parents=True,exist_ok=True)
    (ROOT/'configs/paper').mkdir(parents=True,exist_ok=True)
    (ROOT/'docs/data').mkdir(exist_ok=True)
    payload={'cases':{},'reference':reference(),'workflow':WORKFLOW}
    with threadpool_limits(limits=1):
        for key,c in CASES.items():
            hp=paper_settings(key);plan=_fallback_workflow_plan(prompt(key),hp)
            compiled=compile_workflow_plan(plan,len(c['basis']),hp)
            write_result({'case':key,'source':'Manuscript / SI Tables 5–6','manual_hyperparameters':hp,'resolved_workflow':compiled},ROOT/f'configs/paper/{key}.json')
            path=ROOT/f'results/previews/{key}_landscape.json'
            if not path.exists():
                print('Landscape:',key,flush=True);write_result(landscape(key,25),path)
            df,lp,x,y,rows=coordinates(key);Y=frame(key).loc[rows,c['output']].to_numpy(float)
            group=next((df[n].astype(str).tolist() for n in ('case','source') if n in df and df[n].nunique()<20),['Observations']*len(df))
            payload['cases'][key]={**c,**CASE_STORIES[key],'n_raw':len(frame(key)),'n_clean':len(df),'x':x.tolist(),'y':y.tolist(),'raw_x':np.exp(lp[:,0]).tolist(),'raw_y':Y.tolist(),'group':group,'rows':rows.tolist(),'landscape':json.loads(path.read_text(encoding='utf-8'))}
            if key not in REGIMES:payload['cases'][key].update(display_payload(key))
            else:
                payload['cases'][key]['scopes']={}
                for scope,definition in REGIMES[key].items():
                    prefix=f'results/previews/{key}_{scope}'
                    payload['cases'][key]['scopes'][scope]={**display_payload(key,scope),'name':definition['name'],
                        'landscape':json.loads((ROOT/(prefix+'_landscape.json')).read_text()),
                        'trajectory':json.loads((ROOT/(prefix+'_trajectory.json')).read_text())}
            trajectory=ROOT/f'results/previews/{key}_trajectory.json'
            if trajectory.exists():payload['cases'][key]['trajectory']=json.loads(trajectory.read_text(encoding='utf-8'))
            shutil.copy2(ROOT/'data'/c['file'],ROOT/'docs/data'/c['file'])
        for key in ('cess','pipe'):
            path=ROOT/f'results/previews/{key}_regimes.json'
            if not path.exists():
                print('Gradient diagnostic:',key,flush=True);write_result(regime_diagnostic(key,True),path)
    (ROOT/'docs/assets/research-data.js').write_text('window.RESEARCH = '+json.dumps(jsonable(payload),separators=(',',':'),ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    (ROOT/'docs/assets/plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    (ROOT/'docs/.nojekyll').touch()
    print('Website assets and paper configurations ready.',flush=True)

if __name__=='__main__':main()
