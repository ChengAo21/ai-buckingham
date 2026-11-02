import hashlib
from copy import deepcopy
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from ai_buckingham.catalog import ROOT
from ai_buckingham.settings import paper_settings
from ai_buckingham.execution import connection_environment,execute

@pytest.fixture
def tmp_path():
    with TemporaryDirectory(prefix='buckingham-test-') as directory:yield Path(directory)

def test_connection_configuration_is_session_scoped(monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL','https://old.invalid/v1')
    env=connection_environment({'OPENAI_BASE_URL':'https://new.invalid/v1','OPENAI_API_KEY':'session-key'})
    import os
    assert os.environ['OPENAI_BASE_URL']=='https://old.invalid/v1'
    assert env['OPENAI_BASE_URL']=='https://new.invalid/v1'
    assert env['OPENAI_API_KEY']=='session-key'
    with pytest.raises(ValueError):connection_environment({'OPENAI_BASE_URL':'https://user:pass@example.invalid'})

def test_uploaded_data_reaches_isolated_numerical_worker():
    hp=paper_settings('blasius')
    hp.update(cma_popsize=12,cma_maxiter=2,cma_restarts=1,regime_cma_restarts=1,enable_gpr_validation=False,enable_score_uq=False)
    hp['gradient_regime_options']['surrogate'].update(max_train_samples=100,n_restarts_optimizer=0,optimizer_maxiter=30)
    hp['gradient_regime_options']['clustering'].update(n_init=2)
    data=(ROOT/'data/dataset_blasius.csv').read_bytes()
    result=execute(data,dict(mode='numerical',case_id='blasius',prompt='Find the Blasius similarity coordinate.',settings=hp),timeout=120)
    assert result['dataset_sha256']==hashlib.sha256(data).hexdigest()
    assert result['settings']['cma_maxiter']==2
    assert result['result']['status']=='success'

def test_custom_agent_keeps_user_prompt_and_avoids_benchmark_basis(tmp_path,monkeypatch):
    from ai_buckingham import experiments
    from ai_buckingham.engine import agent
    path=tmp_path/'data.csv';path.write_text('length,speed,response\n1,2,3\n2,3,4\n')
    captured={}
    class Graph:
        def invoke(self,state,config):captured.update(state);return {'workflow_status':'complete','messages':[]}
    monkeypatch.setattr(agent,'build_adaptive_agent_graph',lambda:Graph())
    result=experiments.api_workflow(None,csv_path=path,problem_prompt='Custom physical system and units.',settings={'_benchmark_case_id':'mhd','output_var':'response'})
    assert captured['prompt_problem']=='Custom physical system and units.'
    assert captured['csv_FilePath']==str(path)
    assert '_benchmark_case_id' not in captured['manual_hyperparams']
    assert result['dataset_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()

def test_custom_numerical_definition_reaches_worker():
    import numpy as np
    import pandas as pd
    from ai_buckingham.catalog import CASES
    c=CASES['blasius']
    names={name:'input_'+str(i) for i,name in enumerate(c['inputs'])}
    names[c['output']]='response'
    data=pd.read_csv(ROOT/'data'/c['file']).rename(columns=names).to_csv(index=False).encode()
    definition=dict(inputs=[names[n] for n in c['inputs']],output='response',dimension_matrix=np.asarray(c['dimensions']).T.tolist(),basis=c['basis'])
    hp=paper_settings('blasius')
    hp.update(cma_popsize=12,cma_maxiter=2,cma_restarts=1,regime_cma_restarts=1,enable_gpr_validation=False,enable_score_uq=False,workflow_template='global')
    result=execute(data,dict(mode='numerical',case_id='blasius',definition=definition,settings=hp),timeout=120)
    assert result['case']=='custom'
    assert result['definition']==definition
    assert result['settings']['output_var']=='response'
    assert '_benchmark_case_id' not in result['settings']
    assert result['dataset_sha256']==hashlib.sha256(data).hexdigest()
    assert result['result']['status']=='success'

def test_invalid_supplied_basis_is_rejected_before_execution():
    from ai_buckingham.definitions import validate_definition
    with pytest.raises(ValueError,match='dimensionally admissible'):
        validate_definition(dict(inputs=['length','time'],output='response',dimension_matrix=[[1,1]],basis=[[1,1]]),['length','time','response'])
