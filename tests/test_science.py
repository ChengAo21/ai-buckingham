import json
import hashlib
import numpy as np
import pytest
from ai_buckingham.catalog import ROOT,CASES,prepare_data,coordinates,reference
from ai_buckingham.settings import paper_settings,prompt
from ai_buckingham.engine.information import global_calc_MI,normalized_information_score
from ai_buckingham.engine.agent import _fallback_workflow_plan,compile_workflow_plan
from ai_buckingham.engine.basis import calc_basis_tool

@pytest.mark.parametrize('key',list(CASES))
def test_si_basis_is_complete_dimensionless_and_finite(key):
    c=CASES[key];D=np.array(c['dimensions']).T;B=np.array(c['basis'])
    assert np.linalg.matrix_rank(B)==B.shape[0]==D.shape[1]-np.linalg.matrix_rank(D)
    np.testing.assert_allclose(D@B.T,0,atol=1e-12)
    df,lp,x,y,rows=coordinates(key)
    assert len(rows)==len(x)==len(y)>20
    assert np.isfinite(lp).all() and (x>0).all() and (y>0).all()

@pytest.mark.parametrize('key',list(CASES))
def test_paper_precedence_reaches_compiled_policy(key):
    hp=paper_settings(key);compiled=compile_workflow_plan(_fallback_workflow_plan(prompt(key),hp),len(CASES[key]['basis']),hp)
    cluster=compiled['resolved_gradient_regime_options']['clustering']
    assert cluster['min_cluster_samples']==60
    assert cluster['max_intercluster_cosine']==.90
    assert compiled['template']=='auto_mechanism_adaptive'
    assert hp['max_mi_drop_percent']==.10
    assert hp['cma_popsize']==300 and hp['cma_maxiter']==30000
    assert hp['gradient_regime_options']['surrogate']['n_restarts_optimizer']==5
    assert hp['gradient_regime_options']['surrogate']['optimizer_maxiter']==500
    if key=='pressure': assert compiled['local_workflow']['similarity_type']=='first_kind'
    else: assert compiled['local_workflow']['similarity_type']=='auto'

def test_blasius_and_velocity_wall_rows_preserve_original_filter():
    assert len(prepare_data('blasius')[0])==196
    assert len(prepare_data('velocity')[0])==1103

def test_binning_mi_recovers_entropy_for_exact_copy():
    x=np.repeat(np.arange(4),100).reshape(-1,1)
    assert global_calc_MI(x,x,'binning',4)==pytest.approx(np.log(4))
    assert normalized_information_score(-.1)==1
    assert normalized_information_score(np.log(4))==pytest.approx(.25)

def test_exact_symbolic_basis_matches_dimensional_constraint():
    c=CASES['mhd'];D=np.array(c['dimensions']).T.tolist()
    result=calc_basis_tool.invoke({'D_in':D,'var_names':c['inputs']})
    assert result['status']=='success'
    assert result['null_space_residual_inf']<1e-12

def test_original_data_hashes_unchanged():
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']

def test_failed_validation_is_retained_in_reference():
    record=next(r for r in reference()['stages'] if r['case']=='pipe' and r['regime']=='1' and r['stage']=='Rationalization')
    assert record['r2']==pytest.approx(-3.079)

def test_export_redacts_keys_and_nonfinite():
    from ai_buckingham.experiments import jsonable
    assert jsonable({'OPENAI_API_KEY':'not-a-real-key','value':np.inf,'array':np.array([1,2])})=={'value':None,'array':[1,2]}

def test_model_setup_respects_manuscript_and_endpoint_alias(monkeypatch):
    from ai_buckingham.engine import shared
    captured={}
    monkeypatch.setenv('OPENAI_API_KEY','unit-test-placeholder')
    monkeypatch.setenv('OPENAI_API_BASE','https://example.invalid/v1')
    monkeypatch.delenv('OPENAI_BASE_URL',raising=False)
    monkeypatch.setattr(shared,'ChatOpenAI',lambda **kw:captured.update(kw))
    shared.setup_environment('gpt-5-mini')
    assert captured['temperature']==0
    assert captured['extra_body']=={'temperature':0}
    assert captured['base_url']=='https://example.invalid/v1'

def test_api_basis_constraints_remain_in_si_coordinates_after_tool_reordering():
    from types import SimpleNamespace
    from ai_buckingham.engine.agent import adaptive_basis_executor_node
    c=CASES['velocity'];order=list(reversed(range(len(c['inputs']))))
    message=SimpleNamespace(tool_calls=[{'name':'calc_basis_tool','args':{'D_in':np.array(c['dimensions']).T[:,order].tolist(),'var_names':[c['inputs'][i] for i in order]}}])
    state={'basis_tool_messages':[message],'csv_FilePath':str(ROOT/'data'/c['file']),'analyzed_output_var':c['output'],'prompt_problem':prompt('velocity'),'manual_hyperparams':paper_settings('velocity')}
    result=adaptive_basis_executor_node(state)
    assert result['workflow_status']=='basis_ready'
    assert result['input_vars']==c['inputs']
    np.testing.assert_allclose(result['basis_pi_matrices'],c['basis'])
    assert 'original_tool_result' in result['basis_tool_history'][0]

def test_real_cma_trace_records_monotone_best_objective():
    from ai_buckingham.experiments import optimize_slice
    run=optimize_slice('mhd',generations=3,population=12)
    trace=run['trace']
    assert len(trace)==3
    assert [v['evaluations'] for v in trace]==[12,24,36]
    assert np.all(np.diff([v['best_objective'] for v in trace])<=1e-12)

def test_numerical_smoke_without_api(monkeypatch):
    from threadpoolctl import threadpool_limits
    from ai_buckingham.experiments import numerical_workflow
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with threadpool_limits(limits=1):run=numerical_workflow('blasius',quick=True)
    assert run['mode']=='quick_numerical'
    assert run['result']['status']=='success'

def test_scatter_markers_are_large_and_categories_are_distinct():
    from ai_buckingham.charts import scatter
    chart=scatter([1,2,3],[2,3,4],'x','y',['Outer','Intermediate','Near-wall'])
    assert {trace.marker.color for trace in chart.data}=={'#d55e00','#009e73','#0072b2'}
    assert len({trace.marker.symbol for trace in chart.data})==3
    assert all(trace.marker.size>=9 for trace in chart.data)

def test_interface_source_is_english_and_uses_descriptive_references():
    import re
    for path in [ROOT/'app.py',ROOT/'docs/index.html',ROOT/'docs/assets/site.js']:
        text=path.read_text(encoding='utf-8')
        assert not re.search('[\u4e00-\u9fff]',text)
        assert not re.search(r'(?:Fig\.|Figure|Table) [0-9]',text)

def test_case_artwork_covers_every_case():
    from PIL import Image
    for key in CASES:
        with Image.open(ROOT/f'docs/assets/schematics/{key}.png') as im:assert im.size==(1200,520)

def test_landscape_views_share_the_same_mi_and_real_trajectory():
    from ai_buckingham.charts import surface
    p=json.loads((ROOT/'results/previews/mhd_landscape.json').read_text())
    run=json.loads((ROOT/'results/previews/mhd_trajectory.json').read_text())
    contour=surface(p,False,run['trace']);three_d=surface(p,True,run['trace'])
    np.testing.assert_allclose(contour.data[0].z,three_d.data[0].z)
    assert len(contour.data)==len(three_d.data)==4
    assert contour.data[-1].marker.symbol=='diamond'
    assert three_d.data[0].colorscale==contour.data[0].colorscale
    assert contour.data[0].contours.size<(np.max(contour.data[0].z)-np.min(contour.data[0].z))/40
    assert contour.layout.yaxis.scaleanchor=='x'
    assert contour.layout.legend.y<0 and three_d.layout.legend.y<0
    assert three_d.layout.scene.aspectratio.z==.82
