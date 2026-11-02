import json
import numpy as np
import pytest
from ai_buckingham.catalog import ROOT,CASES

def test_global_selection_without_labels_and_alternative_candidates():
    from ai_buckingham.ui.explorer import selected_labels
    result=json.loads((ROOT/'results/computed/mhd_regimes.json').read_text())['result']
    assert result['best_k']==1
    labels=selected_labels(result)
    assert len(labels)==len(result['feature_coordinates'])
    assert set(labels)=={0}
    for k in ['2','3']:
        np.testing.assert_array_equal(selected_labels(result,'K = '+k),result['candidate_partitions'][k]['labels'])

@pytest.mark.parametrize('key',list(CASES))
def test_all_case_surfaces_have_small_start_and_end_markers(key):
    from ai_buckingham.charts import surface
    from ai_buckingham.experiments import display_trace
    run=json.loads((ROOT/f'results/previews/{key}_trajectory.json').read_text())
    grid=json.loads((ROOT/f'results/previews/{key}_landscape.json').read_text())
    fig=surface(grid,False,display_trace(run),run['settings']['basis_axes'])
    assert [trace.name for trace in fig.data][-2:]==['Start','End']
    assert fig.data[-2].marker.size<=8 and fig.data[-1].marker.size<=10
    assert len(run['initial']['mean_parameters'])==len(CASES[key]['basis'])

def test_arbitrary_search_axes_preserve_fixed_coefficients():
    from ai_buckingham.experiments import optimize_slice,display_trace,landscape
    from ai_buckingham.charts import surface
    run=optimize_slice('pressure',generations=2,population=6,axis_i=2,axis_j=0)
    assert run['settings']['basis_axes']==[2,0]
    for item in run['trace']:
        assert item['mean_parameters'][1]==CASES['pressure']['coefficients'][1]
        assert item['mean_parameters'][3]==CASES['pressure']['coefficients'][3]
    fig=surface(landscape('pressure',5,2,0),False,display_trace(run),[2,0])
    assert fig.data[-2].x[0]==run['initial']['mean_parameters'][2]
    assert fig.data[-2].y[0]==run['initial']['mean_parameters'][0]

def test_knowledge_retrieval_is_actual_and_network_free(monkeypatch):
    from ai_buckingham.engine import knowledge
    from ai_buckingham.retrieval import local_case_retrieval,knowledge_records
    monkeypatch.setattr(knowledge.ParateraEmbeddingClient,'__init__',lambda *a,**k:pytest.fail('No embedding call is allowed in local retrieval.'))
    result=local_case_retrieval('mhd')
    assert set(result['quantities']['matches_by_variable'])==set(CASES['mhd']['inputs'])
    assert any(result['quantities']['matches_by_variable'].values())
    assert all(r['retrieval_mode']=='lexical_only' for r in result['relations']['matches'])
    assert len(knowledge_records('quantities'))==247
    assert len(knowledge_records('relations'))==251

def test_validation_panel_uses_metric_subplots_and_stage_colors():
    from ai_buckingham.charts import validation_panel,STAGE_COLORS
    from ai_buckingham.catalog import reference
    fig=validation_panel([r for r in reference()['stages'] if r['case']=='mhd'])
    assert {t.xaxis for t in fig.data}=={'x','x2'}
    assert {t.marker.color for t in fig.data}==set(STAGE_COLORS.values())
    assert fig.layout.height<=370
