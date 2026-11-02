import pytest
from streamlit.testing.v1 import AppTest
from ai_buckingham.catalog import ROOT,CASES

def app():return AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()

@pytest.mark.parametrize('page',['Overview','Methods','Explorer','Run'])
def test_pages_render_without_keys(page,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    at=app();at.radio[0].set_value(page).run()
    assert not at.exception,[e.message for e in at.exception]

@pytest.mark.parametrize('key',list(CASES))
def test_integrated_case_analysis(key):
    at=app();at.session_state['selected_case']=key;at.radio[0].set_value('Explorer').run()
    assert not at.exception,[e.message for e in at.exception]
    assert {'Scaling','Regimes','Dimensionless learning','Retrieval'}<={tab.label for tab in at.tabs}

@pytest.mark.parametrize('key',list(CASES))
def test_overview_cards_open_matching_explorer(key):
    at=app();next(b for b in at.button if b.key=='overview_open_'+key).click().run()
    assert not at.exception,[e.message for e in at.exception]
    assert at.radio[0].value=='Explorer'
    assert next(w for w in at.selectbox if w.key=='selected_case').value==key

def test_filter_and_no_repeated_workflow():
    at=app();at.session_state['case_family']='Type II';at.run()
    assert {b.key for b in at.button if str(b.key).startswith('overview_open_')}=={'overview_open_velocity','overview_open_goldenfeld'}
    assert len(at.expander)==0

def test_run_has_editable_prompt_and_provider_fields():
    at=app();at.radio[0].set_value('Run').run()
    assert not at.exception,[e.message for e in at.exception]
    labels={w.label for w in at.text_input}
    assert {'LLM API base URL','LLM API key','Embedding API base URL','Embedding API key','Model'}<=labels
    assert next(w for w in at.text_area if w.label=='Problem statement').value
    assert {'Problem','API connection','Settings'}<={tab.label for tab in at.tabs}

@pytest.mark.parametrize('key',['cess','pipe'])
def test_candidate_partition_can_be_inspected(key):
    at=app();at.session_state['selected_case']=key;at.session_state['partition_'+key]='K = 2'
    at.radio[0].set_value('Explorer').run()
    assert not at.exception,[e.message for e in at.exception]


def test_supplied_basis_mode_hides_llm_inputs():
    at=app();at.radio[0].set_value('Run').run()
    next(b for b in at.button if b.key=='execution_numerical').click().run()
    assert not at.exception,[e.message for e in at.exception]
    assert 'Data & basis' in {t.label for t in at.tabs}
    assert 'API connection' not in {t.label for t in at.tabs}
    assert 'Problem statement' not in {w.label for w in at.text_area}
    assert 'LLM API base URL' not in {w.label for w in at.text_input}
    assert not at.error
    assert not next(b for b in at.button if b.label=='Run numerical inference').disabled

@pytest.mark.parametrize('key',list(CASES))
def test_all_case_configuration_editors_are_complete_and_runnable(key):
    at=app();at.session_state['run_template']=key;at.radio[0].set_value('Run').run()
    assert not at.exception,[e.message for e in at.exception]
    assert not at.error,[e.value for e in at.error]
    assert not next(b for b in at.button if b.label=='Start autonomous discovery').disabled
    assert {'Held-out GPR validation','MI and score uncertainty'}<={w.label for w in at.checkbox}
    assert len(at.number_input)>=30
