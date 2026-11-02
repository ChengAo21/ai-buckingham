import json
from copy import deepcopy
import numpy as np
import pytest
from ai_buckingham.catalog import ROOT,CASES
from ai_buckingham.ui.eigenpairs import eigenpair_records

@pytest.mark.parametrize('key',list(CASES))
def test_calculated_details_preserve_eigenpairs_and_dimensional_null_space(key):
    result=json.loads((ROOT/f'results/computed/{key}_regimes.json').read_text())['result']
    original=deepcopy(result);basis=np.asarray(result['schmidt_basis'])
    D=np.asarray(CASES[key]['dimensions']).T
    for k in [1]+[c['k'] for c in result['candidate_summary']]:
        records=eigenpair_records(result,k)
        stored=[result['single_regime_baseline']] if k==1 else next(c['clusters'] for c in result['candidate_summary'] if c['k']==k)
        assert len(records)==len(stored)
        for record,source in zip(records,stored):
            np.testing.assert_array_equal(record['eigenvalues'],source['eigenvalues'])
            np.testing.assert_array_equal(record['screening'],source['eigenvectors_in_log_base_pi'])
            np.testing.assert_allclose(record['physical'],basis.T@record['screening'],atol=1e-14)
            np.testing.assert_allclose(D@record['physical'],0,atol=1e-12)
    assert result==original

def test_eigenpair_panel_uses_calculated_results_without_duplicate_summary():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    at.session_state['selected_case']='pipe';at.radio[0].set_value('Explorer').run()
    assert not at.exception,[e.message for e in at.exception]
    assert 'Gradient eigenpair details' in {e.label for e in at.expander}
    assert not {'Reported assessment','Current calculation','Screening eigenvectors','Physical exponent directions'} & {t.label for t in at.tabs}
    spectra=next(d.value for d in at.dataframe if 'λ1' in d.value.columns)
    assert list(spectra.columns)==['Regime','λ1','λ2']
    assert set(spectra['Regime'])=={'Regime A','Regime B'}
