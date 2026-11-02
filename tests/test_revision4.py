import json
import numpy as np
import pytest
from ai_buckingham.catalog import ROOT,CASES,prepare_data
from ai_buckingham.scopes import REGIMES,scope_data,regime_label,physical_cluster_mapping
from ai_buckingham.representations import scaling_data,axis_defaults

def test_velocity_original_coordinate_is_wall_distance_not_density_ratio():
    data=scaling_data('velocity');df=data['frame']
    np.testing.assert_allclose(data['raw_x'],df['y']*df['rho_w']*df['u_tau']/df['mu_w'])
    assert data['raw_label']=='y⁺'
    assert axis_defaults('velocity')==((True,False),(True,False))

@pytest.mark.parametrize('key,expected',[('pressure',(False,False)),('rayleigh',(True,True)),('keyhole',(False,True)),('blasius',(False,False)),('goldenfeld',(False,False))])
def test_learned_axes_follow_reported_panels(key,expected):
    assert axis_defaults(key)[1]==expected
    assert scaling_data(key)['formula'].startswith(r'\Pi_1=')

@pytest.mark.parametrize('case',list(REGIMES))
def test_regime_learning_uses_disjoint_complete_physical_subsets(case):
    rows=[]
    for letter in REGIMES[case]:
        d=scope_data(case,letter)
        grid=json.loads((ROOT/f'results/previews/{case}_{letter}_landscape.json').read_text())
        run=json.loads((ROOT/f'results/previews/{case}_{letter}_trajectory.json').read_text())
        assert grid['scope_fingerprint']==run['scope_fingerprint']==d['fingerprint']
        assert grid['csv_row_indices']==run['csv_row_indices']==d['rows'].tolist()
        assert grid['rows']==len(d['rows'])
        assert grid['method']==run['settings']['mi_method']==d['method']
        assert grid['param']==run['settings']['mi_parameter']==d['param']
        rows.extend(d['rows'])
    assert len(set(rows))==len(rows)==len(prepare_data(case)[0])
    assert set(rows)==set(prepare_data(case)[3])

def test_pipe_local_estimators_retain_the_local_histogram_override():
    assert scope_data('pipe','A')['param']==20
    assert scope_data('pipe','A')['method']=='binning'
    assert scope_data('pipe','B')['method']=='kraskov'
    assert scope_data('pipe','B')['param']==5
    assert axis_defaults('pipe','A')[1]==(True,False)
    assert axis_defaults('pipe','B')[1]==(True,True)

def test_reported_eigenpairs_are_complete_without_invented_spectra():
    data=json.loads((ROOT/'results/reference/regime_eigenpairs.json').read_text())['cases']
    assert set(data)==set(CASES)
    count=0
    for case,record in data.items():
        for candidate in record['candidates']:
            for cluster in candidate['clusters']:
                count+=1
                assert cluster['eigenvalues'] is None
                assert np.linalg.norm(cluster['leading_eigenvector'])==pytest.approx(1,abs=2e-4)
    assert count==77

def test_display_regimes_use_letters_without_mutating_raw_records():
    assert regime_label('0')=='Regime A'
    assert regime_label('1')=='Regime B'
    assert regime_label('2')=='Regime C'
    assert regime_label('global')=='Global'
    from ai_buckingham.charts import validation_panel
    from ai_buckingham.catalog import reference
    chart=validation_panel([r for r in reference()['stages'] if r['case']=='cess'])
    assert set(x for t in chart.data for x in t.x)=={'Regime A','Regime B','Regime C'}

def test_standalone_legend_clears_axis_and_triptych_has_no_duplicate_legend():
    from ai_buckingham.charts import surface
    from ai_buckingham.experiments import display_trace
    p=json.loads((ROOT/'results/previews/mhd_landscape.json').read_text())
    r=json.loads((ROOT/'results/previews/mhd_trajectory.json').read_text())
    fig=surface(p,False,display_trace(r),show_legend=False,height=390)
    assert not fig.layout.showlegend
    assert fig.layout.margin.b>=60
    standalone=surface(p,False,display_trace(r))
    assert standalone.layout.legend.y<=-.3 and standalone.layout.margin.b>=100

def test_overview_uses_original_ppt_artwork_including_the_missing_panels():
    import base64
    manifest=json.loads((ROOT/'docs/assets/workflow/original-artwork-provenance.json').read_text())
    rows=manifest['illustrations'];svg=(ROOT/'docs/assets/discovery-map.svg').read_text()
    expected={'mi-bars-original.png','sparsification-original.png','rationalization-original.png','regime-collapse-original.png','description-original.png','dataset-original.png'}
    assert expected<={r['asset'].split('/')[-1] for r in rows}
    for r in rows:
        assert r['source'].endswith('Fig2_workflow_.pptx')
        assert len(r['source_sha256'])==64 and len(r['original_image_sha256'])==64
        assert base64.b64encode((ROOT/r['asset']).read_bytes()).decode() in svg

def test_scaling_axes_are_independent_and_learning_controls_are_unobscured():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    at.session_state['selected_case']='velocity';at.radio[0].set_value('Explorer').run()
    assert not at.exception
    controls={w.key:w for w in at.toggle}
    assert controls['log_x_velocity_global_0'].value is True
    assert controls['log_x_velocity_global_1'].value is True
    assert controls['log_y_velocity_global_0'].value is False
    controls['log_x_velocity_global_0'].set_value(False).run()
    assert next(w for w in at.toggle if w.key=='log_x_velocity_global_1').value is True
    assert 'Dimensionless learning' in {t.label for t in at.tabs}
    assert not any('Stage values' in e.label for e in at.expander)
    assert not any('Landscape view'==s.label for s in at.selectbox)
