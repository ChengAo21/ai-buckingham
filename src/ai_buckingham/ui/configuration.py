"""Controls for the scientific settings actually passed to workflow execution."""
import json
from copy import deepcopy
import streamlit as st

def editor(configuration,prefix):
    hp=deepcopy(configuration)
    hp.setdefault('enable_gpr_validation',True)
    def number(container,key,label,low,high,step=None):
        value=container[key]
        kw=dict(min_value=low,max_value=high,value=value,key=prefix+'::'+key)
        if step is not None:kw['step']=step
        container[key]=st.number_input(label,**kw)
    def choice(container,key,label,options,labels=None):
        value=container.get(key,options[0]);index=options.index(value) if value in options else 0
        container[key]=st.selectbox(label,options,index=index,format_func=(lambda v:labels.get(v,v)) if labels else str,key=prefix+'::'+key)
    route,search,geometry,refine,validate,advanced=st.tabs(['Route & MI','CMA-ES','GPR & clustering','Formulations','Validation & uncertainty','Complete configuration'])
    with route:
        choice(hp,'workflow_template','Physical-regime policy',['auto_mechanism_adaptive','single_regime_adaptive','multi_regime_adaptive'],{'auto_mechanism_adaptive':'Autonomous global / multiple assessment','single_regime_adaptive':'Prescribed global regime','multi_regime_adaptive':'Prescribed multiple regimes'})
        choice(hp,'similarity_type','Similarity policy',['auto','first_kind','second_kind'],{'auto':'Autonomous Type I / II screening','first_kind':'Type I · complete similarity','second_kind':'Type II · incomplete similarity'})
        method=st.selectbox('MI estimator',['auto','kraskov','binning'],index=['auto','kraskov','binning'].index(hp.get('mi_method_used','auto')),format_func=lambda x:{'auto':'Automatic','kraskov':'KSG nearest-neighbor','binning':'Histogram'}[x],key=prefix+'::mi_method')
        if method=='auto':hp.pop('mi_method_used',None);hp.pop('mi_param_used',None)
        else:
            hp['mi_method_used']=method
            hp['mi_param_used']=int(st.number_input('Neighbors ω' if method=='kraskov' else 'Histogram bins b',min_value=2,max_value=200,value=int(hp.get('mi_param_used',5 if method=='kraskov' else 40)),key=prefix+'::mi_param'))
        if 'regime_histogram_bins' in hp:number(hp,'regime_histogram_bins','Local-regime histogram bins',2,200)
        number(hp,'abs_score_drop_threshold','Minimum additional-group score reduction Δε',0.,1.,.01)
        a,b=st.columns(2)
        with a:number(hp,'regime_min_samples','Minimum local MI samples',2,10000)
        with b:number(hp,'regime_min_samples_per_mi_dimension','Samples required per MI dimension',2,1000)
        number(hp,'workflow_minimum_samples_for_second_count','Samples required to test a second group',2,10000)
        st.caption('Automatic MI selection uses KSG below 800 samples and histograms otherwise. Candidate comparisons retain the same estimator configuration.')
    with search:
        a,b,c=st.columns(3)
        with a:number(hp,'cma_popsize','Population',6,2000)
        with b:number(hp,'cma_maxiter','Maximum generations',1,100000)
        with c:number(hp,'cma_restarts','Independent global runs',1,30)
        a,b,c=st.columns(3)
        with a:number(hp,'regime_cma_restarts','Independent local runs',1,30)
        with b:number(hp,'cma_init_position','Initial coefficient',-10.,10.,.05)
        with c:number(hp,'cma_init_stepsize','Initial search scale',.001,10.,.05)
        a,b=st.columns(2)
        with a:number(hp,'cma_bounds_min','Lower coefficient bound',-100.,100.,.1)
        with b:number(hp,'cma_bounds_max','Upper coefficient bound',-100.,100.,.1)
        choice(hp,'normalize_input','Input-coefficient normalization',['none','norm2'],{'none':'None','norm2':'Unit Euclidean norm'})
        number(hp,'max_cosine_similarity','Maximum group-direction cosine',0.,1.,.05)
        penalty=st.selectbox('Group-direction collinearity penalty',['Automatic','Enabled','Disabled'],index=0 if 'use_collinearity_penalty' not in hp else 1 if hp['use_collinearity_penalty'] else 2,key=prefix+'::collinearity')
        if penalty=='Automatic':hp.pop('use_collinearity_penalty',None)
        else:hp['use_collinearity_penalty']=penalty=='Enabled'
        hp['penalty_opt_w_fixed']=st.checkbox('Include output-scaling directions in the penalty',value=bool(hp.get('penalty_opt_w_fixed',True)),key=prefix+'::output_penalty')
        st.caption('Random seed: 42. Global and local run counts are independent settings.')
    with geometry:
        surrogate=hp['gradient_regime_options']['surrogate'];cl=hp['gradient_regime_options']['clustering']
        st.markdown('**GPR posterior-mean gradients**')
        limited=st.checkbox('Limit surrogate training samples',value=surrogate.get('max_train_samples') is not None,key=prefix+'::limit_gpr')
        surrogate['max_train_samples']=int(st.number_input('Surrogate training limit',min_value=20,max_value=100000,value=int(surrogate.get('max_train_samples') or 800),key=prefix+'::surrogate_limit')) if limited else None
        a,b=st.columns(2)
        with a:number(surrogate,'n_restarts_optimizer','GPR optimizer restarts',0,30)
        with b:number(surrogate,'optimizer_maxiter','GPR optimizer iteration limit',10,10000)
        choice(surrogate,'mean_mode','GPR mean function',['constant','affine_sigmoid_transition'],{'constant':'Constant','affine_sigmoid_transition':'Affine–sigmoid transition'})
        choice(surrogate,'regime_target_transform','Regime-surrogate response',['log','raw'],{'log':'log Πy','raw':'Πy'})
        st.markdown('**Physical-regime clustering**')
        cl['candidate_cluster_counts']=st.multiselect('Candidate cluster counts',list(range(2,9)),default=cl['candidate_cluster_counts'],key=prefix+'::candidate_k')
        a,b,c=st.columns(3)
        with a:number(cl,'min_cluster_samples','Minimum samples per cluster',2,10000)
        with b:number(cl,'min_cluster_fraction','Minimum cluster fraction',.001,.5,.01)
        with c:number(cl,'eigenvalue_ratio_threshold','Minimum λ1 / λn',1.,10000.,1.)
        a,b,c=st.columns(3)
        with a:number(cl,'max_intercluster_cosine','K = 2 directional-separation threshold',0.,1.,.01)
        with b:number(cl,'n_init','Clustering initializations',1,300)
        with c:number(cl,'max_iter','Clustering iteration limit',1,10000)
        cl['require_convergence']=st.checkbox('Require converged assignments',value=cl['require_convergence'],key=prefix+'::converged')
    with refine:
        st.markdown('**Type I / II screening**')
        number(hp,'similarity_screening_output_exponent_threshold','Negligible output exponent',0.,1.,.01)
        number(hp,'similarity_screening_max_relative_mi_change','Maximum MI loss without output rescaling',0.,1.,.01)
        number(hp,'similarity_screening_max_cosine_similarity','Screening direction-cosine limit',0.,1.,.05)
        st.markdown('**Sparsification and rationalization**')
        a,b=st.columns(2)
        with a:number(hp,'regime_sparsify_exponent_threshold','Physical-exponent removal threshold',0.,1.,.01)
        with b:number(hp,'regime_sparsify_max_score_increase','Maximum sparsification score increase',0.,1.,.01)
        a,b,c=st.columns(3)
        with a:number(hp,'max_rational_denominator','Maximum rational denominator',1,50)
        with b:number(hp,'max_exponent_error','Maximum rational approximation error',0.,1.,.01)
        with c:number(hp,'max_mi_drop_percent','Maximum rationalization MI loss',0.,1.,.01)
        st.markdown('**Fixed powers in the Buckingham–Π basis**')
        for field,label in [('fixed_in_power','Fixed input-group powers'),('fixed_out_power','Fixed output-scaling powers')]:
            text=st.text_area(label,json.dumps(hp.get(field,{})),height=75,key=prefix+'::'+field)
            parsed=json.loads(text)
            if not isinstance(parsed,dict):raise ValueError('Fixed powers must be a JSON object, such as {"Pi_3": 1}.')
            if parsed:hp[field]=parsed
            else:hp.pop(field,None)
    with validate:
        hp['enable_gpr_validation']=st.checkbox('Held-out GPR validation',value=hp['enable_gpr_validation'],key=prefix+'::gpr_validation')
        a,b,c=st.columns(3)
        with a:number(hp,'gpr_max_samples','Validation sample limit',20,100000)
        with b:number(hp,'gpr_n_restarts_optimizer','Validation GPR optimizer restarts',0,30)
        with c:number(hp,'gpr_repeats','Train/test repeats',1,30)
        a,b=st.columns(2)
        with a:number(hp,'gpr_test_size','Held-out fraction',.05,.5,.05)
        with b:number(hp,'gpr_r2_threshold','Validation R² threshold',0.,1.,.01)
        hp['enable_score_uq']=st.checkbox('MI and score uncertainty',value=hp['enable_score_uq'],key=prefix+'::uq')
        number(hp,'score_uq_trials','Uncertainty subsampling trials',2,300)
    with advanced:
        st.caption('All effective settings, including kernel bounds and case-specific constraints, are included in the exported JSON.')
        if st.checkbox('Edit the complete configuration as JSON',key=prefix+'::use_json'):
            raw=st.text_area('Configuration JSON',json.dumps(hp,indent=2),height=410,key=prefix+'::json')
            parsed=json.loads(raw)
            if not isinstance(parsed,dict):raise ValueError('Configuration JSON must contain an object.')
            hp=parsed
            st.caption('The JSON configuration overrides the individual controls.')
    if hp['cma_bounds_min']>=hp['cma_bounds_max']:raise ValueError('The lower CMA-ES bound must be smaller than the upper bound.')
    if not hp['gradient_regime_options']['clustering']['candidate_cluster_counts']:raise ValueError('Select at least one candidate cluster count.')
    st.download_button('Effective configuration JSON',json.dumps(hp,indent=2),file_name='buckingham_configuration.json')
    return hp
