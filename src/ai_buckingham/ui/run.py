from copy import deepcopy
from io import BytesIO
import json,os
import numpy as np
import pandas as pd
import streamlit as st
from ..catalog import ROOT,CASES
from ..settings import paper_settings,prompt,SHARED
from ..showcase import CASE_STORIES
from ..execution import execute
from ..definitions import validate_definition
from .theme import intro
from .configuration import editor
from .dimensions import view as dimension_view


def defaults():
    from dotenv import dotenv_values
    values={**dotenv_values(ROOT/'.env'),**os.environ}
    try:values.update(dict(st.secrets))
    except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):pass
    return values


def choose_mode(mode):st.session_state['run_mode']=mode


def results(result):
    st.subheader('Discovery result')
    content=result.get('result',{});numeric=content.get('workflow_result',content)
    local_results=numeric.get('local_workflows',[])
    for local in local_results:
        final=local.get('final_solution',{})
        st.markdown('**'+str(local.get('scope_id','Global scaling'))+'**')
        for expression in final.get('input_formulas',[]):st.code(expression,language=None)
        if final.get('output_dimensionless_formula'):st.code(final['output_dimensionless_formula'],language=None)
        a,b=st.columns(2)
        score=final.get('normalized_information_score');r2=(final.get('gpr_validation') or {}).get('test_r2_mean',final.get('gpr_test_r2'))
        a.metric('ε = exp(−MI)',f'{score:.4g}' if score is not None else '—')
        b.metric('Validation R²',f'{r2:.4f}' if r2 is not None else '—')
    if not local_results:st.write(numeric.get('status',content.get('workflow_status','See the exported result.')))
    with st.expander('Numerical results'):st.json(numeric,expanded=1)
    st.download_button('Download results JSON',json.dumps(result,indent=2,ensure_ascii=False),file_name='buckingham_discovery.json')


def render():
    intro('Run a discovery','Choose an autonomous workflow or start numerical inference from a known Buckingham–Π basis.')
    mode=st.session_state.get('run_mode','agent')
    a,b=st.columns(2,gap='large')
    with a:
        st.button('Autonomous discovery',icon='🤖',type='primary' if mode=='agent' else 'secondary',width='stretch',on_click=choose_mode,args=('agent',),key='execution_agent')
        st.caption('Natural-language problem + CSV → RAG, semantic planning, basis construction and numerical inference.')
    with b:
        st.button('Inference with a supplied Π basis',icon='🔢',type='primary' if mode=='numerical' else 'secondary',width='stretch',on_click=choose_mode,args=('numerical',),key='execution_numerical')
        st.caption('Data + variable definitions + an admissible basis → numerical inference, without LLM or embedding APIs.')
    key=st.selectbox('Example system',list(CASES),format_func=lambda k:CASE_STORIES[k]['short'],key='run_template')
    c=CASES[key]
    panels=st.tabs(['Problem','API connection','Settings'] if mode=='agent' else ['Data & basis','Settings'])
    problem=panels[0];connection=panels[1] if mode=='agent' else None;settings=panels[-1]
    data=None;data_frame=None;error=None;output=None;definition=None
    values=defaults() if mode=='agent' else {};cfg={};text=prompt(key)
    with problem:
        st.markdown('<div class="run-step">Observations</div>',unsafe_allow_html=True)
        source=st.radio('Dataset',['Example dataset','Upload CSV'],horizontal=True,key='dataset_source_'+mode)
        if source=='Upload CSV':
            upload=st.file_uploader('Observations CSV',type=['csv'],key='input_csv_'+mode)
            if upload is not None:data=upload.getvalue()
        else:data=(ROOT/'data'/c['file']).read_bytes()
        if data:
            try:
                data_frame=pd.read_csv(BytesIO(data))
                if data_frame.empty or not len(data_frame.select_dtypes(include='number').columns):raise ValueError('The CSV must contain numeric observations.')
                st.caption(f'{len(data_frame):,} observations · {len(data_frame.columns)} columns')
                st.dataframe(data_frame.head(6),width='stretch',hide_index=True)
            except (ValueError,UnicodeError,pd.errors.ParserError) as exc:error=str(exc);st.error(error)
        if mode=='agent':
            st.markdown('<div class="run-step">Physical problem</div>',unsafe_allow_html=True)
            text=st.text_area('Problem statement',value='' if source=='Upload CSV' else prompt(key),height=210,key='problem_'+key+'_'+source,
                placeholder='Describe the physical system, variable definitions and dimensions, and the dimensionless response.')
            if data_frame is not None and source=='Upload CSV' and not error:
                numeric=list(data_frame.select_dtypes(include='number').columns)
                output=st.selectbox('Response column',numeric,index=numeric.index(c['output']) if c['output'] in numeric else len(numeric)-1)
            st.caption('The autonomous workflow checks dimensions and constructs the basis before planning numerical inference.')
        else:
            use_template=st.toggle('Use the selected example’s variable definitions and basis',value=True,key='use_template_basis')
            if use_template:
                dimension_view(c)
                if data_frame is not None:
                    missing=set(c['inputs']+[c['output']])-set(data_frame.columns)
                    if missing:error='Missing example columns: '+', '.join(sorted(missing))
            elif data_frame is not None and not error:
                numeric=list(data_frame.select_dtypes(include='number').columns)
                inputs=st.multiselect('Governing-variable columns',numeric,default=[n for n in c['inputs'] if n in numeric])
                outputs=[n for n in numeric if n not in inputs]
                output=st.selectbox('Dimensionless response column',outputs,index=outputs.index(c['output']) if c['output'] in outputs else 0)
                known=inputs==c['inputs']
                D_text=st.text_area('Dimensional matrix D · JSON',json.dumps(np.asarray(c['dimensions']).T.tolist()) if known else '',height=100,
                    help='Rows follow M, L, T, I, Θ, N, J; columns follow the governing-variable order. Trailing zero rows may be omitted.')
                B_text=st.text_area('Null-space basis B · JSON',json.dumps(c['basis']) if known else '',height=100,help='Each row is one basis vector. Columns follow the governing-variable order.')
                try:
                    definition=validate_definition(dict(inputs=inputs,output=output,dimension_matrix=json.loads(D_text),basis=json.loads(B_text)),data_frame.columns)
                    dimension_view(dict(inputs=inputs,dimensions=np.asarray(definition['dimension_matrix']).T.tolist(),basis=definition['basis']))
                except (ValueError,TypeError,KeyError) as exc:error=str(exc) if D_text and B_text else 'Enter the dimensional matrix and a complete null-space basis.'
                text='Infer dimensionless dependence using the supplied governing variables and verified dimensional basis.'
            if error:st.error(error)
    if connection is not None:
        with connection:
            st.markdown('**LLM orchestration**')
            cfg['OPENAI_BASE_URL']=st.text_input('LLM API base URL',value=values.get('OPENAI_BASE_URL') or values.get('OPENAI_API_BASE','https://api.chatanywhere.tech/v1'))
            a,b=st.columns(2)
            with a:api_key=st.text_input('LLM API key',type='password',placeholder='Session key or local .env')
            with b:model=st.text_input('Model',value=values.get('ANALYZER_MODEL','gpt-5-mini'))
            cfg.update(OPENAI_API_KEY=api_key or values.get('OPENAI_API_KEY',''),ANALYZER_MODEL=model,PLANNER_MODEL=model)
            st.markdown('**Retrieval embeddings**')
            cfg['PARATERA_BASE_URL']=st.text_input('Embedding API base URL',value=values.get('PARATERA_BASE_URL','https://llmapi.paratera.com/v1'))
            a,b=st.columns(2)
            with a:embedding_key=st.text_input('Embedding API key',type='password',placeholder='Session key or local .env')
            with b:cfg['EMBEDDING_MODEL']=st.text_input('Embedding model',value=values.get('EMBEDDING_MODEL','GLM-Embedding-3'))
            cfg['PARATERA_API_KEY']=embedding_key or values.get('PARATERA_API_KEY','')
            tracing=st.checkbox('Enable LangSmith tracing',value=False)
            cfg['LANGCHAIN_TRACING_V2']=str(tracing).lower()
            if tracing:
                cfg['LANGCHAIN_ENDPOINT']=st.text_input('LangSmith endpoint',value=values.get('LANGCHAIN_ENDPOINT','https://api.smith.langchain.com'))
                cfg['LANGCHAIN_API_KEY']=st.text_input('LangSmith API key',type='password') or values.get('LANGCHAIN_API_KEY','')
            st.caption('Requested LLM temperature: 0. Session keys are excluded from exported settings and results.')
    custom_agent=source=='Upload CSV' and mode=='agent'
    custom_numerical=mode=='numerical' and not use_template
    hp=deepcopy(SHARED) if custom_agent or custom_numerical else paper_settings(key)
    if output:hp['output_var']=output
    with settings:
        st.caption('These settings govern the discovery run. Explorer’s plot and two-coefficient search controls are independent.')
        try:hp=editor(hp,mode+'_'+key+'_'+str(custom_agent or custom_numerical))
        except (ValueError,TypeError,KeyError) as exc:error='Invalid configuration: '+str(exc);st.error(error)
    ready=bool(data and text.strip() and not error and (mode=='agent' or use_template or definition))
    if st.button('Start autonomous discovery' if mode=='agent' else 'Run numerical inference',type='primary',disabled=not ready,width='stretch'):
        if mode=='agent' and not cfg.get('OPENAI_API_KEY'):st.error('Enter an LLM API key in API connection, or configure it locally.');return
        request=dict(mode=mode,case_id=None if custom_agent else key,prompt=text,settings=hp,definition=definition)
        try:
            with st.status('Executing the discovery workflow…',expanded=False) as status:
                result=execute(data,request,cfg)
                st.session_state['last_run']=result
                status.update(label='Run finished',state='complete')
        except Exception as exc:
            st.error(str(exc) if isinstance(exc,(RuntimeError,ValueError)) else f'The run could not finish ({type(exc).__name__}).');return
    if st.session_state.get('last_run'):results(st.session_state['last_run'])
