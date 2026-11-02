from html import escape
import json
import numpy as np
import streamlit as st
from ..catalog import ROOT,CASES
from ..retrieval import knowledge_records,local_case_retrieval

@st.cache_data(show_spinner=False)
def records(kind):return knowledge_records(kind)
@st.cache_data(show_spinner=False)
def evidence(key):return local_case_retrieval(key)

def source_link(record):
    source=str(record.get('source',''))
    identifier=str(record.get('source_identifier',''))
    url=identifier if identifier.startswith(('https://','http://')) else 'https://doi.org/'+identifier[4:] if identifier.startswith('doi:') else None
    st.caption(source)
    if url:st.markdown(f'[Source reference]({url})')

def record_card(r,quantity=False,expected=None):
    with st.container(border=True):
        st.markdown('**'+str(r.get('canonical_quantity') if quantity else r.get('canonical_name'))+'**')
        if quantity:
            st.markdown(str(r.get('definition','')))
            st.code(str(r.get('si_dimension',''))+'   ·   '+str(r.get('si_unit','')),language=None)
            if expected is not None:
                vector=json.loads(r['dimension_vector']) if isinstance(r['dimension_vector'],str) else r['dimension_vector']
                match=np.allclose(vector,expected)
                st.caption('Dimensions agree with the case definition.' if match else 'Alternative quantity meaning; compare the physical context.')
            if r.get('ambiguity_notes'):st.caption(r['ambiguity_notes'])
        else:
            st.code(r.get('defining_expression',''),language=None)
            st.markdown(str(r.get('physical_interpretation','')))
        score=r.get('hybrid_score',r.get('retrieval_score'))
        if score is not None:st.caption(f"Retrieval score {score:.3f} · {r.get('retrieval_mode','local lookup').replace('_',' ')}")
        source_link(r)

def browser():
    a,b=st.tabs(['Physical quantities','Dimensionless relations'])
    for pane,kind in [(a,'quantities'),(b,'relations')]:
        with pane:
            items=records(kind);quantity=kind=='quantities';name='canonical_quantity' if quantity else 'canonical_name'
            query=st.text_input('Search names, symbols or domains',key='kb_search_'+kind)
            filtered=[r for r in items if not query or query.lower() in ' '.join(str(v) for v in r.values()).lower()]
            st.caption(f'{len(items)} curated entries · {len(filtered)} shown')
            if not filtered:st.info('No matching entries.');continue
            index=st.selectbox('Knowledge-base entry',range(len(filtered)),format_func=lambda i:filtered[i][name],key='kb_entry_'+kind)
            record_card(filtered[index],quantity)
            file='dimensional_physical_quantities.md' if quantity else 'dimensionless_relations.md'
            st.download_button('Knowledge base · Markdown',(ROOT/'data'/file).read_bytes(),file_name=file,key='kb_download_'+kind)

def case_view(key):
    run=st.session_state.get('last_run',{})
    live=run.get('result',{}) if run.get('mode')=='API_agent' else {}
    use_live=run.get('case')==key
    if live.get('knowledge_retrieval') and run.get('case')!=key:
        source=st.segmented_control('Evidence source',['Displayed case','Latest autonomous run'],default='Displayed case')
        use_live=source=='Latest autonomous run'
    data=live if use_live else {}
    if data.get('knowledge_retrieval'):
        quantities=data['knowledge_retrieval'];relations=data.get('interpretation_knowledge_retrieval',{})
        st.caption('Retrieval evidence from the latest autonomous run.')
    else:
        snapshot=evidence(key);quantities=snapshot['quantities'];relations=snapshot['relations']
        st.caption('Local lexical retrieval from the supplied knowledge bases. Autonomous runs also record semantic retrieval evidence.')
    a,b=st.tabs(['Dimensional grounding','Scaling interpretation'])
    with a:
        matches=quantities.get('matches_by_variable',{})
        variables=list(matches) or CASES[key]['inputs']
        variable=st.selectbox('Input variable',variables,key='rag_variable_'+key)
        expected=CASES[key]['dimensions'][CASES[key]['inputs'].index(variable)]+[0,0] if variable in CASES[key]['inputs'] and (not data or run.get('case')==key) else None
        rows=matches.get(variable,[])
        if not rows:st.info('No sufficiently relevant quantity record was retrieved for this variable.')
        for r in rows:record_card(r,True,expected)
    with b:
        st.caption('Retrieved relations support interpretation; they do not alter the numerical groups or scores.')
        for r in relations.get('matches',[]):record_card(r)
        if not relations.get('matches'):st.info('No relation evidence is available for this run.')
