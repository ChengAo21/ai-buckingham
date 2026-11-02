"""Knowledge-base browsing and explicit, network-free retrieval for the explorer."""
import numpy as np
from .catalog import ROOT,CASES
from .settings import prompt

def knowledge_records(kind):
    from .engine.knowledge import load_dimensional_knowledge_base,load_dimensionless_relation_knowledge_base
    loader=load_dimensional_knowledge_base if kind=='quantities' else load_dimensionless_relation_knowledge_base
    file='dimensional_physical_quantities.md' if kind=='quantities' else 'dimensionless_relations.md'
    return loader(str(ROOT/'data'/file)).to_dict('records')

def local_case_retrieval(key):
    from .engine.knowledge import DimensionalKnowledgeRetriever,DimensionlessRelationRetriever
    c=CASES[key]
    quantities=DimensionalKnowledgeRetriever(enable_embeddings=False,allow_lexical_fallback=True)
    relations=DimensionlessRelationRetriever(enable_embeddings=False,allow_lexical_fallback=True)
    context=prompt(key)
    dimensions={name:quantities.retrieve(variable_name=name,problem_context=context,
                analyzer_context=f'{name}: dimension vector {dim+[0,0]}',top_k=3) for name,dim in zip(c['inputs'],c['dimensions'])}
    powers=np.asarray(c['basis']).T@np.asarray(c['coefficients'])
    expression=' * '.join(f'{name}^({power:g})' for name,power in zip(c['inputs'],powers) if abs(power)>1e-12)
    matches=relations.retrieve(context+'\nDiscovered physical monomial: '+expression,top_k=5)
    return dict(case=key,mode='local_lexical',quantities=dict(matches_by_variable=dimensions),relations=dict(matches=matches))
