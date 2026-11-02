"""Validation for numerical inference with a user-supplied dimensional basis."""
import numpy as np

def validate_definition(definition,columns):
    inputs=list(definition['inputs']);output=str(definition['output'])
    if not inputs or len(set(inputs))!=len(inputs) or output in inputs:raise ValueError('Choose distinct input variables and a separate response.')
    missing=set(inputs+[output])-set(columns)
    if missing:raise ValueError('Missing CSV columns: '+', '.join(sorted(missing)))
    D=np.asarray(definition['dimension_matrix'],float);B=np.asarray(definition['basis'],float)
    if D.ndim!=2 or B.ndim!=2 or not 1<=D.shape[0]<=7 or D.shape[1]!=len(inputs) or B.shape[1]!=len(inputs):raise ValueError('Matrix columns must follow the selected input-variable order, with up to seven canonical dimension rows.')
    if not np.isfinite(D).all() or not np.isfinite(B).all():raise ValueError('Dimensional matrices and basis coefficients must be finite.')
    if len(B)==0 or np.linalg.matrix_rank(B)!=len(inputs)-np.linalg.matrix_rank(D) or np.linalg.matrix_rank(B)!=len(B):raise ValueError('Provide a complete, linearly independent null-space basis.')
    if not np.allclose(D@B.T,0,atol=1e-10,rtol=0):raise ValueError('The supplied basis is not dimensionally admissible: D Bᵀ must be zero.')
    return dict(inputs=inputs,output=output,dimension_matrix=D.tolist(),basis=B.tolist())
