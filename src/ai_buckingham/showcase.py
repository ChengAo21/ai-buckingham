"""Visual case atlas and concise workflow copy shared by the interfaces."""

CASE_STORIES = {
    'mhd': dict(short='MHD duct flow', system='Pressure-driven conducting fluid', result='Pressure–viscous coordinate', target='Dimensionless mean velocity'),
    'pressure': dict(short='Pressure-gradient friction', system='Boundary layers under ZPG, APG and FPG', result='Integral-profile coordinate', target='Normalized friction velocity'),
    'rayleigh': dict(short='Rayleigh–Bénard convection', system='A fluid layer heated from below', result='Rayleigh-number dependence', target='Nusselt number'),
    'keyhole': dict(short='Laser keyhole depth', system='Laser heating and a vapor cavity', result='Keyhole-number representation', target='Normalized keyhole depth'),
    'blasius': dict(short='Blasius boundary layer', system='Laminar flow over a flat plate', result='Classical similarity coordinate', target='Normalized streamwise velocity'),
    'velocity': dict(short='Compressible velocity', system='Density and viscosity vary across the wall layer', result='Joint input and output scaling', target='Rescaled mean-velocity profiles'),
    'goldenfeld': dict(short='Goldenfeld rough-pipe scaling', system='Turbulent flow over a rough wall', result='Roughness-induced similarity', target='Rescaled friction factor'),
    'cess': dict(short='Cess eddy viscosity', system='Inner, intermediate and outer channel regions', result='Three regime-specific coordinates', target='Normalized eddy viscosity'),
    'pipe': dict(short='Rough circular pipe', system='Viscous and rough-turbulent branches', result='Two regime-specific coordinates', target='Darcy friction factor'),
}

WORKFLOW = [
    dict(id='context', title='Physical context', layer='Semantic planning', detail='Read the problem, identify variables and retrieve dimensional evidence.', output='Audited dimensions', image='grounding.png'),
    dict(id='basis', title='Admissible basis', layer='Semantic planning', detail='Construct the Buckingham–Π null space with the symbolic tool.', output='Dimensionless search space', image='basis.png'),
    dict(id='plan', title='Analysis policy', layer='Semantic planning', detail='Choose the regime route, similarity policy and numerical settings.', output='Executable workflow', image='planning.png'),
    dict(id='regimes', title='Physical regimes', layer='Numerical inference', detail='Fit GPR gradients and assess converged candidate partitions.', output='Global or local scopes', image='regimes.png'),
    dict(id='learn', title='Similarity learning', layer='Numerical inference', detail='Screen output rescaling and optimize the retained groups by mutual information.', output='Type I / II and group count', image='learning.png'),
    dict(id='validate', title='Refine and validate', layer='Numerical inference', detail='Sparsify, rationalize and evaluate conditional held-out collapse.', output='Compact representations', image='refinement.png'),
]
