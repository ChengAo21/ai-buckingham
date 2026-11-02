"""Reported coordinates, formulas and independent axis defaults."""
import numpy as np
from .catalog import CASES,coordinates
from .scopes import REGIMES,scope_data

FORMULAS={
 'mhd':r'\Pi_1=\left(\frac{h^3\rho}{\mu^2}\frac{dp}{dx}\right)^{1/3}',
 'pressure':r'\Pi_1=\left(\frac{\mathcal{M}}{U_\infty\nu}\right)^{2/3}\left(\frac{U_\infty\delta}{\nu}\right)^{-2/3}\left(\frac{U_\infty\delta^*}{\nu}\right)^{-1}\left(\frac{U_\infty\theta}{\nu}\right)^{3/5}',
 'rayleigh':r'\Pi_1=Ra^{1/3}=\left(\frac{g\alpha\Delta T\,h^3}{\nu\kappa}\right)^{1/3}',
 'keyhole':r'\Pi_1=\left(\frac{r_0\eta P}{\rho\alpha^3}\right)^{2/3}\left(\frac{(T_l-T_0)r_0^2c_p}{\alpha^2}\right)^{-2/3}\left(\frac{r_0V_s}{\alpha}\right)^{-1/3}',
 'blasius':r'\Pi_1=y\sqrt{\frac{U}{\nu x}}',
 'velocity':r'\Pi_1=y^+\left(\frac{\rho}{\rho_w}\right)^{2/3}\left(\frac{\mu}{\mu_w}\right)^{-1/3}',
 'goldenfeld':r'\Pi_1=Re^{3/4}\frac{r}{D}',
}
OUTPUT_FORMULAS={
 'velocity':r'\widetilde\Pi_y=u^+\left(\frac{\rho}{\rho_w}\right)^{0.86}\left(\frac{\mu}{\mu_w}\right)^{0.40}',
 'goldenfeld':r'\widetilde\Pi_y=c_f Re^{1/4}',
}
# (log x, log y), matching the reported response plots.
AXES={
 'mhd':((True,True),(True,True)),
 'pressure':((True,False),(False,False)),
 'rayleigh':((True,True),(True,True)),
 'keyhole':((True,True),(False,True)),
 'blasius':((False,False),(False,False)),
 'velocity':((True,False),(True,False)),
 'goldenfeld':((True,False),(False,False)),
 'cess':((True,True),(True,True)),
 'pipe':((True,False),(True,False)),
}

def axis_defaults(case_id,scope='global'):
    left,right=AXES[case_id]
    if case_id=='pipe' and scope=='B':right=(True,True)
    return left,right

def scaling_data(case_id,scope='global',partition=None):
    c=CASES[case_id]
    if case_id in REGIMES:
        d=scope_data(case_id,scope,partition);df=d['frame'];lp=d['logpi'];rows=d['rows']
        x=np.exp(np.clip(lp@d['coefficients'],-700,700));y=d['response']
        formula=REGIMES[case_id][scope]['formula']
    else:
        df,lp,x,y,rows=coordinates(case_id);formula=FORMULAS[case_id]
    raw_index={'velocity':2,'pressure':1,'blasius':1}.get(case_id,0)
    raw_x=np.exp(lp[:,raw_index]);raw_label=c['basis_labels'][raw_index]
    return dict(frame=df,rows=rows,raw_x=raw_x,raw_y=df[c['output']].to_numpy(float),raw_label=raw_label,
                x=x,y=y,formula=formula,output_formula=OUTPUT_FORMULAS.get(case_id))
