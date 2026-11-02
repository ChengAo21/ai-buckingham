from fractions import Fraction
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .theme import plot_title

SYMBOLS={'mu':r'\mu','nu':r'\nu','rho':r'\rho','eta':r'\eta','dp_dx':r'\frac{dp}{dx}',
 'U_inf':r'U_\infty','u_tau':r'u_\tau','rho_w':r'\rho_w','mu_w':r'\mu_w','delta':r'\delta','delta*':r'\delta^*',
 'theta':r'\theta','lambda':r'\lambda','kappa':r'\kappa','alpha':r'\alpha','alpha_T':r'\alpha_T',
 'Tl-T0':r'(T_l-T_0)','etaP':r'(\eta P)','cp':r'c_p','r0':r'r_0','Vs':r'V_s','epsilon':r'\epsilon','MKE':r'\mathcal M'}

def symbol(name):return SYMBOLS.get(name,r'\mathrm{'+str(name).replace('_',r'\_')+'}')
def exponent(value):
    f=Fraction(float(value)).limit_denominator(48)
    return str(f.numerator) if f.denominator==1 else r'\frac{'+str(f.numerator)+'}{'+str(f.denominator)+'}'
def monomial(names,powers):
    parts=[]
    for name,power in zip(names,powers):
        if abs(power)<1e-12:continue
        parts.append(symbol(name) if abs(power-1)<1e-12 else '{'+symbol(name)+'}^{'+exponent(power)+'}')
    return r'\,'.join(parts) or '1'

def view(c):
    D=np.asarray(c['dimensions'],float).T;B=np.asarray(c['basis'],float)
    active=np.flatnonzero(np.any(D!=0,axis=1));labels=['M','L','T','I','Θ','N','J'][:len(D)]
    left,right=st.columns([1.1,1],gap='large')
    with left:
        plot_title('Dimensional structure · D')
        fig=go.Figure(go.Heatmap(z=D[active],x=c['inputs'],y=[labels[i] for i in active],
            colorscale=[[0,'#b9d6e7'],[.5,'#f8faf5'],[1,'#e8bd9d']],zmid=0,text=[[f'{v:g}' for v in row] for row in D[active]],texttemplate='%{text}',textfont=dict(size=14),
            hovertemplate='%{x}<br>Dimension %{y}: %{z:g}<extra></extra>',showscale=False,xgap=3,ygap=3))
        fig.update_layout(height=max(180,45*len(active)+70),margin=dict(l=28,r=8,t=8,b=40),paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Arial',color='#36555b',size=13),yaxis=dict(autorange='reversed',showgrid=False),xaxis=dict(showgrid=False,tickangle=0))
        st.plotly_chart(fig,width='stretch')
    with right:
        plot_title('Admissible Buckingham–Π basis')
        for i,row in enumerate(B):st.latex(r'\Pi^{('+str(i+1)+')}='+monomial(c['inputs'],row))
    st.caption(f'D Bᵀ = 0 · {len(c["inputs"])} governing variables → {len(B)} independent dimensionless groups')
