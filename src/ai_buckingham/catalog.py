"""Case definitions transcribed from manuscript and SI, with explicit provenance."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Dimension vectors use (M, L, T, I, Theta). Outputs are already dimensionless.
DIMS = {
    "length": [0,1,0,0,0], "velocity": [0,1,-1,0,0],
    "nu": [0,2,-1,0,0], "mu": [1,-1,-1,0,0], "rho": [1,-3,0,0,0],
    "eta": [1,3,-3,-2,0], "dp_dx": [1,-2,-2,0,0], "B": [1,0,-2,-1,0],
    "MKE": [0,3,-2,0,0], "temperature": [0,0,0,0,1],
    "lambda": [1,1,-3,0,-1], "g": [0,1,-2,0,0], "alpha_T": [0,0,0,0,-1],
    "power": [1,2,-3,0,0], "cp": [0,2,-2,0,-1],
}

def case(id, title, zh, file, inputs, dimensions, output, basis, basis_labels, coeff, formula, category, figure, si_table, mi, **extra):
    return dict(id=id, title=title, zh=zh, file=file, inputs=inputs.split(), dimensions=[DIMS[d] for d in dimensions.split()], output=output,
                basis=basis, basis_labels=basis_labels, coefficients=coeff, formula=formula, category=category, figure=figure,
                si_table=si_table, mi_method=mi[0], mi_param=mi[1], **extra)

CASES = {
    "mhd": case("mhd", "MHD duct flow", "磁流体管道流动", "dataset_mhd.csv", "h mu rho eta dp_dx B", "length mu rho eta dp_dx B", "u*",
        [[3,-2,1,0,1,0],[1,-.5,0,-.5,0,1]], ["h³ρ(dp/dx)/μ²", "hB/√(μη)"], [1/3,0], "Π₁ = [h³ρ(dp/dx)/μ²]^(1/3)", "Type I", 3, 9, ("kraskov",4),
        question="Can the pressure–viscous dependence be isolated while magnetic variables are removed?", exponent_note="An invertible cube-root coordinate; it does not specify a velocity power law.", screening=[3.555557,3.640408,.023308], expected_k=1),
    "pressure": case("pressure", "Pressure-gradient friction", "压力梯度壁面摩擦", "dataset_presGrad.csv", "U_inf nu MKE delta delta* theta", "velocity nu MKE length length length", "U_tau/U_inf",
        [[-1,-1,1,0,0,0],[1,-1,0,1,0,0],[1,-1,0,0,1,0],[1,-1,0,0,0,1]], ["MKE/(U∞ν)","U∞δ/ν","U∞δ*/ν","U∞θ/ν"], [2/3,-2/3,-1,3/5], "Π₁ = Π⁽¹⁾^(2/3) Π⁽²⁾^(-2/3) Π⁽³⁾^(-1) Π⁽⁴⁾^(3/5)", "Type I", 4, 11, ("kraskov",3),
        question="Can ZPG, APG and FPG observations collapse with one integral-profile coordinate?", expected_k=1),
    "rayleigh": case("rayleigh", "Rayleigh–Bénard convection", "瑞利–贝纳德对流", "dataset_rayben.csv", "h lambda delta_T g alpha nu kappa", "length lambda temperature g alpha_T nu nu", "Nu",
        [[0,0,1,0,1,0,0],[-1.5,0,0,-.5,0,1,0],[-1.5,0,0,-.5,0,0,1]], ["ΔTα","ν/(h^(3/2)g^(1/2))","κ/(h^(3/2)g^(1/2))"], [1/3,-1/3,-1/3], "Π₁ = Ra^(1/3)", "Type I", 4, 13, ("kraskov",6),
        question="Does the learning recover the Rayleigh-number dependence and remove thermal conductivity?", exponent_note="Ra^(1/3) is a coordinate; it does not imply Nu ∝ Ra^(1/3).", screening=[2.600264,2.694659,.035030], expected_k=1),
    "keyhole": case("keyhole", "Laser keyhole depth", "激光熔化匙孔深度", "dataset_keyhole.csv", "rho Tl-T0 r0 alpha etaP cp Vs", "rho temperature length nu power cp velocity", "e*",
        [[-1,0,1,-3,1,0,0],[0,1,2,-2,0,1,0],[0,0,1,-1,0,0,1]], ["r₀ηP/(ρα³)","(Tl−T₀)r₀²cp/α²","r₀Vs/α"], [2/3,-2/3,-1/3], "Π₁ = Π⁽¹⁾^(2/3) Π⁽²⁾^(-2/3) Π⁽³⁾^(-1/3)", "Type I", 4, 15, ("kraskov",4),
        question="Can material and processing quantities be reduced to a Keyhole-number representation?", screening=[1.633773,1.881104,.131481], expected_k=1),
    "blasius": case("blasius", "Blasius boundary layer", "布拉修斯边界层", "dataset_blasius.csv", "U nu x y", "velocity nu length length", "u/U",
        [[1,-1,1,0],[1,-1,0,1]], ["Ux/ν","Uy/ν"], [-.5,1], "Π₁ = y√[U/(νx)]", "Type I", 4, 17, ("kraskov",5),
        question="Can the classical wall-normal similarity coordinate be recovered?", screening=[2.763663,2.851984,.030969], expected_k=1),
    "velocity": case("velocity", "Compressible velocity scaling", "可压缩湍流速度变换", "dataset_veloTrans.csv", "y rho_w mu_w rho mu u_tau", "length rho mu rho mu velocity", "u+",
        [[0,-1,0,1,0,0],[0,0,-1,0,1,0],[1,1,-1,0,0,1]], ["ρ/ρw","μ/μw","y⁺"], [2/3,-1/3,1], "Π₁ = y⁺(ρ/ρw)^(2/3)(μ/μw)^(-1/3)", "Type II", 5, 19, ("binning",50),
        output_coefficients=[.86,.40,0], output_formula="Ỹ = u⁺(ρ/ρw)^0.86(μ/μw)^0.40", question="Which density and viscosity powers collapse compressible mean-velocity profiles?", screening=[2.734221,3.232046,.154028], expected_k=1),
    "goldenfeld": case("goldenfeld", "Goldenfeld rough-pipe similarity", "Goldenfeld 粗糙管流标度", "dataset_goldenfeld.csv", "rho mu D U r", "rho mu length velocity length", "cf",
        [[1,-1,1,1,0],[0,0,-1,0,1]], ["Re = ρDU/μ","r/D"], [.75,1], "Π₁ = Re^(3/4)(r/D)", "Type II", 5, 21, ("kraskov",5),
        output_coefficients=[.25,0], output_formula="Ỹ = cf Re^(1/4)", question="Can the roughness-induced criticality transformation be discovered?", screening=[1.178833,1.994507,.408960], expected_k=1),
    "cess": case("cess", "Cess eddy viscosity", "Cess 涡黏性模型", "dataset_cess_channel.csv", "h y u_tau nu", "length length velocity nu", "nu_t/nu",
        [[-1,1,0,0],[-1,0,-1,1]], ["η = y/h","1/Reτ = ν/(huτ)"], [1,-1], "Outer: Reτ · Intermediate: Reτ η^0.57 · Near wall: y⁺", "Multi-regime", 6, 23, ("kraskov",5),
        question="Can gradient directions distinguish outer, intermediate and near-wall dependence?", expected_k=3),
    "pipe": case("pipe", "Rough circular pipe", "粗糙圆管流动", "dataset_pipe.csv", "rho mu D U epsilon", "rho mu length velocity length", "f_D",
        [[1,-1,1,1,0],[0,0,-1,0,1]], ["Re = ρDU/μ","ε/D"], [-1,0], "Laminar: Re^(-1) · Rough turbulent: ε/D", "Multi-regime", 6, 25, ("auto",None),
        question="Can a data-supported partition separate viscous and roughness-dominated flow?", expected_k=2),
}

def get_case(case_id):
    return CASES[case_id]

def frame(case_id):
    return pd.read_csv(DATA / CASES[case_id]["file"])

def prepare_data(case_id):
    """Match the supplied executor's finite and near-zero filter; keep row provenance."""
    c = CASES[case_id]; df = frame(case_id)
    values = df[c["inputs"]+[c["output"]]].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1) & (np.abs(values) >= 1e-15).all(axis=1)
    # Power products and log-Pi gradients require strictly positive inputs.
    if np.any(values[mask,:-1] <= 0): raise ValueError("These benchmark inputs require positive values.")
    selected = df.loc[mask]
    return selected, selected[c["inputs"]].to_numpy(float), selected[[c["output"]]].to_numpy(float), np.flatnonzero(mask)

def coordinates(case_id, coefficients=None, output_coefficients=None):
    c = CASES[case_id]; df, X, Y, rows = prepare_data(case_id)
    logpi = np.log(X) @ np.asarray(c["basis"]).T
    coeff = c["coefficients"] if coefficients is None else coefficients
    out = c.get("output_coefficients", np.zeros(logpi.shape[1])) if output_coefficients is None else output_coefficients
    x = np.exp(np.clip(logpi @ np.asarray(coeff), -700,700))
    y = Y[:,0]*np.exp(np.clip(logpi @ np.asarray(out), -700,700))
    return df, logpi, x, y, rows

def reference():
    return json.loads((ROOT / "results/reference/paper_results.json").read_text(encoding="utf-8"))
