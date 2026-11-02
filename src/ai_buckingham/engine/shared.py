"""Imports and manuscript-aligned numerical constants. No API calls at import."""
from __future__ import annotations


import os


import re


import traceback


import json


import hashlib


import warnings


import pandas as pd


import numpy as np


import matplotlib.pyplot as plt


from IPython.display import display, HTML, Image


from datetime import datetime


from pathlib import Path


from typing import Annotated, List, Literal, Dict, Any, Optional, Sequence, Tuple


import operator


import sympy 


import random


import html


from fractions import Fraction


GLOBAL_RANDOM_SEED = 42


from scipy.special import psi, expit


from scipy.linalg import cho_factor, cho_solve


from scipy.optimize import minimize, minimize_scalar


import scipy.spatial as scispa


from cma import CMAEvolutionStrategy


from sklearn.gaussian_process import GaussianProcessRegressor


from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


from sklearn.model_selection import train_test_split


from sklearn.metrics import r2_score, mean_squared_error


from sklearn.preprocessing import StandardScaler


from sklearn.exceptions import ConvergenceWarning


from pydantic import BaseModel, Field


from langchain.messages import HumanMessage, SystemMessage


from typing_extensions import TypedDict


from langchain_openai import ChatOpenAI


from openai import OpenAI


from langchain_core.tools import tool


from langchain_core.messages import HumanMessage, SystemMessage


from langgraph.graph import StateGraph, END, START


from langgraph.graph.message import add_messages


def setup_environment(model_name: str):
    """Initialize the workspace and read API configuration from environment variables."""
    WORKSPACE = "./_Workspace_"
    os.makedirs(WORKSPACE, exist_ok=True)

    # Never hard-code credentials in source files. Set these in the shell or a .env loader.
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running the agent."
        )

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    # OPENAI_API_BASE is optional and may be set externally for a compatible endpoint.
    options = {"model": model_name, "timeout": 500, "api_key": os.environ["OPENAI_API_KEY"], "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE", "https://api.chatanywhere.tech/v1")}
    options["temperature"] = 0.0
    if model_name.startswith("gpt-5"):
        # SI Section 2 specifies temperature zero. Some LangChain versions strip
        # this parameter for GPT-5; extra_body preserves the declared request.
        # A provider that rejects it must fail explicitly, never silently change it.
        options["extra_body"] = {"temperature": 0.0}
    return ChatOpenAI(**options)


_AGENT_EDITORIAL_STYLE_LOADED = False


_AGENT_EDITORIAL_CSS = r"""
<style>
:root {
  --daa-ink: #17202a;
  --daa-muted: #667085;
  --daa-line: #d9dee5;
  --daa-soft: #f6f8fa;
  --daa-paper: #ffffff;
  --daa-blue: #245b78;
  --daa-blue-soft: #edf4f7;
  --daa-green: #3f7652;
  --daa-green-soft: #eef6f0;
  --daa-amber: #9a641f;
  --daa-amber-soft: #fbf4e8;
  --daa-red: #9a3d3d;
  --daa-red-soft: #faeeee;
}
.daa-shell {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  color: var(--daa-ink);
  max-width: 1180px;
  margin: 10px 0 18px 0;
}
.daa-masthead {
  border-top: 4px solid var(--daa-ink);
  border-bottom: 1px solid var(--daa-line);
  padding: 18px 2px 15px 2px;
  margin: 4px 0 15px 0;
}
.daa-kicker {
  color: var(--daa-blue);
  font-size: 11px;
  font-weight: 750;
  letter-spacing: .13em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.daa-title {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 30px;
  line-height: 1.15;
  font-weight: 600;
  letter-spacing: -.02em;
  margin: 0;
  color: #111820;
}
.daa-deck {
  color: #4b5563;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 15px;
  line-height: 1.5;
  max-width: 980px;
  margin-top: 8px;
}
.daa-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin-top: 13px;
  color: var(--daa-muted);
  font-size: 12px;
}
.daa-meta strong { color: var(--daa-ink); font-weight: 650; }
.daa-route {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  border: 1px solid var(--daa-line);
  background: var(--daa-paper);
  margin: 10px 0 18px 0;
}
.daa-route-step {
  min-height: 52px;
  padding: 9px 11px;
  border-right: 1px solid var(--daa-line);
  position: relative;
}
.daa-route-step:last-child { border-right: none; }
.daa-route-step.complete { background: var(--daa-green-soft); }
.daa-route-step.current { background: var(--daa-blue-soft); }
.daa-route-index {
  color: var(--daa-muted);
  font-size: 10px;
  font-weight: 750;
  letter-spacing: .08em;
}
.daa-route-label {
  color: var(--daa-ink);
  font-size: 12px;
  font-weight: 650;
  line-height: 1.25;
  margin-top: 3px;
}
.daa-card {
  border: 1px solid var(--daa-line);
  border-left: 4px solid var(--daa-blue);
  background: var(--daa-paper);
  padding: 14px 16px 13px 16px;
  margin: 10px 0 12px 0;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .035);
}
.daa-card[data-tone="success"] { border-left-color: var(--daa-green); }
.daa-card[data-tone="warning"] { border-left-color: var(--daa-amber); }
.daa-card[data-tone="danger"] { border-left-color: var(--daa-red); }
.daa-card[data-tone="muted"] { border-left-color: #98a2b3; }
.daa-card-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 8px;
}
.daa-card-label {
  color: var(--daa-blue);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.daa-card-title {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 20px;
  font-weight: 600;
  line-height: 1.22;
  margin: 0;
  color: #151b23;
}
.daa-status {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 4px 7px;
  border: 1px solid var(--daa-line);
  color: var(--daa-muted);
  white-space: nowrap;
}
.daa-status.success { color: var(--daa-green); background: var(--daa-green-soft); border-color: #c9ddcf; }
.daa-status.warning { color: var(--daa-amber); background: var(--daa-amber-soft); border-color: #ead5b5; }
.daa-status.danger { color: var(--daa-red); background: var(--daa-red-soft); border-color: #e8c7c7; }
.daa-status.info { color: var(--daa-blue); background: var(--daa-blue-soft); border-color: #cbdde5; }
.daa-summary {
  font-size: 13px;
  line-height: 1.48;
  color: #4b5563;
  max-width: 1020px;
  margin: 2px 0 10px 0;
}
.daa-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  border-top: 1px solid var(--daa-line);
  border-left: 1px solid var(--daa-line);
  margin-top: 10px;
}
.daa-metric {
  min-height: 51px;
  padding: 8px 10px;
  border-right: 1px solid var(--daa-line);
  border-bottom: 1px solid var(--daa-line);
  background: #fff;
}
.daa-metric-key {
  font-size: 9px;
  line-height: 1.2;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--daa-muted);
}
.daa-metric-value {
  font-size: 13px;
  line-height: 1.3;
  color: var(--daa-ink);
  margin-top: 4px;
  overflow-wrap: anywhere;
}
.daa-formula {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  line-height: 1.45;
  color: #243746;
  background: var(--daa-soft);
  border: 1px solid var(--daa-line);
  padding: 8px 10px;
  margin-top: 9px;
  overflow-wrap: anywhere;
}
.daa-note {
  font-family: Georgia, "Times New Roman", serif;
  color: #404854;
  font-size: 13px;
  line-height: 1.48;
  border-left: 2px solid #9fb6c3;
  padding-left: 10px;
  margin-top: 9px;
}
.daa-details { margin-top: 9px; color: #475467; font-size: 12px; }
.daa-details summary { cursor: pointer; font-weight: 700; color: var(--daa-blue); }
.daa-details table { width: 100%; border-collapse: collapse; margin-top: 7px; }
.daa-details td { padding: 5px 7px; border-bottom: 1px solid #e6e9ed; vertical-align: top; }
.daa-details td:first-child { width: 25%; font-weight: 650; color: #475467; }
.daa-result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.daa-result {
  border: 1px solid var(--daa-line);
  padding: 11px 12px;
  background: #fff;
}
.daa-result h4 {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 15px;
  margin: 0 0 7px 0;
}
.daa-result p { margin: 4px 0; font-size: 12px; color: #4b5563; }

.daa-section-title {
  margin: 14px 0 6px 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 14px;
  font-weight: 600;
  color: #1f2933;
}
.daa-diagnostic-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 7px;
  font-size: 11.5px;
}
.daa-diagnostic-table th,
.daa-diagnostic-table td {
  padding: 6px 7px;
  border-bottom: 1px solid #e6e9ed;
  text-align: left;
  vertical-align: top;
}
.daa-diagnostic-table th {
  background: var(--daa-soft);
  color: #475467;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.daa-diagnostic-table td.num,
.daa-diagnostic-table th.num { text-align: right; font-variant-numeric: tabular-nums; }
.daa-mini-badge {
  display: inline-block;
  padding: 2px 6px;
  border: 1px solid var(--daa-line);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .05em;
  text-transform: uppercase;
  white-space: nowrap;
}
.daa-mini-badge.pass { color: var(--daa-green); background: var(--daa-green-soft); border-color: #c9ddcf; }
.daa-mini-badge.fail { color: var(--daa-red); background: var(--daa-red-soft); border-color: #e8c7c7; }
.daa-mini-badge.neutral { color: var(--daa-blue); background: var(--daa-blue-soft); border-color: #cbdde5; }
.daa-diagnostic-note {
  margin-top: 7px;
  color: var(--daa-muted);
  font-size: 10.5px;
  line-height: 1.42;
}
.daa-regime-details {
  margin-top: 7px;
  border-top: 1px solid var(--daa-line);
  padding-top: 6px;
}
.daa-regime-details summary {
  cursor: pointer;
  color: var(--daa-blue);
  font-size: 11px;
  font-weight: 700;
}

@media (max-width: 720px) {
  .daa-route { grid-template-columns: 1fr 1fr; }
  .daa-card-head { display: block; }
  .daa-status { display: inline-block; margin-top: 7px; }
}
</style>
"""


AGENT_VERSION = "2610_vfinal"


MI_BINNING_SAMPLE_THRESHOLD = 800 # MI estimator binning threshold


SPARSE_THRES = 0.05 # 稀疏化阈值


ABS_SCORE_DROP_THRESHOLD = 0.10  # absolute decrease in exp(-MI)


SIMILARITY_SCREEN_K = 1


SIMILARITY_OUTPUT_EXPONENT_THRESHOLD = 0.05


SIMILARITY_MAX_RELATIVE_MI_CHANGE = 0.15


SIMILARITY_SCREEN_MAX_COSINE = 0.70


GPR_R2_THRESHOLD = 0.9


PHYSICAL_KNOWLEDGE_BASE_FILEPATH = str(Path(__file__).resolve().parents[3] / "data/dimensional_physical_quantities.md")


DIMENSIONLESS_KNOWLEDGE_BASE_FILEPATH = str(Path(__file__).resolve().parents[3] / "data/dimensionless_relations.md")


KNOWLEDGE_INDEX_FILEPATH = "./_Workspace_/dimensional_kb_embeddings3.npz"


KNOWLEDGE_DIMENSIONLESS_INDEX_FILEPATH = "./_Workspace_/dimensional_kb_dimensionless_embeddings3.npz"


KNOWLEDGE_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "GLM-Embedding-3")


PARATERA_BASE_URL = os.getenv("PARATERA_BASE_URL", "https://llmapi.paratera.com/v1")


PARATERA_API_KEY_ENV = "PARATERA_API_KEY"  # Set this environment variable


KNOWLEDGE_TOP_K_PER_VARIABLE = 3


KNOWLEDGE_MAX_EVIDENCE_ROWS = 30


KNOWLEDGE_LEXICAL_WEIGHT = 0.65


KNOWLEDGE_SEMANTIC_WEIGHT = 0.35


KNOWLEDGE_MIN_HYBRID_SCORE = 0.22


KNOWLEDGE_ALLOW_LEXICAL_FALLBACK = False


KNOWLEDGE_INTERPRETATION_TOP_K = 5


SAVE_PLOT = True


CANONICAL_DIMENSION_ORDER = ("M", "L", "T", "I", "Theta", "N", "J")


PHYSICAL_QUANTITY_REQUIRED_COLUMNS = (
    "entry_id",
    "canonical_quantity",
    "symbols",
    "aliases",
    "definition",
    "si_dimension",
    "dimension_vector",
    "si_unit",
    "domain",
    "ambiguity_notes",
    "source",
    "source_identifier",
    "confidence",
)


DIMENSIONLESS_RELATION_REQUIRED_COLUMNS = (
    "entry_id",
    "canonical_name",
    "symbols",
    "aliases",
    "defining_expression",
    "physical_interpretation",
    "domain",
    "source",
    "source_identifier",
    "confidence",
)


_GREEK_NORMALIZATION = {
    "μ": "mu", "ν": "nu", "ρ": "rho", "η": "eta", "σ": "sigma",
    "κ": "kappa", "λ": "lambda", "θ": "theta", "Θ": "theta",
    "ε": "epsilon", "α": "alpha", "β": "beta", "γ": "gamma",
}


_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "with", "variable", "quantity",
    "physical", "input", "parameter",
}


WORKFLOW_TEMPLATE_STAGES: Dict[str, List[str]] = {
    "single_regime_adaptive": [
        "load_and_validate_data",
        "local_dominant_group_search",
        "aggregate_results",
        "generate_report",
    ],
    "multi_regime_adaptive": [
        "load_and_validate_data",
        "gradient_regime_detection",
        "validate_regime_partition",
        "fan_out_local_workflows",
        "aggregate_results",
        "generate_report",
    ],
    "auto_mechanism_adaptive": [
        "load_and_validate_data",
        "gradient_regime_detection",
        "validate_regime_partition",
        "local_dominant_group_search",
        "fan_out_local_workflows",
        "aggregate_results",
        "generate_report",
    ],
}


GRADIENT_REGIME_IMPLEMENTATION = (
    "GPR gradient surrogate + eigenvalue-weighted gradient clustering "
    "+ regime-wise mutual-information optimization"
)


GRADIENT_REGIME_DEFAULT_OPTIONS = {
    "surrogate": {
        # The clustering surrogate is fitted in x = log(Pi) coordinates.
        # "constant" is suitable for smooth responses.  The affine-sigmoid
        # transition mean can be selected for a sharp transition between regimes.
        "mean_mode": "constant",
        "max_train_samples": None,
        "n_restarts_optimizer": 5,
        "optimizer_maxiter": 500,
        "optimizer_ftol": 1e-10,
        "optimizer_gtol": 1e-6,
        "jitter": 1e-10,
        "random_state": GLOBAL_RANDOM_SEED,
        "log_sigma0_sq_bounds": (-30.0, 30.0),
        "log_sigma1_sq_bounds": (-30.0, 30.0),
        "log_sigma_u_sq_bounds": (-30.0, 30.0),
        "mean_parameter_bounds": (None, None),
        "initial_noise_fraction": 1e-6,
        # A finite optimum may still be useful when L-BFGS-B stops because of
        # an iteration limit.  The status is always reported; set this to True
        # when non-converged surrogate fits must be rejected.
        "require_optimizer_success": False,
        "regime_target_transform": "log",  # 'auto', 'log', 'log1p', or 'raw'
    },
    "clustering": {
        "n_clusters": None,
        "candidate_cluster_counts": [2, 3, 4],
        # The objective compares random starts within a fixed K only.
        # Cross-K selection always retains the coarsest admissible partition.
        "min_cluster_fraction": 0.05,
        "min_cluster_samples": 60,
        "max_intercluster_cosine": 0.90,  # Auto requires K=2 directional support when any partition is admissible.
        "eigenvalue_ratio_threshold": 50.0,
        "n_init": 30,
        "max_iter": 200,
        "random_state": GLOBAL_RANDOM_SEED,
        "initialization": "random_eigenpairs",
        "max_initialization_attempts": 100,
        "require_convergence": True,
        # Gradients below this norm do not contain reliable direction
        # information and are excluded from clustering/local MI optimization.
        "gradient_norm_tolerance": 1e-12,
        # Global E is diagnostic only; no spectral threshold determines the route.
    },
    "local_workflow": {
        # Each detected regime runs the same adaptive single-mechanism workflow:
        # test k=1,2,...; lock the parsimonious count; sparsify; re-optimize; and
        # rationalize.  The final count is never hard-coded by the cluster step.
        "start_count": 1,
        "max_count": None,
        "absolute_score_drop_threshold": ABS_SCORE_DROP_THRESHOLD,
        "require_two_counts_when_possible": True,
        "cma_restarts": 3,
        "minimum_samples": 20,
        "minimum_samples_per_mi_dimension": 20,
        "minimum_samples_for_second_count": 40,
        "similarity_type": "first_kind",
        "optimize_output": False,
        "output_scaling_reason": (
            "First-kind similarity is the default/null policy unless output scaling "
            "or automatic similarity screening is explicitly requested."
        ),
        "output_scaling_confidence": "medium",
        "similarity_screening_output_exponent_threshold": (
            SIMILARITY_OUTPUT_EXPONENT_THRESHOLD
        ),
        "similarity_screening_max_relative_mi_change": (
            SIMILARITY_MAX_RELATIVE_MI_CHANGE
        ),
        "similarity_screening_max_cosine_similarity": (
            SIMILARITY_SCREEN_MAX_COSINE
        ),
        "sparsification_enabled": True,
        "sparsify_exponent_threshold": SPARSE_THRES,
        "sparsify_max_score_increase": 0.05,
        "rationalization_enabled": True,
        "max_rational_denominator": 5,
    },
}


_REGIME_PLOT_COLORS = [
    "#6c9c7e", "#cc7a7a", "#d2a24c", "#6f86c9",
    "#9b7ac9", "#4f9da6", "#b07d62", "#87946e",
]

__all__ = [name for name in globals() if not name.startswith("__")]
