"""Config extracted from the author implementation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Dict
from typing import Literal



PROJECT_ROOT = Path(os.getenv("AI_BUCKINGHAM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
DATA_DIR = (PROJECT_ROOT / "data" / "datasets").resolve()
KNOWLEDGE_DIR = (PROJECT_ROOT / "knowledge").resolve()


from .paths import ContextOutputPath
OUTPUT_DIR = ContextOutputPath((PROJECT_ROOT / "generated").resolve())


CACHE_DIR = (PROJECT_ROOT / "generated" / "cache").resolve()


DATA_DIRECTORY = DATA_DIR


GLOBAL_RANDOM_SEED = 42


AGENT_VERSION = "1.0.0"


MI_BINNING_SAMPLE_THRESHOLD = 800 # MI estimator binning threshold


SPARSE_THRES = 0.05 # Sparsification threshold


ABS_SCORE_DROP_THRESHOLD = 0.08  # absolute decrease in exp(-MI)


SIMILARITY_SCREEN_K = 1


_SCREENING_NULL_OUTPUT_TOLERANCE = 0.05


SIMILARITY_MAX_RELATIVE_MI_CHANGE = 0.15


SIMILARITY_SCREEN_MAX_COSINE = 0.70


GPR_R2_THRESHOLD = 0.85


GPR_VALIDATION_DEFAULTS = {
    "gpr_stratify_column": None,
    "gpr_test_size": 0.20,
    "gpr_random_state": GLOBAL_RANDOM_SEED,
    "gpr_n_splits": 3,
    "gpr_n_restarts_optimizer": 3,
    "gpr_log_features": True,
    "gpr_log_target": True,
    "gpr_max_samples": 800,
    "gpr_r2_threshold": GPR_R2_THRESHOLD,
    "gpr_constant_value_bounds": (1e-3, 1e3),
    "gpr_length_scale_bounds": (1e-3, 1e3),
    "gpr_noise_level_bounds": (1e-8, 1e1),
}


PHYSICAL_KNOWLEDGE_BASE_FILEPATH = str(KNOWLEDGE_DIR / "dimensional_physical_quantities.md")


DIMENSIONLESS_KNOWLEDGE_BASE_FILEPATH = str(KNOWLEDGE_DIR / "dimensionless_relations.md")


KNOWLEDGE_INDEX_FILEPATH = str(Path(os.getenv("DIMAGENT_KNOWLEDGE_INDEX_FILEPATH", str(KNOWLEDGE_DIR / "indexes" / "physical_quantity_index.npz"))).resolve())


KNOWLEDGE_DIMENSIONLESS_INDEX_FILEPATH = str(Path(os.getenv("DIMAGENT_DIMENSIONLESS_INDEX_FILEPATH", str(KNOWLEDGE_DIR / "indexes" / "dimensionless_relation_index.npz"))).resolve())


KNOWLEDGE_EMBEDDING_MODEL = "GLM-Embedding-3"


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


WorkflowTemplateName = Literal[
    "single_regime_adaptive",
    "multi_regime_adaptive",
    "auto_mechanism_adaptive",
]


WorkflowStageId = Literal[
    "load_and_validate_data",
    "gradient_regime_detection",
    "validate_regime_partition",
    "local_dominant_group_search",
    "fan_out_local_workflows",
    "aggregate_results",
    "generate_report",
]


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


GRADIENT_REGIME_DEFAULT_OPTIONS = {'surrogate': {'mean_mode': 'constant', 'max_train_samples': None, 'n_restarts_optimizer': 3, 'optimizer_maxiter': 2000, 'optimizer_ftol': 1e-10, 'optimizer_gtol': 1e-06, 'jitter': 1e-10, 'random_state': GLOBAL_RANDOM_SEED, 'log_sigma0_sq_bounds': (-30.0, 30.0), 'log_sigma1_sq_bounds': (-30.0, 30.0), 'log_sigma_u_sq_bounds': (-30.0, 30.0), 'mean_parameter_bounds': (None, None), 'initial_noise_fraction': 1e-06, 'require_optimizer_success': False, 'regime_target_transform': 'log'}, 'clustering': {'n_clusters': None, 'candidate_cluster_counts': [2, 3, 4], 'min_cluster_fraction': 0.05, 'min_cluster_samples': 60, 'max_intercluster_cosine': 0.9, 'eigenvalue_ratio_threshold': 50.0, 'n_init': 30, 'max_iter': 200, 'random_state': GLOBAL_RANDOM_SEED, 'initialization': 'random_eigenpairs', 'max_initialization_attempts': 100, 'require_convergence': True, 'gradient_norm_tolerance': 1e-12}, 'local_workflow': {'start_count': 1, 'max_count': None, 'absolute_score_drop_threshold': ABS_SCORE_DROP_THRESHOLD, 'require_two_counts_when_possible': True, 'cma_restarts': 3, 'minimum_samples': 20, 'minimum_samples_per_mi_dimension': 20, 'minimum_samples_for_second_count': 40, 'similarity_type': 'first_kind', 'optimize_output': False, 'output_scaling_reason': 'First-kind similarity is the default/null policy unless output scaling or automatic similarity screening is explicitly requested.', 'output_scaling_confidence': 'medium', 'similarity_screening_max_relative_mi_change': SIMILARITY_MAX_RELATIVE_MI_CHANGE, 'similarity_screening_max_cosine_similarity': SIMILARITY_SCREEN_MAX_COSINE, 'sparsification_enabled': True, 'sparsify_exponent_threshold': SPARSE_THRES, 'sparsify_max_score_increase': 0.05, 'rationalization_enabled': True, 'max_rational_denominator': 5}}


BASE_DIMENSIONS = ("M", "L", "T", "I", "Theta", "N", "J")
