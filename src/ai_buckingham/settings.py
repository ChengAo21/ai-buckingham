"""Manuscript and SI take precedence over historical source defaults."""
from copy import deepcopy
from .catalog import ROOT, CASES
import json

SHARED = {
    "workflow_template": "auto_mechanism_adaptive", "similarity_type": "auto",
    "random_state": 42, "cma_restarts": 3, "regime_cma_restarts": 3,
    "cma_init_position": .1, "cma_init_stepsize": .5,
    "cma_bounds_min": -1., "cma_bounds_max": 1., "cma_popsize": 300, "cma_maxiter": 30000,
    "abs_score_drop_threshold": .10, "max_rational_denominator": 5, "max_exponent_error": .05, "max_mi_drop_percent": .10,
    "max_cosine_similarity": .70, "regime_sparsify_exponent_threshold": .05, "regime_sparsify_max_score_increase": .05,
    "regime_min_samples": 20, "regime_min_samples_per_mi_dimension": 20, "workflow_minimum_samples_for_second_count": 40,
    "enable_score_uq": True, "score_uq_trials": 20,
    "gpr_max_samples": 800, "gpr_n_restarts_optimizer": 3, "gpr_repeats": 3, "gpr_test_size": .20, "gpr_r2_threshold": .90,
    "gpr_constant_value_bounds": [1e-3,1e3], "gpr_length_scale_bounds": [1e-3,1e3], "gpr_noise_level_bounds": [1e-8,10],
    "show_workflow_progress": False, "enable_workflow_visualization": False, "enable_regime_partition_visualization": False,
    "similarity_screening_output_exponent_threshold": .05, "similarity_screening_max_relative_mi_change": .15, "similarity_screening_max_cosine_similarity": .70,
    "gradient_regime_options": {
        "surrogate": {"mean_mode": "constant", "max_train_samples": None, "n_restarts_optimizer": 5, "optimizer_maxiter": 500, "regime_target_transform": "log"},
        "clustering": {"candidate_cluster_counts": [2,3,4], "min_cluster_samples": 60, "min_cluster_fraction": .05, "eigenvalue_ratio_threshold": 50., "max_intercluster_cosine": .90, "n_init": 30, "max_iter": 200, "require_convergence": True},
    },
}

def paper_settings(case_id):
    c = CASES[case_id]; hp = deepcopy(SHARED)
    hp["output_var"] = c["output"]
    hp["_benchmark_case_id"] = case_id
    if c["mi_method"] != "auto": hp.update(mi_method_used=c["mi_method"], mi_param_used=c["mi_param"])
    if case_id == "pressure": hp["similarity_type"] = "first_kind"
    if case_id == "rayleigh": hp.update(cma_init_position=.3,max_rational_denominator=4)
    if case_id == "blasius": hp.update(cma_init_position=.6,cma_init_stepsize=.3,max_rational_denominator=8)
    if case_id == "velocity": hp.update(cma_init_stepsize=.2, max_cosine_similarity=.8, penalty_opt_w_fixed=False, use_collinearity_penalty=True, fixed_in_power={"Pi_3":1.}, fixed_out_power={"Pi_3":0.})
    if case_id == "goldenfeld": hp.update(cma_init_stepsize=.1,cma_bounds_min=-.1,cma_bounds_max=.5,normalize_input="norm2",use_collinearity_penalty=True,fixed_out_power={"Pi_2":0.})
    if case_id in ("cess", "pipe"):
        hp["gradient_regime_options"]["surrogate"].update(max_train_samples=800 if case_id=="cess" else 1100)
    if case_id == "pipe":
        hp["gradient_regime_options"]["surrogate"].update(mean_mode="affine_sigmoid_transition",log_sigma0_sq_bounds=[-40,40],log_sigma1_sq_bounds=[-30,30],log_sigma_u_sq_bounds=[-30,5])
        # SI Table 5 overrides the histogram default only for local pipe regimes.
        hp.update(regime_histogram_bins=20)
    return hp

def prompt(case_id):
    original = json.loads((ROOT/"configs/source_case_overrides.json").read_text(encoding="utf-8"))
    names = {"mhd":"mhd_generator", "pressure":"presGrad_", "keyhole":"laser_material", "rayleigh":"rayleigh_benard", "blasius":"blasius_", "velocity":"velotrans_compTurb", "goldenfeld":"goldenfeld_", "cess":"CMAME24_CESS_PROMPT", "pipe":"CMAME24_PIPE_PROMPT"}
    text = original[names[case_id]]
    # Correct source prompt's dependent-column alias to the supplied CSV.
    if case_id == "blasius": text = text.replace("denoted as u*", "denoted as u/U")
    return text
