"""Optimization extracted from the author implementation."""
from __future__ import annotations
from . import recording as _recording

import warnings
import numpy as np
from typing import List, Optional
import random
from fractions import Fraction
from scipy.special import psi
from scipy.optimize import minimize_scalar
import scipy.spatial as scispa
from cma import CMAEvolutionStrategy
from . import config as _config
from . import plotting as _plotting
from . import presentation as _presentation
from . import validation as _validation


def global_calc_MI(x, y, mi_method, mi_param):
    if mi_method == "binning":
        # Binning estimator
        def entropy(data, num_bins):
            if data.ndim == 1: data = data.reshape(-1, 1)
            hist, _ = np.histogramdd(data, bins=[num_bins]*data.shape[1])
            p = hist / np.sum(hist)
            return -np.sum(p[p > 0] * np.log(p[p > 0]))
        # MI = H(x) + H(y) - H(x, y)
        return entropy(x, mi_param) + entropy(y, mi_param) - entropy(np.hstack([x, y]), mi_param)
    elif mi_method == "kraskov":
            # Kraskov (K-NN) 
            N = x.shape[0]
            V = np.hstack([x, y])
            kdtree = scispa.KDTree(V)
            ei, _ = kdtree.query(V, mi_param + 1, p=np.inf)
            dM = np.maximum(ei[:, -1], 1e-10) 
            nx = scispa.KDTree(x).query_ball_point(x, dM, p=np.inf, return_length=True)
            ny = scispa.KDTree(y).query_ball_point(y, dM, p=np.inf, return_length=True)
            return psi(mi_param) - (psi(nx) + psi(ny)).mean() + psi(N)


def global_calc_pi(c, basis_matrices):
    c = np.asarray(c).reshape(1, -1)
    coef_pi = np.dot(c, basis_matrices).flatten()
    return coef_pi


def _power_product_sample_mask(X, Y):
    """Keep finite rows except those with a zero input and zero response."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float).reshape(-1)
    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    joint_zero = np.any(X == 0.0, axis=1) & (y == 0.0)
    return finite & ~joint_zero


def global_coef_to_pi_mat(coef, X_data):
    """Evaluate a raw power product without changing input values."""
    return np.prod(X_data ** coef, axis=1).reshape(-1, 1)


def global_format_formula(coefs, vars_list):
    """Format a physical-variable power product with compact unit exponents."""
    coefficients = np.asarray(coefs, dtype=float)
    numerator: List[str] = []
    denominator: List[str] = []

    def term(variable: str, magnitude: float) -> str:
        if np.isclose(magnitude, 1.0, atol=1e-8):
            return str(variable)
        value = f"{float(magnitude):.5f}".rstrip("0").rstrip(".")
        return f"{variable}^{{{value}}}"

    for variable, raw_value in zip(vars_list, coefficients):
        value = float(raw_value)
        if abs(value) < 1e-5:
            continue
        if value > 0:
            numerator.append(term(variable, value))
        else:
            denominator.append(term(variable, abs(value)))

    numerator_text = " * ".join(numerator) if numerator else "1"
    if not denominator:
        return numerator_text
    denominator_text = " * ".join(denominator)
    return f"({numerator_text}) / ({denominator_text})"


def _format_optimized_power(value: float, digits: int = 5) -> str:
    """Compact decimal representation used for optimized basis coefficients."""
    value = float(value)
    if abs(value) < 10 ** (-(digits + 1)):
        return "0"
    if np.isclose(value, round(value), atol=10 ** (-(digits + 1))):
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _basis_generator_formulas(basis_matrices, input_vars) -> List[str]:
    """Return physical formulas for the supplied Buckingham-Pi basis rows."""
    basis = np.asarray(basis_matrices, dtype=float)
    return [global_format_formula(row, input_vars) for row in basis]


def _basis_combination_formula(
    coefficients,
    generator_formulas: List[str],
    *,
    symbolic: bool,
) -> str:
    """Format a learned Pi group as powers of the original basis generators."""
    terms = []
    for index, coefficient in enumerate(np.asarray(coefficients, dtype=float)):
        if abs(coefficient) < 1e-8:
            continue
        exponent = _format_optimized_power(coefficient)
        base = f"Pi_{index + 1}" if symbolic else f"[{generator_formulas[index]}]"
        terms.append(f"{base}**({exponent})")
    return " * ".join(terms) if terms else "1"


def _candidate_formula_lines(
    candidate: dict,
    output_label: str = "Y",
) -> List[str]:
    """Return paired physical-space and basis-coordinate formulas."""
    physical = list(candidate.get("input_formulas") or [])
    symbolic = list(
        candidate.get("input_basis_coordinate_formulas") or []
    )
    expanded = list(
        candidate.get("input_basis_expanded_formulas") or []
    )
    lines: List[str] = []

    for index, physical_formula in enumerate(physical):
        lines.append(
            f"Physical PI_{index + 1}: {physical_formula}"
        )
        symbolic_text = (
            symbolic[index] if index < len(symbolic) else "—"
        )
        expanded_text = (
            expanded[index] if index < len(expanded) else symbolic_text
        )
        lines.append(
            f"Optimizer basis coordinates PI_{index + 1}: "
            f"{symbolic_text} = {expanded_text}"
        )

    if candidate.get("output_optimization"):
        output_physical = (
            candidate.get("output_scaling_formula") or "1"
        )
        output_symbolic = (
            candidate.get("output_basis_coordinate_formula") or "1"
        )
        output_expanded = (
            candidate.get("output_basis_expanded_formula")
            or output_symbolic
        )
        lines.append(
            f"Scaled output: {output_label}* = "
            f"{output_label} * [{output_physical}]"
        )
        lines.append(
            "Output basis coordinates: "
            f"{output_symbolic} = {output_expanded}"
        )
    return lines


def _subspace_stability(exponent_matrices: list[np.ndarray]) -> dict:
    """Assess restart stability of a one-dimensional direction or a k-D subspace.

    Individual Pi groups are not unique when k>1; therefore the comparison is
    performed on the row spaces of the exponent matrices rather than by matching
    formulas one by one.
    """
    if len(exponent_matrices) < 2:
        return {"status": "single_restart", "mean_principal_cosine": None, "minimum_principal_cosine": None}
    pair_scores = []
    for i in range(len(exponent_matrices)):
        Ei = np.asarray(exponent_matrices[i], dtype=float)
        Qi, _ = np.linalg.qr(Ei.T)
        for j in range(i + 1, len(exponent_matrices)):
            Ej = np.asarray(exponent_matrices[j], dtype=float)
            Qj, _ = np.linalg.qr(Ej.T)
            singular_values = np.linalg.svd(Qi.T @ Qj, compute_uv=False)
            pair_scores.append(float(np.min(np.clip(singular_values, 0.0, 1.0))))
    return {
        "status": "success",
        "pairwise_min_principal_cosines": pair_scores,
        "mean_principal_cosine": float(np.mean(pair_scores)),
        "minimum_principal_cosine": float(np.min(pair_scores)),
    }


def _restricted_basis_after_dropping_variables(basis_matrices, drop_indices, tol=1e-10):
    """Restrict the original null-space basis while forcing selected variables to exponent zero.

    If e = c B is a dimensionless exponent vector, dropping variables J imposes
    c B[:, J] = 0.  We compute a basis for this constrained coefficient space and
    map it back to physical-exponent space.  Dimensional homogeneity is therefore
    preserved exactly up to floating-point tolerance.
    """
    B = np.asarray(basis_matrices, dtype=float)
    drop_indices = sorted(set(int(i) for i in drop_indices))
    if not drop_indices:
        return B.copy()
    constraint = B[:, drop_indices].T
    _, singular_values, vh = np.linalg.svd(constraint, full_matrices=True)
    if singular_values.size:
        rank_tol = max(constraint.shape) * np.max(singular_values) * np.finfo(float).eps
        rank = int(np.sum(singular_values > max(rank_tol, tol)))
    else:
        rank = 0
    coefficient_null_basis = vh[rank:, :]
    if coefficient_null_basis.size == 0:
        return np.empty((0, B.shape[1]))
    restricted = coefficient_null_basis @ B
    restricted[np.abs(restricted) < tol] = 0.0
    independent_rows = []
    current_rank = 0
    for row in restricted:
        trial = np.vstack(independent_rows + [row]) if independent_rows else row.reshape(1, -1)
        new_rank = np.linalg.matrix_rank(trial, tol=tol)
        if new_rank > current_rank:
            independent_rows.append(row)
            current_rank = new_rank
    return np.asarray(independent_rows, dtype=float)


def _evaluate_local_pi_solution(
    in_vecs,
    out_vec,
    basis_matrices,
    X,
    Y,
    input_vars,
    mi_method,
    mi_param,
    optimize_output,
    hp,
):
    """Evaluate one local candidate without changing its basis coefficients."""
    B = np.asarray(basis_matrices, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)

    display_mode = hp.get("display_ipt_Pi", "max")
    if display_mode not in ["max", "norm2", "none"]:
        raise ValueError(
            f"display_ipt_Pi must be 'max', 'norm2', or 'none', not {display_mode}.")

    output_label = str(hp.get("_resolved_output_var", "Y"))

    raw_input_exponents = [global_calc_pi(v, B) for v in in_vecs]
    displayed_input_exponents = []
    displayed_input_basis_coefficients = []
    input_display_scales = []
    negative_base_columns = np.any(X < 0, axis=0)
    for basis_coefficients, physical_exponents in zip(in_vecs, raw_input_exponents):
        if np.any(physical_exponents[negative_base_columns] != 0):
            # Taking a display root can turn a real signed power into a complex one.
            scale = 1.0
        elif display_mode == "max":
            scale = float(np.max(np.abs(physical_exponents)))
        elif display_mode == "norm2":
            scale = float(np.linalg.norm(physical_exponents))
        else:
            scale = 1.0
        if scale <= 1e-14:
            scale = 1.0
        displayed_input_exponents.append(np.asarray(physical_exponents, dtype=float) / scale)
        displayed_input_basis_coefficients.append(
            np.asarray(basis_coefficients, dtype=float) / scale
        )
        input_display_scales.append(scale)

    raw_output_exponents = global_calc_pi(out_vec, B) if optimize_output else None

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        pi_objective = np.column_stack([
            global_coef_to_pi_mat(exp_vec, X).ravel()
            for exp_vec in raw_input_exponents
        ])
        y_objective = Y.copy()
        if optimize_output:
            output_scale_objective = global_coef_to_pi_mat(raw_output_exponents, X)
            y_objective = y_objective * output_scale_objective

    mi_value = max(
        float(global_calc_MI(pi_objective, y_objective, mi_method, mi_param)),
        0.0,
    )
    information_score = _validation.normalized_information_score(mi_value)

    input_pi_eval = np.column_stack([
        global_coef_to_pi_mat(exp_vec, X).ravel()
        for exp_vec in displayed_input_exponents
    ])

    gpr_validation = {"status": "disabled"}
    if hp.get("enable_gpr_validation", True):
        gpr_options = hp.get("_resolved_gpr_validation_options")
        if gpr_options is None:
            gpr_options = _validation.resolve_gpr_validation_options(hp)
        first_fit = []
        surface_fits = hp.get("_gpr_surface_fits")
        surface_options = {}
        if surface_fits is not None and input_pi_eval.shape[1] == 2:
            surface_options["on_first_fit"] = first_fit.append
        gpr_validation = _validation.evaluate_gpr_collapse(
            input_pi_eval,
            y_objective,
            max_samples=gpr_options["gpr_max_samples"],
            repeats=gpr_options["gpr_n_splits"],
            n_restarts_optimizer=gpr_options["gpr_n_restarts_optimizer"],
            log_features=gpr_options["gpr_log_features"],
            log_target=gpr_options["gpr_log_target"],
            constant_value_bounds=gpr_options["gpr_constant_value_bounds"],
            length_scale_bounds=gpr_options["gpr_length_scale_bounds"],
            noise_level_bounds=gpr_options["gpr_noise_level_bounds"],
            r2_threshold=gpr_options["gpr_r2_threshold"],
            test_size=gpr_options["gpr_test_size"],
            random_state=gpr_options["gpr_random_state"],
            **surface_options,
            fixed_split=hp.get("_gpr_fixed_split"),
        )
        if first_fit:
            # Keep the fit local. Solution copies retain the same validation record.
            surface_fits[id(gpr_validation)] = (gpr_validation, first_fit[0])
        gpr_validation["validation_random_state"] = gpr_options["gpr_random_state"]
        gpr_validation["resolved_options"] = dict(gpr_options)

    score_uq = {"status": "disabled"}
    if hp.get("enable_score_uq", True):
        score_uq = _validation.estimate_mi_score_subsampling_uq(
            pi_objective,
            y_objective,
            mi_method,
            mi_param,
            num_trials=int(hp.get("score_uq_trials", 20)),
        )

    preprocessed_input_terms = hp.get("_preprocessed_input_formula_terms", {})
    formula_terms = [preprocessed_input_terms.get(name, name) for name in input_vars]
    generator_formulas = _basis_generator_formulas(B, formula_terms)
    physical_formulas = [
        global_format_formula(vector, formula_terms)
        for vector in displayed_input_exponents
    ]
    input_basis_symbolic_raw = [
        _basis_combination_formula(
            vector, generator_formulas, symbolic=True
        )
        for vector in in_vecs
    ]
    input_basis_expanded_raw = [
        _basis_combination_formula(
            vector, generator_formulas, symbolic=False
        )
        for vector in in_vecs
    ]
    input_basis_symbolic_display = [
        _basis_combination_formula(
            vector, generator_formulas, symbolic=True
        )
        for vector in displayed_input_basis_coefficients
    ]
    input_basis_expanded_display = [
        _basis_combination_formula(
            vector, generator_formulas, symbolic=False
        )
        for vector in displayed_input_basis_coefficients
    ]

    output_scaling_formula = None
    output_basis_symbolic = None
    output_basis_expanded = None
    if optimize_output:
        output_scaling_formula = global_format_formula(
            raw_output_exponents, formula_terms
        )
        output_basis_symbolic = _basis_combination_formula(
            out_vec, generator_formulas, symbolic=True
        )
        output_basis_expanded = _basis_combination_formula(
            out_vec, generator_formulas, symbolic=False
        )

    similarity_type = "second_kind" if optimize_output else "first_kind"

    return {
        "mutual_information": mi_value,
        "normalized_information_score": information_score,
        "mi_score_subsampling_uq": score_uq,
        "output_optimization": bool(optimize_output),
        "similarity_type": similarity_type,
        "output_label": output_label,

        "basis_generator_formulas": generator_formulas,
        "input_basis_coefficients": [
            np.asarray(v, dtype=float).tolist() for v in in_vecs
        ],
        "input_basis_coefficients_display": [
            np.asarray(v, dtype=float).tolist()
            for v in displayed_input_basis_coefficients
        ],
        "input_display_scales": input_display_scales,
        "input_basis_coordinate_formulas": input_basis_symbolic_raw,
        "input_basis_expanded_formulas": input_basis_expanded_raw,
        "input_basis_coordinate_formulas_display":
            input_basis_symbolic_display,
        "input_basis_expanded_formulas_display":
            input_basis_expanded_display,

        "output_basis_coefficients": (
            np.asarray(out_vec, dtype=float).tolist()
            if optimize_output else None
        ),
        "output_basis_coordinate_formula": output_basis_symbolic,
        "output_basis_expanded_formula": output_basis_expanded,

        "input_physical_exponents": [
            np.asarray(v, dtype=float).tolist()
            for v in displayed_input_exponents
        ],
        "input_physical_exponents_raw": [
            np.asarray(v, dtype=float).tolist()
            for v in raw_input_exponents
        ],
        "output_physical_exponents": (
            np.asarray(raw_output_exponents, dtype=float).tolist()
            if optimize_output else None
        ),
        "input_formulas": physical_formulas,
        "output_scaling_formula": output_scaling_formula,
        "output_dimensionless_formula": (
            f"{output_label} * [{output_scaling_formula}]"
            if optimize_output else output_label
        ),

        "gpr_validation": gpr_validation,
        "gpr_test_r2": gpr_validation.get("test_r2_mean"),
        "collapse_pass": gpr_validation.get("collapse_pass"),
    }


def _cma_hyperparameter(hp, name, default):
    """Read a CMA setting and preserve precedence for archived regime_ aliases."""
    for key in (f"regime_{name}", name):
        if hp.get(key) is not None:
            return hp[key]
    return default


def _run_local_cma_once(
    X,
    Y,
    basis_matrices,
    input_vars,
    pi_num_input,
    optimize_output,
    fixed_in_power,
    fixed_out_power,
    mi_method,
    mi_param,
    hp,
    seed,
):
    """Run one CMA-ES restart for one regime and one candidate input count."""
    B = np.asarray(basis_matrices, dtype=float)
    num_basis = B.shape[0]
    fixed_in_idx = {int(k.split("_")[1]) - 1: float(v) for k, v in (fixed_in_power or {}).items()}
    fixed_out_idx = {int(k.split("_")[1]) - 1: float(v) for k, v in (fixed_out_power or {}).items()}
    if any(idx >= num_basis for idx in fixed_in_idx) or any(idx >= num_basis for idx in fixed_out_idx):
        return {"status": "invalid_fixed_constraint_for_local_basis"}
    free_in_idx = [i for i in range(num_basis) if i not in fixed_in_idx]
    free_out_idx = [i for i in range(num_basis) if i not in fixed_out_idx]
    n_input_parameters = len(free_in_idx) * int(pi_num_input)
    n_output_parameters = len(free_out_idx) if optimize_output else 0
    n_parameters = n_input_parameters + n_output_parameters

    normalize_mode = hp.get("normalize_input", "none")
    if normalize_mode not in ("none", "norm2"):
        return {"status": "invalid_normalization_mode", 
                "message": f"normalize_input must be 'none' or 'norm2', not '{normalize_mode}'."}

    use_collinearity_penalty = bool(
        hp.get("use_collinearity_penalty", True if pi_num_input > 1 else False)
    )
    max_cosine = float(hp.get("max_cosine_similarity", 0.7))
    # For input-output collinearity, False compares only coordinates that are
    # free on both sides; True also includes fixed basis coordinates.
    penalty_opt_w_fixed = bool(hp.get("penalty_opt_w_fixed", True))

    def reconstruct(parameters):
        parameters = np.asarray(parameters, dtype=float)
        cursor = 0
        in_vecs = []
        for _ in range(int(pi_num_input)):
            vec = np.zeros(num_basis)
            for j in range(num_basis):
                if j in fixed_in_idx:
                    vec[j] = fixed_in_idx[j]
                else:
                    vec[j] = parameters[cursor]
                    cursor += 1
            if not fixed_in_idx and normalize_mode == "norm2":
                vec = vec / max(np.linalg.norm(vec), 1e-12)
            in_vecs.append(vec)
        out_vec = np.zeros(num_basis)
        if optimize_output:
            for j in range(num_basis):
                if j in fixed_out_idx:
                    out_vec[j] = fixed_out_idx[j]
                else:
                    out_vec[j] = parameters[cursor]
                    cursor += 1
        return in_vecs, out_vec

    def get_cosine_similarity(vec1, vec2):
        vec1 = np.asarray(vec1, dtype=float)
        vec2 = np.asarray(vec2, dtype=float)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 1.0
        return abs(np.dot(vec1, vec2) / (norm1 * norm2))

    def check_colinearity_penalty(parameters):
        """Return the accumulated amount by which cosine limits are exceeded."""
        penalty = 0.0
        num_free_in = len(free_in_idx)
        if num_free_in == 0:
            return penalty

        free_in_matrix = np.asarray(
            parameters[:n_input_parameters], dtype=float
        ).reshape(int(pi_num_input), num_free_in)

        if int(pi_num_input) > 1:
            for i in range(int(pi_num_input)):
                for j in range(i + 1, int(pi_num_input)):
                    cosine = get_cosine_similarity(
                        free_in_matrix[i], free_in_matrix[j]
                    )
                    if cosine > max_cosine:
                        penalty += abs(cosine - max_cosine)

        if optimize_output:
            in_vecs_raw, out_vec_raw = reconstruct(parameters)
            if hp.get("_screening_allow_null_output", False):
                output_physical = global_calc_pi(out_vec_raw, B)
                negligible_threshold = float(
                    hp.get(
                        "_screening_null_output_tolerance",
                        _config._SCREENING_NULL_OUTPUT_TOLERANCE,
                    )
                )
                if (
                    output_physical.size == 0
                    or np.max(np.abs(output_physical)) <= negligible_threshold
                ):
                    # A near-zero output scaling is precisely the Type-I null
                    # candidate. It has no meaningful direction and therefore
                    # must not be rejected as "collinear" with the input group.
                    return penalty
            mutual_free_indices = (
                sorted(set(free_in_idx) & set(free_out_idx))
                if free_in_idx != free_out_idx
                else list(free_in_idx)
            )
            indices_to_check = (
                list(range(num_basis))
                if penalty_opt_w_fixed
                else mutual_free_indices
            )
            if indices_to_check:
                for in_vec in in_vecs_raw:
                    cosine = get_cosine_similarity(
                        np.asarray(in_vec, dtype=float)[indices_to_check],
                        np.asarray(out_vec_raw, dtype=float)[indices_to_check],
                    )
                    if cosine > max_cosine:
                        penalty += abs(cosine - max_cosine)

        return penalty

    def objective(parameters):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                if use_collinearity_penalty:
                    penalty_val = check_colinearity_penalty(parameters)
                    if penalty_val > 0:
                        return 1e4 + penalty_val * 1e5

                in_vecs, out_vec = reconstruct(parameters)
                pi_data = np.column_stack([
                    global_coef_to_pi_mat(
                        global_calc_pi(vec, B), X
                    ).ravel()
                    for vec in in_vecs
                ])

                y_data = np.asarray(Y, dtype=float).reshape(-1, 1).copy()

                if optimize_output:
                    out_exp = global_calc_pi(out_vec, B)
                    out_scale = global_coef_to_pi_mat(
                        out_exp, X
                    )
                    y_data = y_data * out_scale
                return -float(global_calc_MI(pi_data, y_data, mi_method, mi_param))
            except Exception:
                return 1e8 + random.uniform(1e6, 1e7)

    if n_parameters == 0:
        # No free coefficients remain: directly evaluate the unique candidate.
        optimizer_name = "direct_evaluation"
        best_parameters = np.empty(0, dtype=float)
        best_objective = objective(best_parameters)

    elif n_parameters == 1:
        # pycma is not robust for a one-dimensional search space.
        # This case naturally occurs after sparsification reduces the
        # admissible Buckingham-Pi space to one free coefficient.
        optimizer_name = "bounded_scalar"
        scalar_result = minimize_scalar(
            lambda value: objective(np.asarray([value], dtype=float)),
            bounds = (float(hp.get("regime_cma_bounds_min", hp.get("cma_bounds_min", -2))),
                      float(hp.get("regime_cma_bounds_max", hp.get("cma_bounds_max", 2)))
                      ),
            method = "bounded",
            options = {
                "xatol": 1e-8,
                "maxiter": int(_cma_hyperparameter(hp, "cma_maxiter", 30000)),
            })
        if not scalar_result.success or not np.isfinite(scalar_result.fun):
            _recording.record_optimizer_end(optimizer_name, np.array([float(scalar_result.x)]), scalar_result.fun, seed=seed, pi_count=pi_num_input, optimize_output=optimize_output, status="failed")
            return {
                "status": "optimization_failed",
                "optimizer": optimizer_name,
                "optimizer_dimension": int(n_parameters),
            }
        best_parameters = np.array([float(scalar_result.x)],dtype = float,)
        best_objective = float(scalar_result.fun)      

    else:
        optimizer_name = "CMA-ES"
        options = {
            "bounds": [
                [float(hp.get("regime_cma_bounds_min", hp.get("cma_bounds_min", -1)))] * n_parameters,
                [float(hp.get("regime_cma_bounds_max", hp.get("cma_bounds_max", 1)))] * n_parameters,
            ],
            "maxiter": int(_cma_hyperparameter(hp, "cma_maxiter", 30000)),
            "popsize": int(_cma_hyperparameter(hp, "cma_popsize", 300)),
            "seed": int(seed),
            "tolx": 1e-8,
            "tolfun": 1e-8,
            "verb_disp": 0,
            "verbose": -9,
        }

        es = CMAEvolutionStrategy(
            [float(hp.get("cma_init_position", 0.1))] * n_parameters,
            float(hp.get("cma_init_stepsize", 0.5)),
            options,
        )
        while not es.stop():
            solutions = es.ask()
            function_values = [objective(candidate) for candidate in solutions]
            es.tell(solutions, function_values)
            _recording.record_cma_generation(es, solutions, function_values, seed=seed, pi_count=pi_num_input, optimize_output=optimize_output, reconstruct=reconstruct, basis=B, input_vars=input_vars)

        best_parameters = es.result.xbest
        best_objective = es.result.fbest

    _recording.record_optimizer_end(optimizer_name, best_parameters, best_objective, seed=seed, pi_count=pi_num_input, optimize_output=optimize_output)
    if not np.isfinite(best_objective) or best_objective >= 1e4:
        return {"status": "optimization_failed"}
    in_vecs, out_vec = reconstruct(best_parameters)
    evaluated = _evaluate_local_pi_solution(
        in_vecs,
        out_vec,
        B,
        X,
        Y,
        input_vars,
        mi_method,
        mi_param,
        optimize_output,
        hp,
    )
    evaluated.update({"status": "success", "seed": int(seed), "pi_count": int(pi_num_input),
                      "optimizer": optimizer_name, "optimizer_dimension": int(n_parameters),})
    return evaluated


def _rationalize_local_solution(solution, basis_matrices, X, Y, input_vars, mi_method, mi_param, optimize_output, hp):
    """Rationalize one regime's basis coefficients and accept only if local MI is preserved."""
    max_denominator = int(hp.get("max_rational_denominator", 4))
    max_coefficient_error = float(hp.get("max_exponent_error", 0.05))
    fallback_decimals = int(hp.get("fallback_decimals", 2))
    max_relative_mi_drop = float(hp.get("max_mi_drop_percent", 0.1))

    def rationalize_vector(vector):
        result = []
        for value in vector:
            value = float(value)
            if abs(value) < 1e-8:
                result.append(0.0)
                continue
            fraction = Fraction(value).limit_denominator(max_denominator)
            if abs(value - float(fraction)) <= max_coefficient_error:
                result.append(float(fraction))
            else:
                result.append(round(value, fallback_decimals))
        return result

    rational_input = [rationalize_vector(v) for v in solution["input_basis_coefficients"]]
    rational_output = rationalize_vector(solution["output_basis_coefficients"]) if optimize_output else np.zeros(np.asarray(basis_matrices).shape[0]).tolist()
    try:
        rational_evaluation = _evaluate_local_pi_solution(
            rational_input,
            rational_output,
            basis_matrices,
            X,
            Y,
            input_vars,
            mi_method,
            mi_param,
            optimize_output,
            hp,
        )
    except FloatingPointError:
        return {"accepted": False, "status": "invalid_rational_power_product", "final": solution}

    restart_metadata_keys = [
            "selection_mode",
            "primary_seed",
            "restart_seeds",
            "restart_count",
            "restart_mutual_information_values",
            "restart_mutual_information_mean",
            "restart_mutual_information_std",
            "restart_information_score_values",
            "restart_information_score_mean",
            "restart_information_score_std",
            "restart_stability",]
    for key in restart_metadata_keys:
        if key in solution:
            rational_evaluation[key] = solution[key]
    
    rational_evaluation["restart_statistics_scope"] = ("inherited_from_pre_rationalization_selected_solution")
    rational_evaluation["subsampling_statistics_scope"] = ("recomputed_for_rationalized_solution")
    
    original_mi = float(solution["mutual_information"])
    rational_mi = float(rational_evaluation["mutual_information"])
    relative_drop = max(original_mi - rational_mi, 0.0) / max(abs(original_mi), 1e-12)
    accepted = bool(relative_drop <= max_relative_mi_drop and hp.get("rationalized_on", True))
    return {
        "accepted": accepted,
        "relative_mi_drop": relative_drop,
        "max_relative_mi_drop": max_relative_mi_drop,
        "candidate": rational_evaluation,
        "final": rational_evaluation if accepted else solution,
    }


def _run_local_restarts(
    X,
    Y,
    basis_matrices,
    input_vars,
    pi_count,
    optimize_output,
    fixed_in_power,
    fixed_out_power,
    mi_method,
    mi_param,
    hp,
    restart_count,
    primary_seed = _config.GLOBAL_RANDOM_SEED,
):
    """Run repeated CMA-ES fits and retain restart and subsampling uncertainty."""
    
    successful = []
    primary_candidate = None
    rng = np.random.default_rng(primary_seed)
    for restart_id in range(max(1, int(restart_count))):
        if restart_id == 0:
            restart_seed = primary_seed
        else:
            restart_seed = rng.integers(_config.GLOBAL_RANDOM_SEED, 2**10 - 1)
        candidate = _run_local_cma_once(
            X=X,
            Y=Y,
            basis_matrices=basis_matrices,
            input_vars=input_vars,
            pi_num_input=int(pi_count),
            optimize_output=bool(optimize_output),
            fixed_in_power=fixed_in_power,
            fixed_out_power=fixed_out_power,
            mi_method=mi_method,
            mi_param=int(mi_param),
            hp=hp,
            seed=int(restart_seed),
        )

        if restart_id == 0:
            primary_candidate = candidate

        if candidate.get("status") == "success":
            successful.append(candidate)

    if not successful:
        return {
            "status": "all_restarts_failed",
            "pi_count": int(pi_count),
            "restart_count": int(restart_count),
        }

    if primary_candidate is None or primary_candidate.get("status") != "success":
        return {
            "status": "primary_restart_failed",
            "primary_seed": int(primary_seed),
            "restart_count_requested": int(restart_count),
            "successful_diagnostic_seeds": [int(item["seed"]) for item in successful],
        }

    selected = dict(primary_candidate)
    mi_values = np.asarray(
        [float(item["mutual_information"]) for item in successful],
        dtype=float,
    )
    score_values = np.asarray(
        [float(item["normalized_information_score"]) for item in successful],
        dtype=float,
    )
    mi_restart_std = float(np.std(mi_values, ddof=1)) if mi_values.size > 1 else 0.0
    score_restart_std = (
        float(np.std(score_values, ddof=1)) if score_values.size > 1 else 0.0
    )
    selected.update({
        "selection_mode": "primary_seed_restart",
        "primary_seed": int(primary_seed),
        "restart_seeds": [int(item["seed"]) for item in successful],
        "restart_count": int(len(successful)),
        "restart_mutual_information_values": mi_values.tolist(),
        "restart_mutual_information_mean": float(np.mean(mi_values)),
        "restart_mutual_information_std": mi_restart_std,
        "restart_information_score_values": score_values.tolist(),
        "restart_information_score_mean": float(np.mean(score_values)),
        "restart_information_score_std": score_restart_std,
        "restart_statistics_scope": ("current_candidate_cma_restarts"),
        "subsampling_statistics_scope": ("current_candidate_subsampling"),
        "restart_stability": _subspace_stability([
            np.asarray(item["input_physical_exponents"], dtype=float)
            for item in successful
        ]),
    })
    return selected


def _similarity_screen_candidate_summary(candidate: dict) -> dict:
    """Keep only compact diagnostics from a k=1 similarity-screen candidate."""
    if not isinstance(candidate, dict):
        return {}
    return {
        "status": candidate.get("status"),
        "mutual_information": candidate.get("mutual_information"),
        "normalized_information_score": candidate.get("normalized_information_score"),
        "input_formulas": candidate.get("input_formulas", []),

        "input_basis_coordinate_formulas":
            candidate.get("input_basis_coordinate_formulas", []),
        "input_basis_expanded_formulas":
            candidate.get("input_basis_expanded_formulas", []),
        "output_basis_coordinate_formula":
            candidate.get("output_basis_coordinate_formula"),
        "output_basis_expanded_formula":
            candidate.get("output_basis_expanded_formula"),        

        "output_scaling_formula": candidate.get("output_scaling_formula"),
        "output_physical_exponents": candidate.get("output_physical_exponents"),
        "output_basis_coefficients": candidate.get("output_basis_coefficients"),
        "restart_count": candidate.get("restart_count"),
        "restart_mutual_information_mean": candidate.get("restart_mutual_information_mean"),
        "restart_mutual_information_std": candidate.get("restart_mutual_information_std"),
        "selection_mode": candidate.get("selection_mode"),
    }


def screen_similarity_type(X, Y, input_vars, basis_matrices, mi_method, mi_param, manual_hyperparams=None, local_workflow_options=None, scope_id='global') -> dict:
    """Compare fixed-output and joint-output k=1 fits on the same samples. Type I is selected only by the configured relative MI loss; otherwise select Type II. Final fixed basis powers are restored after screening. The internal null-output tolerance belongs only to the preserved optimizer cosine rule, not to class selection."""
    hp = dict(manual_hyperparams or {})
    options = _validation._deep_update_dict(_config.GRADIENT_REGIME_DEFAULT_OPTIONS['local_workflow'], local_workflow_options or {})
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    B = np.asarray(basis_matrices, dtype=float)
    if X.ndim != 2 or X.shape[0] != Y.shape[0]:
        return {'status': 'invalid_data_shape', 'scope_id': str(scope_id), 'message': 'Similarity screening requires X and Y with the same sample count.'}
    if B.ndim != 2 or B.shape[0] == 0:
        return {'status': 'invalid_basis', 'scope_id': str(scope_id), 'message': 'Similarity screening requires at least one Buckingham-Pi basis vector.'}
    finite_mask = _power_product_sample_mask(X, Y)
    X_local = X[finite_mask]
    Y_local = Y[finite_mask]
    if X_local.shape[0] == 0:
        return {'status': 'no_valid_samples', 'scope_id': str(scope_id), 'message': 'No finite samples for similarity screening.'}
    planned_method = options.get('mi_method')
    if hp.get('mi_method_used') is not None:
        local_method = str(hp['mi_method_used'])
    elif hp.get('regime_mi_method') is not None:
        local_method = str(hp['regime_mi_method'])
    elif planned_method in (None, 'auto'):
        local_method = str(mi_method)
    else:
        local_method = str(planned_method)
    planned_param = options.get('mi_param')
    if hp.get('mi_param_used') is not None:
        local_param = int(hp['mi_param_used'])
    elif hp.get('regime_mi_param') is not None:
        local_param = int(hp['regime_mi_param'])
    elif planned_param is not None:
        local_param = int(planned_param)
    else:
        local_param = int(mi_param)
    if local_method not in ('kraskov', 'binning'):
        return {'status': 'invalid_mi_method', 'scope_id': str(scope_id), 'mi_method': local_method}
    if local_method == 'binning' and X_local.shape[0] < _config.MI_BINNING_SAMPLE_THRESHOLD:
        local_method = 'kraskov'
        local_param = min(5, max(2, X_local.shape[0] - 2))
    planned_restarts = options.get('cma_restarts', options.get('restarts', 3))
    restart_count = max(1, int(_cma_hyperparameter(hp, 'cma_restarts', planned_restarts)))
    null_output_tolerance = _config._SCREENING_NULL_OUTPUT_TOLERANCE
    mi_change_threshold = max(0.0, float(options.get('similarity_screening_max_relative_mi_change', _config.SIMILARITY_MAX_RELATIVE_MI_CHANGE)))
    max_cosine = min(1.0, max(0.0, float(options.get('similarity_screening_max_cosine_similarity', _config.SIMILARITY_SCREEN_MAX_COSINE))))
    screening_hp = dict(hp)
    screening_hp.update({'show_workflow_progress': False, 'enable_gpr_validation': False, 'enable_score_uq': False, 'use_collinearity_penalty': True, 'max_cosine_similarity': max_cosine, 'penalty_opt_w_fixed': False, '_screening_allow_null_output': True, '_screening_null_output_tolerance': null_output_tolerance})
    screening_hp.pop('fixed_in_power', None)
    screening_hp.pop('fixed_out_power', None)
    fixed_candidate = _run_local_restarts(X=X_local, Y=Y_local, basis_matrices=B, input_vars=input_vars, pi_count=_config.SIMILARITY_SCREEN_K, optimize_output=False, fixed_in_power=None, fixed_out_power=None, mi_method=local_method, mi_param=local_param, hp=screening_hp, restart_count=restart_count, primary_seed=_config.GLOBAL_RANDOM_SEED)
    joint_candidate = _run_local_restarts(X=X_local, Y=Y_local, basis_matrices=B, input_vars=input_vars, pi_count=_config.SIMILARITY_SCREEN_K, optimize_output=True, fixed_in_power=None, fixed_out_power=None, mi_method=local_method, mi_param=local_param, hp=screening_hp, restart_count=restart_count, primary_seed=_config.GLOBAL_RANDOM_SEED)
    if fixed_candidate.get('status') != 'success' or joint_candidate.get('status') != 'success':
        return {'status': 'optimization_failed', 'scope_id': str(scope_id), 'screening_k': _config.SIMILARITY_SCREEN_K, 'fixed_output_candidate': _similarity_screen_candidate_summary(fixed_candidate), 'joint_output_candidate': _similarity_screen_candidate_summary(joint_candidate), 'message': 'Similarity screening requires successful fixed-output and joint-output k=1 optimizations.'}
    fixed_mi = float(fixed_candidate['mutual_information'])
    joint_mi = float(joint_candidate['mutual_information'])
    relative_mi_loss_without_output = max(joint_mi - fixed_mi, 0.0) / max(abs(joint_mi), 1e-12)
    mi_change_negligible = bool(relative_mi_loss_without_output <= mi_change_threshold)
    type_i_accepted = bool(mi_change_negligible)
    selected_similarity_type = 'first_kind' if type_i_accepted else 'second_kind'
    reason = 'Type I was selected because removing output scaling causes no material relative loss of mutual information.' if type_i_accepted else 'Type II was selected because removing output scaling causes a material relative loss of mutual information.'
    return {'status': 'success', 'scope_id': str(scope_id), 'screening_k': _config.SIMILARITY_SCREEN_K, 'selected_similarity_type': selected_similarity_type, 'optimize_output': selected_similarity_type == 'second_kind', 'reason': reason, 'decision_rule': 'Type I iff removing output scaling causes no material relative loss of mutual information', 'n_samples': int(X_local.shape[0]), 'mi_method': local_method, 'mi_param': int(local_param), 'fixed_output_mutual_information': fixed_mi, 'joint_output_mutual_information': joint_mi, 'fixed_output_information_score': float(fixed_candidate['normalized_information_score']), 'joint_output_information_score': float(joint_candidate['normalized_information_score']), 'relative_mi_loss_without_output_scaling': float(relative_mi_loss_without_output), 'max_relative_mi_change': float(mi_change_threshold), 'mi_change_negligible': mi_change_negligible, 'screening_max_cosine_similarity': float(max_cosine), 'screening_fixed_in_power': {}, 'screening_fixed_out_power': {}, 'fixed_output_candidate': _similarity_screen_candidate_summary(fixed_candidate), 'joint_output_candidate': _similarity_screen_candidate_summary(joint_candidate)}


def run_local_workflow_with_similarity_resolution(X, Y, input_vars, basis_matrices, mi_method, mi_param, optimize_output: Optional[bool]=None, fixed_in_power=None, fixed_out_power=None, manual_hyperparams=None, local_workflow_options=None, scope_id='global'):
    """Resolve auto similarity locally, then run the unchanged final workflow."""
    hp = dict(manual_hyperparams or {})
    options = _validation._deep_update_dict(_config.GRADIENT_REGIME_DEFAULT_OPTIONS['local_workflow'], local_workflow_options or {})
    requested_similarity = str(options.get('similarity_type', 'first_kind'))
    screening = {'status': 'not_requested', 'scope_id': str(scope_id), 'selected_similarity_type': requested_similarity}
    resolved_similarity = requested_similarity
    resolved_optimize_output = None if optimize_output is None else bool(optimize_output)
    resolved_fixed_out_power = fixed_out_power
    resolved_options = dict(options)
    if requested_similarity == 'auto':
        screening = screen_similarity_type(X=X, Y=Y, input_vars=input_vars, basis_matrices=basis_matrices, mi_method=mi_method, mi_param=mi_param, manual_hyperparams=hp, local_workflow_options=options, scope_id=scope_id)
        screening['final_fixed_in_power'] = dict(fixed_in_power or {})
        screening['final_fixed_out_power'] = dict(fixed_out_power or {})
        _presentation.display_similarity_screening_summary(screening, hp)
        if screening.get('status') != 'success':
            return {'status': 'similarity_screening_failed', 'scope_id': str(scope_id), 'similarity_type_requested': 'auto', 'similarity_type': 'unresolved', 'optimize_output': False, 'similarity_screening': screening}
        resolved_similarity = str(screening['selected_similarity_type'])
        resolved_optimize_output = bool(screening['optimize_output'])
        resolved_fixed_out_power = fixed_out_power if resolved_optimize_output else None
        resolved_options['similarity_type'] = resolved_similarity
        resolved_options['optimize_output'] = resolved_optimize_output
        resolved_options['output_scaling_reason'] = str(screening['reason'])
    elif requested_similarity == 'first_kind':
        resolved_optimize_output = False
        resolved_fixed_out_power = None
        resolved_options['optimize_output'] = False
    elif requested_similarity == 'second_kind':
        resolved_optimize_output = True
        resolved_options['optimize_output'] = True
    else:
        return {'status': 'invalid_similarity_type', 'scope_id': str(scope_id), 'similarity_type': requested_similarity}
    local_result = run_local_dominant_group_workflow(X=X, Y=Y, input_vars=input_vars, basis_matrices=basis_matrices, mi_method=mi_method, mi_param=mi_param, optimize_output=resolved_optimize_output, fixed_in_power=fixed_in_power, fixed_out_power=resolved_fixed_out_power, manual_hyperparams=hp, local_workflow_options=resolved_options, scope_id=scope_id)
    local_result['similarity_type_requested'] = requested_similarity
    local_result['similarity_type'] = resolved_similarity
    local_result['optimize_output'] = bool(resolved_optimize_output)
    local_result['similarity_screening'] = screening
    return local_result


def _score_difference_uq_margin(previous_result: dict, current_result: dict) -> float:
    previous_uq = _plotting._candidate_uncertainty(previous_result)
    current_uq = _plotting._candidate_uncertainty(current_result)
    combined_previous_std = previous_uq["combined_score_std"]
    combined_current_std = current_uq["combined_score_std"]
    return float(np.hypot(combined_previous_std, combined_current_std))


def run_local_dominant_group_workflow(
    X,
    Y,
    input_vars,
    basis_matrices,
    mi_method,
    mi_param,
    optimize_output=False,
    fixed_in_power=None,
    fixed_out_power=None,
    manual_hyperparams=None,
    local_workflow_options=None,
    scope_id="global",
):
    """Execute the reusable single-mechanism adaptive workflow.

    Runtime policy:
      1. Test consecutive dominant counts k=1,2,... when data allow.
      2. Continue only when the absolute decrease in S=exp(-MI) reaches the
         configured threshold; otherwise lock the simpler preceding count.
      3. After locking, detect variables negligible in every discovered formula,
         construct the constrained Pi basis, and re-optimize.
      4. Rationalize coefficients and accept the rational form only when the MI
         degradation remains within tolerance.

    The same function is used once for a single-mechanism problem and once per
    detected regime for a multi-mechanism problem.
    """
    hp = dict(manual_hyperparams or {})
    surface_fits = {}
    if hp.get("_gpr_fixed_split") is not None and scope_id != "global":
        raise ValueError("A full-dataset fixed GPR split requires a global workflow, not a partitioned subset.")
    if hp.get("enable_workflow_visualization", True) and hp.get("enable_gpr_validation", True):
        hp["_gpr_surface_fits"] = surface_fits
    options = _validation._deep_update_dict(
        _config.GRADIENT_REGIME_DEFAULT_OPTIONS["local_workflow"],
        local_workflow_options or {},
    )
    similarity_type = "second_kind" if bool(optimize_output) else "first_kind"
    output_scaling_reason = str(
        options.get(
            "output_scaling_reason",
            "Dependent-variable scaling enabled."
            if optimize_output
            else "Dependent-variable scaling not requested.",
        )
    )
    output_scaling_confidence = str(
        options.get("output_scaling_confidence", "high")
    )

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    B_global = np.asarray(basis_matrices, dtype=float)
    if X.ndim != 2 or X.shape[0] != Y.shape[0]:
        return {"status": "invalid_local_data_shape", "scope_id": str(scope_id)}
    if B_global.ndim != 2 or B_global.shape[0] == 0:
        return {"status": "invalid_local_basis", "scope_id": str(scope_id)}

    finite_mask = _power_product_sample_mask(X, Y)
    X_local = X[finite_mask]
    Y_local = Y[finite_mask]
    if X_local.shape[0] == 0:
        return {"status": "no_valid_local_samples", "scope_id": str(scope_id)}

    planned_method = options.get("mi_method")
    if hp.get("mi_method_used") is not None:
        local_method = str(hp["mi_method_used"])
    elif hp.get("regime_mi_method") is not None:
        local_method = str(hp["regime_mi_method"])
    elif planned_method in (None, "auto"):
        local_method = str(mi_method)
    else:
        local_method = str(planned_method)

    planned_param = options.get("mi_param")
    if hp.get("mi_param_used") is not None:
        local_param = int(hp["mi_param_used"])
    elif hp.get("regime_mi_param") is not None:
        local_param = int(hp["regime_mi_param"])
    elif planned_param is not None:
        local_param = int(planned_param)
    else:
        local_param = int(mi_param)

    if local_method not in ("kraskov", "binning"):
        return {
            "status": "invalid_mi_method",
            "scope_id": str(scope_id),
            "mi_method": local_method,
        }
    if local_method == "binning" and X_local.shape[0] < _config.MI_BINNING_SAMPLE_THRESHOLD:
        local_method = "kraskov"
        local_param = min(5, max(2, X_local.shape[0] - 2))

    start_count = max(
        1,
        min(
            int(hp.get("workflow_start_count", options.get("start_count", 1))),
            B_global.shape[0],
        ),
    )
    requested_max = hp.get("workflow_max_count", options.get("max_count"))
    max_count = B_global.shape[0] if requested_max is None else int(requested_max)
    max_count = max(start_count, min(max_count, B_global.shape[0]))
    score_drop_threshold = float(
        hp.get(
            "abs_score_drop_threshold",
            options.get("absolute_score_drop_threshold", _config.ABS_SCORE_DROP_THRESHOLD),
        )
    )
    minimum_samples = int(
        hp.get("regime_min_samples", options.get("minimum_samples", 20))
    )
    minimum_per_dimension = int(
        hp.get(
            "regime_min_samples_per_mi_dimension",
            options.get("minimum_samples_per_mi_dimension", 20),
        )
    )
    require_two_counts = bool(
        hp.get(
            "workflow_require_two_counts_when_possible",
            options.get("require_two_counts_when_possible", True),
        )
    )
    minimum_samples_for_second_count = int(
        hp.get(
            "workflow_minimum_samples_for_second_count",
            options.get("minimum_samples_for_second_count", 40),
        )
    )
    planned_restarts = options.get("cma_restarts", options.get("restarts", 3))
    restart_count = max(1, int(_cma_hyperparameter(hp, "cma_restarts", planned_restarts)))


    _presentation.display_workflow_stage(
        f"Local adaptive workflow · {scope_id}",
        status="running",
        summary="Testing k sequentially, then applying sparsification and rationalization.",
        rows=[
            ("Samples", X_local.shape[0]),
            ("Base Pi dimension", B_global.shape[0]),
            ("Test range", f"{start_count}–{max_count}"),
            ("MI", f"{local_method} ({local_param})"),
            ("CMA-ES restarts", restart_count),
            ("Similarity", similarity_type.replace("_", " ")),
            ("Output optimization", bool(optimize_output)),
            (
                "Output free coefficients",
                (
                    B_global.shape[0] - len(fixed_out_power or {})
                    if optimize_output else 0
                ),
            ),
            ("Scaling confidence", output_scaling_confidence),
            ("Fixed input powers", fixed_in_power or "none"),
            ("Fixed output powers", fixed_out_power or "none"),
            (
                "Collinearity penalty",
                bool(hp.get("use_collinearity_penalty", int(max_count) > 1)),
            ),
            (
                "Maximum cosine",
                hp.get("max_cosine_similarity", 0.7),
            ),
            (
                "Penalize fixed coordinates",
                bool(hp.get("penalty_opt_w_fixed", True)),
            ),
        ],
        bullets=[output_scaling_reason],
        hyperparameters=hp,
    )

    dimension_path = []
    count_decisions = []
    locked_count = None
    locked_solution = None
    lock_reason = None

    for pi_count in range(start_count, max_count + 1):
        minimum_required = max(minimum_samples, minimum_per_dimension * (pi_count + 1))
        sample_guard = "standard"
        if (
            require_two_counts
            and pi_count == start_count + 1
            and max_count >= pi_count
        ):
            # The second count is essential for a data-driven lock decision.
            # This explicit, reproducible floor replaces the overly conservative
            minimum_required = min(minimum_required, minimum_samples_for_second_count)
            sample_guard = "second-count floor"
        if X_local.shape[0] < minimum_required:
            decision = {
                "event": "COUNT_TEST_SKIPPED_INSUFFICIENT_SAMPLES",
                "pi_count": int(pi_count),
                "n_samples": int(X_local.shape[0]),
                "minimum_required": int(minimum_required),
                "sample_guard": sample_guard,
            }
            count_decisions.append(decision)
            _presentation.display_count_decision(str(scope_id), decision, hp)
            break

        candidate = _run_local_restarts(
            X=X_local,
            Y=Y_local,
            basis_matrices=B_global,
            input_vars=input_vars,
            pi_count=pi_count,
            optimize_output=optimize_output,
            fixed_in_power=fixed_in_power,
            fixed_out_power=fixed_out_power,
            mi_method=local_method,
            mi_param=local_param,
            hp=hp,
            restart_count=restart_count,
        )
        if candidate.get("status") != "success":
            decision = {
                "event": "COUNT_OPTIMIZATION_FAILED",
                "pi_count": int(pi_count),
                "details": candidate,
            }
            count_decisions.append(decision)
            _presentation.display_count_decision(str(scope_id), decision, hp)
            break

        candidate = dict(candidate)
        candidate["stage"] = "count_test"
        candidate["similarity_type"] = similarity_type
        candidate["output_optimization"] = bool(optimize_output)
        candidate["output_scaling_reason"] = output_scaling_reason
        candidate["fixed_in_power"] = dict(fixed_in_power or {})
        candidate["fixed_out_power"] = dict(fixed_out_power or {})
        dimension_path.append(candidate)
        _presentation.display_local_candidate(str(scope_id), candidate, hp)

        if len(dimension_path) == 1:
            baseline_decision = {
                "event": "BASELINE_COUNT_TESTED",
                "pi_count": int(pi_count),
                "score": float(candidate["normalized_information_score"]),
            }
            count_decisions.append(baseline_decision)
            _presentation.display_count_decision(str(scope_id), baseline_decision, hp)
            if max_count == pi_count:
                locked_count = int(pi_count)
                locked_solution = candidate
                lock_reason = "maximum_count_reached_after_baseline"
                break
            continue

        previous = dimension_path[-2]
        current = dimension_path[-1]
        previous_score = float(previous["normalized_information_score"])
        current_score = float(current["normalized_information_score"])
        absolute_drop = previous_score - current_score
        uq_margin = _score_difference_uq_margin(previous, current)
        passes_threshold = bool(absolute_drop >= score_drop_threshold)
        exceeds_uq_margin = bool(absolute_drop > uq_margin)
        decision_record = {
            "event": "COUNT_COMPARISON",
            "previous_count": int(previous["pi_count"]),
            "current_count": int(current["pi_count"]),
            "previous_score": previous_score,
            "current_score": current_score,
            "absolute_score_drop": float(absolute_drop),
            "required_drop": score_drop_threshold,
            "passes_absolute_threshold": passes_threshold,
            "score_difference_uq_margin": uq_margin,
            "exceeds_uq_margin": exceeds_uq_margin,
        }
        count_decisions.append(decision_record)
        _presentation.display_count_decision(str(scope_id), decision_record, hp)

        if passes_threshold:
            if pi_count == max_count:
                locked_count = int(pi_count)
                locked_solution = current
                lock_reason = "maximum_count_reached_with_material_improvement"
                break
            continue

        locked_count = int(previous["pi_count"])
        locked_solution = previous
        lock_reason = "added_count_failed_absolute_score_drop_threshold"
        break

    if locked_solution is None and dimension_path:
        locked_solution = dimension_path[-1]
        locked_count = int(locked_solution["pi_count"])
        if lock_reason is None:
            lock_reason = "locked_last_successful_count_after_runtime_stop"

    if locked_solution is None:
        return {
            "status": "local_count_search_failed",
            "scope_id": str(scope_id),
            "dimension_path": dimension_path,
            "count_decisions": count_decisions,
            "n_samples": int(X_local.shape[0]),
        }

    selected = dict(locked_solution)
    selected["stage"] = "locked"
    selected_basis = B_global

    sparsification = {"status": "disabled"}
    sparsification_enabled = bool(
        hp.get(
            "workflow_sparsification_enabled",
            options.get("sparsification_enabled", True),
        )
    )
    if sparsification_enabled:
        if fixed_in_power or (optimize_output and fixed_out_power):
            sparsification = {
                "status": "disabled_with_fixed_basis_constraints",
                "reason": (
                    "Dropping physical variables changes the local null-space basis; "
                    "fixed powers in the original basis would no longer have the same meaning."
                ),
            }
        else:
            # Normalize input physical exponents specifically for sparsification,
            ### UPDATE: independently of the display normalization convention.
            raw_input_exponents = selected.get("input_physical_exponents_raw")
            if raw_input_exponents is None:
                raw_input_exponents = selected.get("input_physical_exponents", [])
            exponent_sets = []
            for row in raw_input_exponents:
                row = np.asarray(row, dtype=float)
                scale = float(np.max(np.abs(row))) if row.size else 0.0

                if scale <= 1e-14:
                    exponent_sets.append(np.zeros_like(row))
                else:
                    exponent_sets.append(row / scale)
            # Type-II output-rescaling exponents retain their physical magnitude.
            if optimize_output and selected.get("output_physical_exponents") is not None:
                exponent_sets.append(
                    np.asarray(selected["output_physical_exponents"], dtype=float)
                )
            exponent_array = (
                np.vstack(exponent_sets)
                if exponent_sets
                else np.empty((0, X_local.shape[1]))
            )
            ### UPDATE: independently of the display normalization convention.

            sparsify_threshold = float(
                hp.get(
                    "regime_sparsify_exponent_threshold",
                    options.get("sparsify_exponent_threshold", _config.SPARSE_THRES),
                )
            )
            drop_indices = (
                np.flatnonzero(np.max(np.abs(exponent_array), axis=0) < sparsify_threshold).tolist()
                if exponent_array.size
                else []
            )

            if not drop_indices:
                sparsification = {
                    "status": "no_negligible_variables",
                    "threshold": sparsify_threshold,
                }
            else:
                restricted_basis = _restricted_basis_after_dropping_variables(B_global, drop_indices)
                if restricted_basis.shape[0] < int(locked_count):
                    sparsification = {
                        "status": "rejected_basis_dimension_too_small",
                        "dropped_variables": [input_vars[index] for index in drop_indices],
                        "remaining_basis_dimension": int(restricted_basis.shape[0]),
                        "locked_count": int(locked_count),
                    }
                else:
                    sparse_candidate = _run_local_restarts(
                        X=X_local,
                        Y=Y_local,
                        basis_matrices=restricted_basis,
                        input_vars=input_vars,
                        pi_count=int(locked_count),
                        optimize_output=optimize_output,
                        fixed_in_power=None,
                        fixed_out_power=None,
                        mi_method=local_method,
                        mi_param=local_param,
                        hp=hp,
                        restart_count=restart_count,
                    )
                    if sparse_candidate.get("status") != "success":
                        sparsification = {
                            "status": "optimization_failed",
                            "dropped_variables": [input_vars[index] for index in drop_indices],
                        }
                    else:
                        score_increase = float(
                            sparse_candidate["normalized_information_score"]
                            - selected["normalized_information_score"]
                        )
                        maximum_increase = float(
                            hp.get(
                                "regime_sparsify_max_score_increase",
                                options.get("sparsify_max_score_increase", 0.05),
                            )
                        )
                        accepted = bool(score_increase <= maximum_increase)
                        sparsification = {
                            "status": "accepted" if accepted else "rejected",
                            "dropped_variables": [input_vars[index] for index in drop_indices],
                            "drop_indices": [int(index) for index in drop_indices],
                            "score_increase": score_increase,
                            "maximum_allowed_score_increase": maximum_increase,
                            "candidate": sparse_candidate,
                        }
                        if accepted:
                            selected = dict(sparse_candidate)
                            selected["stage"] = "sparsified_reoptimized"
                            selected_basis = restricted_basis

    rationalization_enabled = bool(
        hp.get(
            "rationalized_on",
            options.get("rationalization_enabled", True),
        )
    )
    if rationalization_enabled:
        rationalization_hp = dict(hp)
        rationalization_hp["max_rational_denominator"] = int(
            hp.get(
                "max_rational_denominator",
                options.get("max_rational_denominator", 5),
            )
        )
        rationalization = _rationalize_local_solution(
            selected,
            selected_basis,
            X_local,
            Y_local,
            input_vars,
            local_method,
            local_param,
            optimize_output,
            rationalization_hp,
        )
        final_solution = rationalization["final"]
    else:
        rationalization = {
            "accepted": False,
            "status": "disabled",
            "final": selected,
        }
        final_solution = selected

    local_result = {
        "status": "success",
        "scope_id": str(scope_id),
        "n_samples": int(X_local.shape[0]),
        "mi_method": local_method,
        "mi_param": int(local_param),
        "similarity_type": similarity_type,
        "optimize_output": bool(optimize_output),
        "output_scaling_reason": output_scaling_reason,
        "output_scaling_confidence": output_scaling_confidence,
        "fixed_in_power": dict(fixed_in_power or {}),
        "fixed_out_power": dict(fixed_out_power or {}),
        "basis_dimension": int(B_global.shape[0]),
        "tested_counts": [int(item["pi_count"]) for item in dimension_path],
        "dimension_path": dimension_path,
        "count_decisions": count_decisions,
        "locked_count": int(locked_count),
        "lock_reason": lock_reason,
        "locked_solution": locked_solution,
        "sparsification": sparsification,
        "rationalization": rationalization,
        "final_solution": final_solution,
        "review_flags": {
            "gpr_collapse_pass": final_solution.get("collapse_pass"),
            "restart_subspace_stable": (
                selected.get("restart_stability", {}).get("minimum_principal_cosine") is None
                or selected.get("restart_stability", {}).get("minimum_principal_cosine", 0.0)
                >= float(hp.get("regime_stability_cosine_threshold", 0.90))
            ),
        },
    }
    fit_record = surface_fits.get(id(final_solution.get("gpr_validation")))
    if locked_count == 2 and fit_record is not None:
        try:
            local_result["gpr_surface"] = _validation._prepare_gpr_surface(fit_record[1])
        except (ValueError, FloatingPointError, scispa.QhullError) as exc:
            local_result["gpr_surface_warning"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    _presentation.display_local_final(local_result, hp)
    return local_result


def optimize_regimes_with_mutual_information(
    X,
    Y,
    input_vars,
    basis_matrices,
    regime_detection,
    mi_method,
    mi_param,
    optimize_output: Optional[bool] = None,
    fixed_in_power=None,
    fixed_out_power=None,
    manual_hyperparams=None,
    local_workflow_options=None,
):
    """Run the adaptive single-mechanism workflow independently in each regime."""
    hp = dict(manual_hyperparams or {})
    if regime_detection.get("status") != "success":
        return {
            "status": "regime_detection_not_successful",
            "detection_status": regime_detection.get("status"),
        }

    options = _validation._deep_update_dict(
        _config.GRADIENT_REGIME_DEFAULT_OPTIONS["local_workflow"],
        local_workflow_options or {},
    )
    # Preserve None for auto similarity screening instead of collapsing it to
    # False. Explicit first-/second-kind policies are already canonicalized by
    # compile_workflow_plan.
    if options.get("similarity_type") == "auto":
        options["optimize_output"] = None
    elif options.get("optimize_output") is None:
        options["optimize_output"] = optimize_output

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    labels = np.asarray(regime_detection["labels"], dtype=int)
    valid_indices = np.asarray(regime_detection.get("valid_indices", np.arange(labels.size)), dtype=int)
    confidence = np.asarray(regime_detection.get("assignment_confidence", np.ones(labels.size)), dtype=float)
    if labels.size != valid_indices.size or labels.size != confidence.size:
        return {
            "status": "invalid_regime_detection_mapping",
            "message": "labels, valid_indices, and assignment_confidence must have the same length.",
        }

    confidence_threshold = float(
        options.get(
            "assignment_confidence_threshold",
            hp.get("regime_assignment_confidence_threshold", 0.05),
        )
    )
    exclude_ambiguous = bool(
        options.get(
            "exclude_low_confidence_points",
            hp.get("regime_exclude_ambiguous", False),
        )
    )
    minimum_cluster_samples = int(options.get("minimum_samples", hp.get("regime_min_samples", 60)))
    # Carry the mandatory accepted cluster floor through boundary filtering.
    acceptance = regime_detection.get("acceptance_criteria") or {}
    minimum_cluster_samples = max(
        minimum_cluster_samples, int(acceptance.get("minimum_cluster_samples", 60))
    )

    regime_results = []
    regime_ids = regime_detection.get("mi_expected_regime_ids")
    if regime_ids is None:
        regime_ids = np.unique(labels)
    for regime_id in sorted(regime_ids):
        regime_mask = labels == regime_id
        hard_indices = valid_indices[regime_mask]
        hard_confidence = confidence[regime_mask]
        ambiguous_count = int(np.sum(hard_confidence < confidence_threshold))

        if exclude_ambiguous:
            retained = hard_indices[hard_confidence >= confidence_threshold]
            if retained.size < minimum_cluster_samples:
                retained = hard_indices
                ambiguity_policy = "kept_boundary_points_because_exclusion_left_too_few_samples"
            else:
                ambiguity_policy = "excluded_low-confidence_boundary_points"
        else:
            retained = hard_indices
            ambiguity_policy = "kept_all_hard_assignments"

        if retained.size < minimum_cluster_samples:
            regime_results.append({
                "regime": int(regime_id),
                "scope_id": f"regime_{int(regime_id)}",
                "status": "insufficient_local_samples",
                "n_samples": int(retained.size),
                "minimum_required": int(minimum_cluster_samples),
            })
            continue

        local_result = run_local_workflow_with_similarity_resolution(
            X=X[retained],
            Y=Y[retained],
            input_vars=input_vars,
            basis_matrices=basis_matrices,
            mi_method=mi_method,
            mi_param=mi_param,
            optimize_output=options.get("optimize_output", optimize_output),
            fixed_in_power=fixed_in_power,
            fixed_out_power=fixed_out_power,
            manual_hyperparams=hp,
            local_workflow_options=options,
            scope_id=f"regime_{int(regime_id)}",
        )
        local_result.update({
            "regime": int(regime_id),
            "hard_assigned_samples": int(hard_indices.size),
            "optimized_samples": int(retained.size),
            "global_sample_indices": retained.astype(int).tolist(),
            "ambiguous_samples": ambiguous_count,
            "ambiguity_policy": ambiguity_policy,
        })
        regime_results.append(local_result)

    successful = [result for result in regime_results if result.get("status") == "success"]
    return {
        "status": "success" if successful else "no_successful_local_workflow",
        "method": "gradient clustering followed by adaptive regime-wise MI/CMA-ES workflows",
        "regime_count": int(len(np.unique(labels))),
        "successful_regime_count": int(len(successful)),
        "confidence_threshold": confidence_threshold,
        "excluded_ambiguous_points": exclude_ambiguous,
        "regimes": regime_results,
    }


# Observation wrappers preserve arguments, return values and optimizer choices.
screen_similarity_type = _recording.trace_scope("similarity_screening")(screen_similarity_type)
run_local_dominant_group_workflow = _recording.trace_scope("local_model_selection")(run_local_dominant_group_workflow)
_run_local_restarts = _recording.trace_scope("restart_batch")(_run_local_restarts)

# Conventional names; legacy names remain compatible.
estimate_mutual_information = global_calc_MI
combine_pi_basis = global_calc_pi
evaluate_power_product = global_coef_to_pi_mat
format_power_product = global_format_formula
