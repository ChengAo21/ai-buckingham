"""Mutual information, formulas, uncertainty and held-out collapse checks."""
from __future__ import annotations
from .shared import *


def _pydantic_dump(model: BaseModel) -> dict:
    """Support both Pydantic v1 and v2 without version branching elsewhere."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _pydantic_validate(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def global_calc_MI(x, y, mi_method, mi_param):
    if mi_method == "binning":
        # 分箱法
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


def global_coef_to_pi_mat(coef, X_data, is_log = False):
    if is_log:
        return np.sum(coef * np.log(X_data), axis = 1).reshape(-1, 1)
    else:
        return np.prod(X_data ** coef, axis = 1).reshape(-1, 1)


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


def normalized_information_score(mi_value: float) -> float:
    """Return exp(-MI) on [0, 1] after clipping negative finite-sample MI estimates."""
    mi_nonnegative = max(float(mi_value), 0.0)
    return float(np.exp(-mi_nonnegative))


def evaluate_gpr_collapse(
    input_pi,
    output_pi,
    max_samples,
    repeats,
    n_restarts_optimizer,
    log_features,
    log_target,
    constant_value_bounds,
    length_scale_bounds,
    noise_level_bounds,    
    r2_threshold: float = GPR_R2_THRESHOLD,
    test_size: float = 0.20,
    random_state: int = GLOBAL_RANDOM_SEED,
) -> dict:
    """Evaluate the discovered collapse with repeated held-out GPR tests.

    The reported R² is always out-of-sample. The function uses a deterministic
    random subsample when the dataset is larger than ``max_samples`` because
    exact GPR training scales cubically with the number of training samples.
    """
    X = np.asarray(input_pi, dtype=float)
    y = np.asarray(output_pi, dtype=float).reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y[:, 0])
    X, y = X[finite_mask], y[finite_mask]
    if X.shape[0] < 20:
        return {"status": "insufficient_data", "n_samples": int(X.shape[0])}
    if np.std(y[:, 0]) < 1e-14:
        return {"status": "constant_target", "n_samples": int(X.shape[0])}

    if log_features:
        if np.any(X <= 0):
            return {"status": "invalid_log_features", "message": "GPR log-features require positive Pi values."}
        X = np.log(X)
    if log_target:
        if np.any(y <= 0):
            return {"status": "invalid_log_target", "message": "GPR log-target requires a positive output."}
        y = np.log(y)

    rng = np.random.default_rng(random_state)
    if max_samples and X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    r2_values, rmse_values, nrmse_values = [], [], []
    kernel_strings = []
    fit_warnings = []
    for repeat_id in range(max(1, int(repeats))):
        split_seed = random_state + repeat_id
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed
        )

        kernel = (
            ConstantKernel(1.0, tuple(constant_value_bounds))
            * RBF(
                length_scale=np.ones(X.shape[1]),
                length_scale_bounds=tuple(length_scale_bounds),
            )
            + WhiteKernel(
                noise_level=1e-5,
                noise_level_bounds=tuple(noise_level_bounds),
            )
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=max(0, int(n_restarts_optimizer)),
            random_state=split_seed,
        )
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                gpr.fit(X_train, y_train)
            for caught in caught_warnings:
                fit_warnings.append({
                    "repeat": int(repeat_id),
                    "category": caught.category.__name__,
                    "message": str(caught.message),
                })
            y_pred = gpr.predict(X_test).reshape(-1, 1)
            y_true = y_test.ravel()
            r2 = float(r2_score(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            nrmse = float(rmse / (np.std(y_true) + 1e-12))
            r2_values.append(r2)
            rmse_values.append(rmse)
            nrmse_values.append(nrmse)
            kernel_strings.append(str(gpr.kernel_))
        except Exception as exc:
            return {"status": "gpr_failed", "message": str(exc), "n_samples": int(X.shape[0])}

    r2_mean = float(np.mean(r2_values))
    return {
        "status": "success",
        "evaluation_space": {
            "log_features": bool(log_features),
            "log_target": bool(log_target),
        },
        "n_samples": int(X.shape[0]),
        "test_size": float(test_size),
        "repeats": int(len(r2_values)),
        "test_r2_values": r2_values,
        "test_r2_mean": r2_mean,
        "test_r2_std": float(np.std(r2_values, ddof=1)) if len(r2_values) > 1 else 0.0,
        "test_r2_min": float(np.min(r2_values)),
        "test_rmse_mean": float(np.mean(rmse_values)),
        "test_nrmse_mean": float(np.mean(nrmse_values)),
        "r2_threshold": float(r2_threshold),
        "collapse_pass": bool(r2_mean >= r2_threshold),
        "fitted_kernels": kernel_strings,
        "fit_warnings": fit_warnings,
        "warning_policy": (
            "sklearn convergence warnings are captured and reported here instead of "
            "being emitted as raw notebook warnings"
        ),
        "kernel_bounds": {
            "constant_value": list(constant_value_bounds),
            "length_scale": list(length_scale_bounds),
            "noise_level": list(noise_level_bounds),
        },
    }


def estimate_mi_score_subsampling_uq(
    input_pi,
    output_pi,
    mi_method: str,
    mi_param: int,
    num_trials: int = 20,
    sample_fraction: float = 0.5,
    random_state: int = GLOBAL_RANDOM_SEED,
) -> dict:
    """Subsampling uncertainty for MI and exp(-MI); used as a diagnostic, not as a theorem."""
    X = np.asarray(input_pi, dtype=float)
    y = np.asarray(output_pi, dtype=float).reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y[:, 0])
    X, y = X[mask], y[mask]
    n = X.shape[0]
    min_needed = max(int(mi_param) + 3, 10)
    sample_n = max(min_needed, int(n * sample_fraction))
    if n < min_needed or sample_n > n:
        return {"status": "insufficient_data", "n_samples": int(n)}

    full_mi = max(float(global_calc_MI(X, y, mi_method, mi_param)), 0.0)
    rng = np.random.default_rng(random_state)
    mi_subsamples, score_subsamples = [], []
    for _ in range(max(1, int(num_trials))):
        idx = rng.choice(n, size=sample_n, replace=False)
        mi_sub = max(float(global_calc_MI(X[idx], y[idx], mi_method, mi_param)), 0.0)
        mi_subsamples.append(mi_sub)
        score_subsamples.append(normalized_information_score(mi_sub))
    return {
        "status": "success",
        "full_score": normalized_information_score(full_mi),
        "subsample_mi_mean": float(np.mean(mi_subsamples)),
        "subsample_mi_std": float(np.std(mi_subsamples, ddof=1)) if len(mi_subsamples) > 1 else 0.0,
        "subsample_score_mean": float(np.mean(score_subsamples)),
        "subsample_score_std": float(np.std(score_subsamples, ddof=1)) if len(score_subsamples) > 1 else 0.0,
        "score_ci95": [float(np.quantile(score_subsamples, 0.025)), float(np.quantile(score_subsamples, 0.975))],
        "num_trials": int(len(score_subsamples)),
        "sample_fraction": float(sample_fraction),
    }


def _deep_update_dict(base, updates):
    """Recursively merge nested dictionaries without mutating either input."""
    result = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update_dict(result[key], value)
        else:
            result[key] = value
    return result
