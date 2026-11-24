"""Validation extracted from the author implementation."""
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import scipy.spatial as scispa
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from . import config as _config
from . import optimization as _optimization


def normalized_information_score(mi_value: float) -> float:
    """Return exp(-MI) on [0, 1] after clipping negative finite-sample MI estimates."""
    mi_nonnegative = max(float(mi_value), 0.0)
    return float(np.exp(-mi_nonnegative))


def resolve_gpr_validation_options(hyperparameters=None):
    """Normalize public overrides once and validate the final GPR configuration."""
    overrides = dict(hyperparameters or {})
    accepted = set(_config.GPR_VALIDATION_DEFAULTS) | {"gpr_split", "gpr_fixed_split"}
    unknown = sorted(key for key in overrides if key.startswith("gpr_") and key not in accepted)
    if unknown:
        raise ValueError(f"Unknown GPR validation options: {unknown}")
    options = {**_config.GPR_VALIDATION_DEFAULTS,
               **{key: value for key, value in overrides.items() if key in _config.GPR_VALIDATION_DEFAULTS}}
    for key, minimum in (("gpr_n_splits", 1), ("gpr_n_restarts_optimizer", 0),
                         ("gpr_max_samples", 0), ("gpr_random_state", 0)):
        value = options[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
        options[key] = int(value)
    if options["gpr_random_state"] + options["gpr_n_splits"] - 1 >= 2**32:
        raise ValueError("GPR random seeds must remain below2**32 for every split.")
    for key in ("gpr_log_features", "gpr_log_target"):
        if not isinstance(options[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a boolean.")
        options[key] = bool(options[key])
    options["gpr_test_size"] = float(options["gpr_test_size"])
    if not 0.0 < options["gpr_test_size"] < 1.0:
        raise ValueError("gpr_test_size must lie strictly between0 and1.")
    options["gpr_r2_threshold"] = float(options["gpr_r2_threshold"])
    if not np.isfinite(options["gpr_r2_threshold"]):
        raise ValueError("gpr_r2_threshold must be finite.")
    column = options["gpr_stratify_column"]
    if column is not None and (not isinstance(column, str) or not column.strip()):
        raise ValueError("gpr_stratify_column must be a non-empty column name or None.")
    for key in ("gpr_constant_value_bounds", "gpr_length_scale_bounds", "gpr_noise_level_bounds"):
        bounds = options[key]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"{key} must contain two positive bounds.")
        lower, upper = map(float, bounds)
        if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
            raise ValueError(f"{key} must satisfy0 < lower < upper.")
        options[key] = (lower, upper)
    return options


def _resolve_gpr_validation_split(
    frame, specification=None, *, stratify_column=None, test_size=None,
    random_state=None,
):
    """Resolve the configured source-stratified split once for all representations."""
    if test_size is None:
        test_size = _config.GPR_VALIDATION_DEFAULTS["gpr_test_size"]
    if random_state is None:
        random_state = _config.GPR_VALIDATION_DEFAULTS["gpr_random_state"]
    if specification is None:
        if stratify_column is None:
            return None
        id_column = "case_id"
        if id_column not in frame.columns:
            # Metadata only: ordinary datasets need no manually supplied sample IDs.
            frame = frame.copy()
            id_column = "__gpr_row_id__"
            frame[id_column] = [str(i) for i in range(len(frame))]
        specification = {"method": "source_stratified", "id_column": id_column,
                         "stratify_column": stratify_column, "test_size": test_size}
    elif stratify_column is not None:
        raise ValueError("Use either the legacy split dictionary or gpr_stratify_column, not both.")
    if isinstance(specification, dict):
        specification = {"random_state": random_state, **specification}
    required = {"method", "id_column", "stratify_column", "test_size", "random_state"}
    if not isinstance(specification, dict) or set(specification) != required:
        raise ValueError("gpr_split requires method, id_column, stratify_column, test_size and random_state.")
    if specification["method"] != "source_stratified":
        raise ValueError("The supported gpr_split method is source_stratified.")
    id_column, source_column = specification["id_column"], specification["stratify_column"]
    if any(column not in frame.columns or frame[column].isna().any()
           for column in (id_column, source_column)):
        raise ValueError("Source-stratified GPR needs non-missing sample-ID and source columns.")
    train_rows, test_rows = train_test_split(
        np.arange(len(frame)), test_size=float(specification["test_size"]),
        random_state=int(specification["random_state"]), stratify=frame[source_column],
    )
    ids = frame[id_column].astype(str).tolist()
    resolved = _resolve_gpr_fixed_split(frame, {
        "id_column": id_column, "train_ids": [ids[i] for i in train_rows],
        "test_ids": [ids[i] for i in test_rows],
    })
    return {**resolved, "method": specification["method"], "stratify_column": source_column,
            "stratify_labels": frame[source_column].tolist(),
            "test_size": float(specification["test_size"]),
            "random_state": int(specification["random_state"])}


def _resolve_gpr_fixed_split(frame, specification):
    """Bind a published train/test manifest to sample IDs, not CSV row order."""
    if specification is None:
        return None
    required = {"id_column", "train_ids", "test_ids"}
    if not isinstance(specification, dict) or set(specification) != required:
        raise ValueError("gpr_fixed_split requires id_column, train_ids and test_ids.")
    column = specification["id_column"]
    if column not in frame.columns or frame[column].isna().any():
        raise ValueError("The fixed GPR split requires a non-missing sample-ID column.")
    sample_ids = frame[column].astype(str).tolist()
    train_ids, test_ids = specification["train_ids"], specification["test_ids"]
    if not all(isinstance(ids, list) and ids and all(isinstance(x, str) and x for x in ids)
               for ids in (train_ids, test_ids)):
        raise ValueError("Fixed GPR train/test IDs must be non-empty lists of strings.")
    assigned = train_ids + test_ids
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Fixed GPR sample IDs must be unique in the dataset.")
    if len(set(assigned)) != len(assigned):
        raise ValueError("Fixed GPR train/test IDs contain duplicates or overlap.")
    if set(assigned) != set(sample_ids):
        raise ValueError("Fixed GPR train/test IDs must cover the dataset exactly; no missing or extra samples.")
    rows = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    return {"id_column": column, "sample_ids": sample_ids,
            "train_ids": list(train_ids), "test_ids": list(test_ids),
            "train_rows": [rows[x] for x in train_ids],
            "test_rows": [rows[x] for x in test_ids]}


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
    r2_threshold: float = _config.GPR_VALIDATION_DEFAULTS["gpr_r2_threshold"],
    test_size: float = _config.GPR_VALIDATION_DEFAULTS["gpr_test_size"],
    random_state: int = _config.GPR_VALIDATION_DEFAULTS["gpr_random_state"],
    *,
    on_first_fit=None,
    fixed_split=None,
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

    n_input = int(X.shape[0])
    source_stratified = fixed_split is not None and fixed_split.get("method") == "source_stratified"
    if fixed_split is not None:
        if not source_stratified and int(repeats) != 1:
            raise ValueError("An explicit-ID GPR split requires exactly one validation repeat.")
        if max_samples and int(max_samples) < n_input:
            raise ValueError("A fixed GPR split cannot be subsampled by max_samples.")
        ids = fixed_split.get("sample_ids", [])
        if len(ids) != n_input:
            raise ValueError("Fixed GPR split does not match the current sample count or scope.")
        # Revalidate direct calls as well as manifests resolved by the workflow.
        checked_split = _resolve_gpr_fixed_split(
            pd.DataFrame({fixed_split["id_column"]: ids}),
            {key: fixed_split[key] for key in ("id_column", "train_ids", "test_ids")},
        )
        if any(fixed_split.get(key) != checked_split[key] for key in ("train_rows", "test_rows")):
            raise ValueError("Fixed GPR row indices disagree with the supplied sample IDs.")
        if source_stratified:
            labels = fixed_split.get("stratify_labels")
            if labels is None and int(repeats) != 1:
                raise ValueError("Repeated source-stratified GPR requires source labels in sample-ID order.")
            if labels is not None:
                if len(labels) != n_input or pd.isna(labels).any():
                    raise ValueError("GPR source labels must match every sample ID without missing values.")
                expected_train, expected_test = train_test_split(
                    np.arange(n_input), test_size=float(fixed_split["test_size"]),
                    random_state=int(fixed_split["random_state"]), stratify=labels,
                )
                if (expected_train.tolist() != fixed_split["train_rows"] or
                        expected_test.tolist() != fixed_split["test_rows"]):
                    raise ValueError("Resolved GPR split disagrees with its source labels or seed.")
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y[:, 0])
    log_domain_mask = np.ones(n_input, dtype=bool)
    if log_features:
        log_domain_mask &= np.all(X > 0, axis=1)
    if log_target:
        log_domain_mask &= y[:, 0] > 0
    sample_selection = {
        "n_input": n_input,
        "n_removed_nonfinite": int(np.sum(~finite_mask)),
        "n_removed_log_domain": int(np.sum(finite_mask & ~log_domain_mask)),
        "policy": "Filter only values outside the requested logarithm domain; no clipping or value replacement.",
    }
    if fixed_split is not None and not np.all(finite_mask & log_domain_mask):
        raise ValueError("A fixed GPR split cannot silently discard non-finite or log-domain samples.")
    X, y = X[finite_mask & log_domain_mask], y[finite_mask & log_domain_mask]
    sample_selection["n_before_cap"] = int(X.shape[0])
    if X.shape[0] < 20:
        return {"status": "insufficient_data", "n_samples": int(X.shape[0]), "sample_selection": sample_selection}
    if np.std(y[:, 0]) < 1e-14:
        return {"status": "constant_target", "n_samples": int(X.shape[0]), "sample_selection": sample_selection}

    observations = np.column_stack((X, y)) if on_first_fit is not None else None
    if log_features:
        X = np.log(X)
    if log_target:
        y = np.log(y)

    rng = np.random.default_rng(random_state)
    if max_samples and X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        if observations is not None:
            observations = observations[idx]

    r2_values, rmse_values, nrmse_values = [], [], []
    kernel_strings = []
    fit_warnings = []
    split_history = []
    for repeat_id in range(max(1, int(repeats))):
        optimizer_seed = random_state + repeat_id
        split_seed = (int(fixed_split["random_state"]) + repeat_id
                      if source_stratified else optimizer_seed)
        split_arrays = (X, y)
        if repeat_id == 0 and observations is not None:
            split_arrays += (observations,)
        if fixed_split is None:
            X_train, X_test, y_train, y_test, *observed_split = train_test_split(
                *split_arrays, test_size=test_size, random_state=split_seed
            )
        else:
            if source_stratified and repeat_id > 0:
                train_rows, test_rows = train_test_split(
                    np.arange(n_input), test_size=float(fixed_split["test_size"]),
                    random_state=split_seed, stratify=fixed_split["stratify_labels"],
                )
            else:
                train_rows = np.asarray(fixed_split["train_rows"], dtype=int)
                test_rows = np.asarray(fixed_split["test_rows"], dtype=int)
            split_history.append({
                "repeat": int(repeat_id), "split_random_state": int(split_seed),
                "optimizer_random_state": int(optimizer_seed),
                "train_rows": train_rows.tolist(), "test_rows": test_rows.tolist(),
                "train_ids": [fixed_split["sample_ids"][i] for i in train_rows],
                "test_ids": [fixed_split["sample_ids"][i] for i in test_rows],
            })
            split_values = [part for array in split_arrays for part in (array[train_rows], array[test_rows])]
            X_train, X_test, y_train, y_test, *observed_split = split_values

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
            random_state=optimizer_seed,
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
            if observed_split:
                first_fit = (gpr, *observed_split, split_seed, log_features, log_target)
        except Exception as exc:
            return {"status": "gpr_failed", "message": str(exc), "n_samples": int(X.shape[0])}

    if observations is not None:
        on_first_fit(first_fit)
    r2_mean = float(np.mean(r2_values))
    return {
        "status": "success",
        "evaluation_space": {
            "log_features": bool(log_features),
            "log_target": bool(log_target),
        },
        "sample_selection": {**sample_selection, "n_used": int(X.shape[0])},
        "n_samples": int(X.shape[0]),
        "test_size": float(fixed_split["test_size"] if source_stratified else test_size),
        "repeats": int(len(r2_values)),
        **({"fixed_split": fixed_split, "split_history": split_history,
            "split_policy": "repeated_source_stratified" if source_stratified else "explicit_sample_ids"}
           if fixed_split is not None else {}),
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


def _prepare_gpr_surface(fit):
    """Sample the fitted response inside the training hull on the observed scale."""
    gpr, train, test, seed, log_features, log_target = fit
    coordinates = train[:, :2]
    lower = coordinates.min(axis=0)
    span = np.ptp(coordinates, axis=0)
    if np.any(span == 0):
        raise ValueError("Training coordinates do not span a two-dimensional surface.")

    # Unit ranges make the hull calculation robust to disparate Pi scales.
    scaled_coordinates = (coordinates - lower) / span
    hull = scispa.Delaunay(scaled_coordinates)
    axis = np.linspace(0, 1, 58)
    grid = np.column_stack([component.ravel() for component in np.meshgrid(axis, axis)])
    nodes = np.unique(np.vstack((grid[hull.find_simplex(grid) >= 0], scaled_coordinates)), axis=0)
    triangles = scispa.Delaunay(nodes).simplices
    features = nodes * span + lower
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        predictions = gpr.predict(np.log(features) if log_features else features).ravel()
        if log_target:
            predictions = np.exp(predictions)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("The GPR surface has non-finite predictions on the observed scale.")
    return {
        "split_seed": int(seed),
        "train_observations": train.tolist(),
        "test_observations": test.tolist(),
        "coordinates": features.tolist(),
        "predictions": predictions.tolist(),
        "triangles": triangles.tolist(),
    }


def estimate_mi_score_subsampling_uq(
    input_pi,
    output_pi,
    mi_method: str,
    mi_param: int,
    num_trials: int = 20,
    sample_fraction: float = 0.5,
    random_state: int = _config.GLOBAL_RANDOM_SEED,
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

    full_mi = max(float(_optimization.global_calc_MI(X, y, mi_method, mi_param)), 0.0)
    rng = np.random.default_rng(random_state)
    mi_subsamples, score_subsamples = [], []
    for _ in range(max(1, int(num_trials))):
        idx = rng.choice(n, size=sample_n, replace=False)
        mi_sub = max(float(_optimization.global_calc_MI(X[idx], y[idx], mi_method, mi_param)), 0.0)
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
