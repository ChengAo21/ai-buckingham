"""GPR gradients and eigenvalue-weighted physical regime identification."""
from __future__ import annotations
from .shared import *


def _schmidt_orthonormalize_basis_rows(B, tol=1e-12):
    """Modified Schmidt orthonormalization preserving the supplied row order.

    CMAME 2024 constructs the base Pi exponent vectors through Schmidt
    orthogonalization while preserving the supplied row order.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.size == 0:
        return B
    q_rows = []
    for row in B:
        v = row.astype(float, copy=True)
        for q in q_rows:
            v -= np.dot(v, q) * q
        norm = np.linalg.norm(v)
        if norm > tol:
            q_rows.append(v / norm)
    return np.asarray(q_rows, dtype=float)


def _normalize_regime_gpr_mean_mode(mean_mode: str) -> str:
    """Validate the surrogate mean-function name."""
    mode = str(mean_mode)
    allowed = {"constant", "affine_sigmoid_transition"}
    if mode not in allowed:
        raise ValueError(
            "Gradient-regime GPR mean_mode must be 'constant' or "
            "'affine_sigmoid_transition'."
        )
    return mode


def _regime_gpr_mean_parameter_count(input_dim, mean_mode):
    mean_mode = _normalize_regime_gpr_mean_mode(mean_mode)
    if mean_mode == "constant":
        return 1
    return 2 * int(input_dim) + 3


def _unpack_regime_gpr_mean_parameters(theta_mean, input_dim, mean_mode):
    theta_mean = np.asarray(theta_mean, dtype=float)
    mean_mode = _normalize_regime_gpr_mean_mode(mean_mode)
    if mean_mode == "constant":
        return {"beta": float(theta_mean[0])}
    d = int(input_dim)
    cursor = 0
    beta = float(theta_mean[cursor]); cursor += 1
    w1 = theta_mean[cursor:cursor + d]; cursor += d
    b1 = float(theta_mean[cursor]); cursor += 1
    w2 = theta_mean[cursor:cursor + d]; cursor += d
    b2 = float(theta_mean[cursor])
    return {"beta": beta, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _regime_gpr_mean_and_parameter_jacobian(X, theta_mean, mean_mode):
    """Evaluate the surrogate mean and its parameter Jacobian."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mean_mode = _normalize_regime_gpr_mean_mode(mean_mode)
    pars = _unpack_regime_gpr_mean_parameters(theta_mean, d, mean_mode)
    if mean_mode == "constant":
        mean = np.full(n, pars["beta"], dtype=float)
        jacobian = np.ones((n, 1), dtype=float)
        return mean, jacobian

    a = X @ pars["w1"] + pars["b1"]
    z = X @ pars["w2"] + pars["b2"]
    sigmoid = expit(z)
    sigmoid_prime = sigmoid * (1.0 - sigmoid)
    mean = pars["beta"] + a * sigmoid

    # Parameter order: beta, w1, b1, w2, b2.
    jacobian = np.column_stack([
        np.ones(n),
        X * sigmoid[:, None],
        sigmoid,
        X * (a * sigmoid_prime)[:, None],
        a * sigmoid_prime,
    ])
    return mean, jacobian


def _regime_gpr_mean_input_gradient(X, theta_mean, mean_mode):
    """Analytic input gradient of the selected surrogate mean function."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mean_mode = _normalize_regime_gpr_mean_mode(mean_mode)
    pars = _unpack_regime_gpr_mean_parameters(theta_mean, d, mean_mode)
    if mean_mode == "constant":
        return np.zeros((n, d), dtype=float)

    a = X @ pars["w1"] + pars["b1"]
    z = X @ pars["w2"] + pars["b2"]
    sigmoid = expit(z)
    sigmoid_prime = sigmoid * (1.0 - sigmoid)
    return (
        sigmoid[:, None] * pars["w1"][None, :]
        + (a * sigmoid_prime)[:, None] * pars["w2"][None, :]
    )


def _pairwise_squared_distance(X1, X2):
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    x1_sq = np.sum(X1 * X1, axis=1)[:, None]
    x2_sq = np.sum(X2 * X2, axis=1)[None, :]
    return np.maximum(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), 0.0)


class _GradientRegimeGPRModel:
    """Gaussian-process surrogate used to obtain gradients for regime detection.

    The covariance is isotropic:
        sigma_0^2 exp(-sigma_1^2 ||x-x'||^2)
    where x = log(Pi).  The response supplied to this model is assumed to be
    dimensionless.  The class exposes the analytic gradient of the posterior
    mean, which is used only for gradient-based clustering.
    """

    def __init__(self, X_train, y_train, theta, mean_mode, jitter, optimization_result):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.theta = np.asarray(theta, dtype=float)
        self.mean_mode = _normalize_regime_gpr_mean_mode(mean_mode)
        self.jitter = float(jitter)
        self.optimization_result = optimization_result
        self.input_dim = self.X_train.shape[1]
        self.n_mean_parameters = _regime_gpr_mean_parameter_count(
            self.input_dim, self.mean_mode
        )
        self.theta_mean = self.theta[:self.n_mean_parameters]
        self.log_sigma0_sq, self.log_sigma1_sq, self.log_sigma_u_sq = self.theta[-3:]
        self.sigma0_sq = float(np.exp(self.log_sigma0_sq))
        self.sigma1_sq = float(np.exp(self.log_sigma1_sq))
        self.sigma_u_sq = float(np.exp(self.log_sigma_u_sq))
        self._prepare_posterior()

    def _prepare_posterior(self):
        mean_train, _ = _regime_gpr_mean_and_parameter_jacobian(
            self.X_train, self.theta_mean, self.mean_mode
        )
        residual = self.y_train - mean_train
        r2 = _pairwise_squared_distance(self.X_train, self.X_train)
        self.K_signal = self.sigma0_sq * np.exp(-self.sigma1_sq * r2)
        K_y = self.K_signal + (self.sigma_u_sq + self.jitter) * np.eye(
            self.X_train.shape[0]
        )
        self.cho = cho_factor(K_y, lower=True, check_finite=False)
        self.dual_coef = cho_solve(self.cho, residual, check_finite=False)

    def predict_mean(self, X_eval):
        X_eval = np.asarray(X_eval, dtype=float)
        mean_eval, _ = _regime_gpr_mean_and_parameter_jacobian(
            X_eval, self.theta_mean, self.mean_mode
        )
        r2 = _pairwise_squared_distance(X_eval, self.X_train)
        K_cross = self.sigma0_sq * np.exp(-self.sigma1_sq * r2)
        return mean_eval + K_cross @ self.dual_coef

    def predict_gradient(self, X_eval):
        """Evaluate the posterior-mean gradient in raw x=log(Pi) coordinates."""
        X_eval = np.asarray(X_eval, dtype=float)
        gradient_mean = _regime_gpr_mean_input_gradient(
            X_eval, self.theta_mean, self.mean_mode
        )
        difference = self.X_train[None, :, :] - X_eval[:, None, :]
        squared_distance = np.sum(difference * difference, axis=2)
        K_cross = self.sigma0_sq * np.exp(-self.sigma1_sq * squared_distance)
        gradient_kernel = np.einsum(
            "ni,nij->nj",
            K_cross * self.dual_coef[None, :],
            2.0 * self.sigma1_sq * difference,
        )
        return gradient_mean + gradient_kernel

    def summary(self):
        mean_parameters = _unpack_regime_gpr_mean_parameters(
            self.theta_mean, self.input_dim, self.mean_mode
        )
        serializable_mean = {
            key: (
                np.asarray(value, dtype=float).tolist()
                if isinstance(value, np.ndarray)
                else float(value)
            )
            for key, value in mean_parameters.items()
        }
        return {
            "implementation": "gradient_regime_gpr_surrogate",
            "mean_mode": self.mean_mode,
            "mean_parameters": serializable_mean,
            "sigma0_squared": self.sigma0_sq,
            "sigma1_squared": self.sigma1_sq,
            "sigma_u_squared": self.sigma_u_sq,
            "negative_log_likelihood": float(self.optimization_result.fun),
            "optimizer_success": bool(self.optimization_result.success),
            "optimizer_message": str(self.optimization_result.message),
            "optimizer_iterations": int(
                getattr(self.optimization_result, "nit", -1)
            ),
            "optimizer_restarts": list(
                getattr(self, "restart_diagnostics", []) or []
            ),
            "n_train_samples": int(self.X_train.shape[0]),
            "coordinate_system": "raw x = log(Pi)",
            "target": "dimensionless Y supplied by the caller",
            "kernel": (
                "sigma0^2 * exp(-sigma1^2 * ||x-x'||^2) "
                "+ sigma_u^2 I"
            ),
        }


def _initial_regime_gpr_theta(X, y, mean_mode, rng, restart_id, options):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    y_scale = max(float(np.std(y)), 1e-8)
    beta = float(np.mean(y))

    if mean_mode == "constant":
        theta_mean = np.array([beta], dtype=float)
    else:
        design = np.column_stack([X, np.ones(n)])
        try:
            linear_solution, *_ = np.linalg.lstsq(design, y, rcond=None)
            linear_w = linear_solution[:-1]
            linear_b = float(linear_solution[-1])
        except Exception:
            linear_w = np.zeros(d)
            linear_b = beta
        w1 = 2.0 * linear_w
        b1 = 2.0 * (linear_b - beta)
        direction = rng.normal(size=d)
        direction /= max(np.linalg.norm(direction), 1e-12)
        x_std = np.maximum(np.std(X, axis=0), 1e-8)
        w2 = direction / x_std
        b2 = -float(np.median(X @ w2))
        theta_mean = np.concatenate([[beta], w1, [b1], w2, [b2]])

    # Robust distance scale for the isotropic kernel in Eq. (A.3).
    sample_count = min(n, 400)
    sample_idx = rng.choice(n, size=sample_count, replace=False) if n > sample_count else np.arange(n)
    r2_sample = _pairwise_squared_distance(X[sample_idx], X[sample_idx])
    nonzero = r2_sample[r2_sample > 1e-14]
    median_r2 = float(np.median(nonzero)) if nonzero.size else 1.0
    sigma1_sq = 1.0 / max(median_r2, 1e-12)
    sigma0_sq = max(float(np.var(y)), 1e-8)
    sigma_u_sq = max(
        sigma0_sq * float(options.get("initial_noise_fraction", 1e-6)), 1e-12
    )
    theta_kernel = np.log([sigma0_sq, sigma1_sq, sigma_u_sq])

    theta = np.concatenate([theta_mean, theta_kernel])
    if restart_id > 0:
        theta[:theta_mean.size] += rng.normal(scale=0.25 * y_scale, size=theta_mean.size)
        # Avoid destroying the sigmoid transition initialization too aggressively.
        # if mean_mode == "affine_sigmoid_transition":
        #     theta[1 + d + 1:1 + d + 1 + d] += rng.normal(scale=0.25, size=d)
        theta[-3:] += rng.normal(scale=1.0, size=3)
    return theta


def _fit_gradient_regime_gpr(X, y, options=None):
    """Maximum-likelihood fit of the GPR surrogate used for gradient clustering."""
    from .information import _deep_update_dict
    options = _deep_update_dict(
        GRADIENT_REGIME_DEFAULT_OPTIONS["surrogate"], options or {}
    )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(
            "Gradient-regime GPR requires X with shape (N, d) and y with N values."
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(
            "Gradient-regime GPR received non-finite training data."
        )

    mean_mode = _normalize_regime_gpr_mean_mode(
        options.get("mean_mode", "constant")
    )
    n_mean = _regime_gpr_mean_parameter_count(X.shape[1], mean_mode)
    random_state = int(options.get("random_state", 42))
    rng = np.random.default_rng(random_state)

    max_train_samples = options.get("max_train_samples")
    if max_train_samples is not None and X.shape[0] > int(max_train_samples):
        train_idx = rng.choice(
            X.shape[0], size=int(max_train_samples), replace=False
        )
        X_train, y_train = X[train_idx], y[train_idx]
    else:
        X_train, y_train = X, y

    pairwise_r2 = _pairwise_squared_distance(X_train, X_train)
    identity = np.eye(X_train.shape[0])
    jitter = float(options.get("jitter", 1e-10))

    mean_bound = tuple(options.get("mean_parameter_bounds", (None, None)))
    bounds = [mean_bound] * n_mean + [
        tuple(options.get("log_sigma0_sq_bounds", (-20.0, 20.0))),
        tuple(options.get("log_sigma1_sq_bounds", (-20.0, 20.0))),
        tuple(options.get("log_sigma_u_sq_bounds", (-30.0, 10.0))),
    ]

    def objective_and_gradient(theta):
        theta = np.asarray(theta, dtype=float)
        theta_mean = theta[:n_mean]
        log_sigma0_sq, log_sigma1_sq, log_sigma_u_sq = theta[-3:]
        sigma0_sq = float(np.exp(log_sigma0_sq))
        sigma1_sq = float(np.exp(log_sigma1_sq))
        sigma_u_sq = float(np.exp(log_sigma_u_sq))

        mean_train, mean_jacobian = _regime_gpr_mean_and_parameter_jacobian(
            X_train, theta_mean, mean_mode
        )
        residual = y_train - mean_train
        K_signal = sigma0_sq * np.exp(-sigma1_sq * pairwise_r2)
        K_y = K_signal + (sigma_u_sq + jitter) * identity
        try:
            cho = cho_factor(K_y, lower=True, check_finite=False)
            alpha_vector = cho_solve(cho, residual, check_finite=False)
            nll = (
                0.5 * float(residual @ alpha_vector)
                + float(np.sum(np.log(np.diag(cho[0]))))
                + 0.5 * X_train.shape[0] * np.log(2.0 * np.pi)
            )
            K_inverse = cho_solve(cho, identity, check_finite=False)
        except Exception:
            return 1e50, np.zeros_like(theta)

        gradient_mean = -(mean_jacobian.T @ alpha_vector)
        Q = K_inverse - np.outer(alpha_vector, alpha_vector)
        dK_log_sigma0 = K_signal
        dK_log_sigma1 = -sigma1_sq * pairwise_r2 * K_signal
        dK_log_noise = sigma_u_sq * identity
        gradient_kernel = 0.5 * np.array([
            np.sum(Q * dK_log_sigma0),
            np.sum(Q * dK_log_sigma1),
            np.sum(Q * dK_log_noise),
        ])
        gradient = np.concatenate([gradient_mean, gradient_kernel])
        if not np.isfinite(nll) or not np.all(np.isfinite(gradient)):
            return 1e50, np.zeros_like(theta)
        return nll, gradient

    results = []
    restart_diagnostics = []
    n_restarts = max(1, int(options.get("n_restarts_optimizer", 5)))
    for restart_id in range(n_restarts):
        theta0 = _initial_regime_gpr_theta(
            X_train, y_train, mean_mode, rng, restart_id, options
        )
        result = minimize(
            objective_and_gradient,
            theta0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(options.get("optimizer_maxiter", 500)),
                "ftol": float(options.get("optimizer_ftol", 1e-10)),
                "gtol": float(options.get("optimizer_gtol", 1e-6)),
            },
        )
        restart_diagnostics.append({
            "restart_id": int(restart_id),
            "negative_log_likelihood": (
                float(result.fun) if np.isfinite(result.fun) else None
            ),
            "success": bool(result.success),
            "iterations": int(getattr(result, "nit", -1)),
            "message": str(result.message),
        })

        if np.isfinite(result.fun):
            results.append(result)

    if not results:
        raise RuntimeError(
            "Gradient-regime GPR maximum-likelihood optimization failed."
        )

    successful = [result for result in results if bool(result.success)]
    if successful:
        best_result = min(successful, key=lambda result: float(result.fun))
    else:
        best_result = min(results, key=lambda result: float(result.fun))
        if bool(options.get("require_optimizer_success", False)):
            raise RuntimeError(
                "All gradient-regime GPR optimizer restarts stopped without "
                f"success. Best message: {best_result.message}"
            )
        warnings.warn(
            "All gradient-regime GPR restarts stopped without optimizer "
            "success; the best finite solution is used and reported.",
            RuntimeWarning,
        )

    model = _GradientRegimeGPRModel(
        X_train=X_train,
        y_train=y_train,
        theta=best_result.x,
        mean_mode=mean_mode,
        jitter=jitter,
        optimization_result=best_result,
    )
    model.restart_diagnostics = restart_diagnostics
    return model


def _fit_gpr_gradients(
    X,
    y,
    random_state=42,
    max_train_samples=None,
    finite_difference_step=None,
    gradient_gpr_options=None,
):
    """Fit the surrogate and evaluate its analytic posterior-mean gradients."""
    options = dict(gradient_gpr_options or {})
    options.setdefault("random_state", random_state)
    if max_train_samples is not None:
        options.setdefault("max_train_samples", max_train_samples)
    model = _fit_gradient_regime_gpr(X, y, options=options)
    gradients = model.predict_gradient(np.asarray(X, dtype=float))
    return gradients, model.summary()


def _gradient_similarity_matrix(gradients, eigenpairs):
    """Compute the eigenvalue-weighted gradient-to-cluster similarities."""
    gradients = np.asarray(gradients, dtype=float)
    similarities = np.empty(
        (gradients.shape[0], len(eigenpairs)), dtype=float
    )
    for cluster_id, (eigenvalues, eigenvectors) in enumerate(eigenpairs):
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        eigenvectors = np.asarray(eigenvectors, dtype=float)
        eigen_norm = max(np.linalg.norm(eigenvalues), 1e-15)
        weights = (eigenvalues / eigen_norm) ** 2
        projections = gradients @ eigenvectors
        similarities[:, cluster_id] = np.sum(
            weights[None, :] * projections ** 2,
            axis=1,
        )
    return similarities


def _cluster_gradient_eigenpairs(
    gradients,
    labels,
    n_clusters,
    gradient_norm_tolerance=1e-12,
):
    """Compute normalized-gradient covariance eigenpairs for every cluster."""
    gradients = np.asarray(gradients, dtype=float)
    norms = np.linalg.norm(gradients, axis=1)
    if np.any(norms <= float(gradient_norm_tolerance)):
        return None, None

    normalized_gradients = gradients / norms[:, None]
    eigenpairs = []
    covariance_matrices = []
    for cluster_id in range(int(n_clusters)):
        idx = np.asarray(labels) == cluster_id
        if not np.any(idx):
            return None, None

        C = (
            normalized_gradients[idx].T
            @ normalized_gradients[idx]
            / int(np.sum(idx))
        )
        # Each summand u u^T has trace one; this is a useful implementation
        # invariant for the normalized covariance.
        if not np.isclose(np.trace(C), 1.0, atol=1e-8, rtol=1e-8):
            warnings.warn(
                "Normalized-gradient covariance trace differs from one.",
                RuntimeWarning,
            )

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        eigenpairs.append((eigenvalues, eigenvectors))
        covariance_matrices.append(C)
    return eigenpairs, covariance_matrices


def _random_initial_eigenpairs(input_dim, n_clusters, rng):
    """Random positive eigenvalues and random orthogonal eigenvectors."""
    eigenpairs = []
    for _ in range(int(n_clusters)):
        random_matrix = rng.normal(size=(input_dim, input_dim))
        eigenvectors, _ = np.linalg.qr(random_matrix)
        eigenvalues = np.sort(
            rng.uniform(0.05, 1.0, size=input_dim)
        )[::-1]
        eigenpairs.append((eigenvalues, eigenvectors))
    return eigenpairs


def _cluster_by_gradient_similarity(
    gradients,
    n_clusters,
    random_state=42,
    max_iter=200,
    n_init=30,
    initialization="random_eigenpairs",
    max_initialization_attempts=100,
    require_convergence=True,
    gradient_norm_tolerance=1e-12,
):
    """Iteratively cluster samples using their surrogate-gradient directions.

    Repeated random starts are compared with the clustering objective only for
    the same K.  A candidate that has not reached a fixed label assignment is
    rejected when ``require_convergence`` is true.
    """
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 2 or gradients.shape[0] < int(n_clusters):
        return None
    if np.any(
        np.linalg.norm(gradients, axis=1)
        <= float(gradient_norm_tolerance)
    ):
        raise ValueError(
            "Directionless gradients must be filtered before clustering."
        )

    n_samples, input_dim = gradients.shape
    rng_master = np.random.default_rng(int(random_state))
    candidates = []

    for init_id in range(max(1, int(n_init))):
        initialized = False
        seed = None
        for _ in range(max(1, int(max_initialization_attempts))):
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            if initialization != "random_eigenpairs":
                raise ValueError(
                    "Gradient clustering currently supports only "
                    "initialization='random_eigenpairs'."
                )

            eigenpairs = _random_initial_eigenpairs(
                input_dim, n_clusters, rng
            )
            similarities = _gradient_similarity_matrix(
                gradients, eigenpairs
            )
            labels = np.argmax(similarities, axis=1)
            if len(np.unique(labels)) == int(n_clusters):
                initialized = True
                break

        if not initialized:
            continue

        converged = False
        cycle_detected = False
        termination_reason = "maximum_iterations"
        covariance_matrices = None
        seen_partitions = {labels.tobytes()}

        for iteration in range(max(1, int(max_iter))):
            eigenpairs, covariance_matrices = _cluster_gradient_eigenpairs(
                gradients,
                labels,
                n_clusters,
                gradient_norm_tolerance=gradient_norm_tolerance,
            )
            if eigenpairs is None:
                termination_reason = "empty_or_invalid_cluster"
                break

            similarities = _gradient_similarity_matrix(
                gradients, eigenpairs
            )
            new_labels = np.argmax(similarities, axis=1)
            if len(np.unique(new_labels)) < int(n_clusters):
                termination_reason = "empty_cluster_after_assignment"
                break

            if np.array_equal(new_labels, labels):
                converged = True
                labels = new_labels
                termination_reason = "fixed_partition"
                break

            partition_key = new_labels.tobytes()
            if partition_key in seen_partitions:
                cycle_detected = True
                labels = new_labels
                termination_reason = "partition_cycle"
                break

            seen_partitions.add(partition_key)
            labels = new_labels

        if len(np.unique(labels)) < int(n_clusters):
            continue

        eigenpairs, covariance_matrices = _cluster_gradient_eigenpairs(
            gradients,
            labels,
            n_clusters,
            gradient_norm_tolerance=gradient_norm_tolerance,
        )
        if eigenpairs is None:
            continue

        similarities = _gradient_similarity_matrix(
            gradients, eigenpairs
        )
        objective = float(
            np.sum(similarities[np.arange(n_samples), labels])
        )
        candidate = {
            "labels": labels.copy(),
            "cluster_eigens": eigenpairs,
            "covariance_matrices": covariance_matrices,
            "similarities": similarities,
            "objective": objective,
            "converged": bool(converged),
            "cycle_detected": bool(cycle_detected),
            "termination_reason": termination_reason,
            "iterations": int(iteration + 1),
            "initialization_id": int(init_id),
            "initialization_seed": int(seed),
        }
        candidates.append(candidate)

    if not candidates:
        return None

    eligible = (
        [candidate for candidate in candidates if candidate["converged"]]
        if require_convergence
        else candidates
    )
    if not eligible:
        return {
            "status": "no_converged_initialization",
            "all_candidates": candidates,
        }

    best = max(eligible, key=lambda candidate: candidate["objective"])
    best = dict(best)
    best["status"] = "success"
    best["successful_initializations"] = int(len(candidates))
    best["eligible_initializations"] = int(len(eligible))
    return best


def _resolve_gradient_regime_target_transform(values, mode="log") -> str:
    """Resolve the requested monotone target transform.

    ``auto`` uses log for strictly positive data, log1p for nonnegative data,
    and raw values otherwise.  A strictly increasing scalar transform preserves
    exact normalized-gradient directions, while often making the GPR surrogate
    easier to fit.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    requested = str(mode).strip().lower()
    if requested != "auto":
        return requested
    if np.all(values > 0.0):
        return "log"
    if np.all(values >= 0.0):
        return "log1p"
    return "raw"


def _transform_gradient_regime_target(values, mode="log"):
    """Transform target values for gradient-regime GPR fitting."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    mode = _resolve_gradient_regime_target_transform(values, mode)

    if mode == "raw":
        return values
    if mode == "log":
        if np.any(values <= 0.0):
            raise ValueError(
                "Gradient-regime target transform 'log' requires strictly positive outputs."
            )
        return np.log(values)
    if mode == "log1p":
        if np.any(values < 0.0):
            raise ValueError(
                "Gradient-regime target transform 'log1p' requires nonnegative outputs."
            )
        return np.log1p(values)
    raise ValueError(
        "Gradient-regime GPR target transform must be 'auto', 'log', "
        "'log1p', or 'raw'."
    )


def _normalize_gradient_regime_options(user_options):
    """Normalize the current semantic gradient-regime configuration."""
    options = dict(user_options or {})
    options["surrogate"] = dict(options.get("surrogate", {}))
    options["clustering"] = dict(options.get("clustering", {}))
    options["local_workflow"] = dict(options.get("local_workflow", {}))
    return options


def _intercluster_direction_evidence(cluster_eigens, max_cosine=0.90):
    """Compare leading COLUMN eigenvectors in the common orthonormal log-Pi basis.

    Absolute inner products compare one-dimensional subspaces, so reversing an
    eigenvector sign does not create a different regime. This is a candidate
    geometry diagnostic. The caller uses it only for the auto K=2
    redundancy check, not as a general K-cluster admissibility gate.
    """
    enabled = max_cosine is not None
    vectors = []
    invalid_reason = None
    for _, eigenvectors in cluster_eigens:
        matrix = np.asarray(eigenvectors, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            invalid_reason = "missing_leading_eigenvector"
            break
        leading = matrix[:, 0]
        norm = float(np.linalg.norm(leading))
        if not np.all(np.isfinite(leading)) or not np.isfinite(norm) or norm <= 1e-15:
            invalid_reason = "invalid_leading_eigenvector"
            break
        if vectors and leading.size != vectors[0].size:
            invalid_reason = "inconsistent_eigenvector_dimensions"
            break
        vectors.append(leading / norm)
    pairs = []
    if invalid_reason is None:
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cosine = float(np.clip(abs(np.dot(vectors[i], vectors[j])), 0.0, 1.0))
                pairs.append({
                    "cluster_i": i, "cluster_j": j,
                    "absolute_cosine": cosine,
                    "angle_degrees": float(np.degrees(np.arccos(cosine))),
                })
    max_pairwise = max((pair["absolute_cosine"] for pair in pairs), default=None)
    minimum_angle = min((pair["angle_degrees"] for pair in pairs), default=None)
    valid = invalid_reason is None and len(vectors) >= 2
    passed = not enabled or (valid and max_pairwise < float(max_cosine))
    return {
        "enabled": enabled,
        "maximum_allowed_absolute_cosine": max_cosine,
        "coordinate_system": "common Schmidt-orthonormalized log-Pi screening basis",
        "max_pairwise_absolute_cosine": max_pairwise,
        "minimum_pairwise_angle_degrees": minimum_angle,
        "passes": bool(passed),
        "invalid_reason": invalid_reason,
        "pairs": pairs,
    }


def _reject_removed_regime_controls(hp=None, cluster_options=None, plan=None):
    """Fail explicitly on retired regime switches instead of silently ignoring them."""
    hp = dict(hp or {})
    removed_flat = {
        "regime_selection_mode", "regime_max_clusters", "regime_prefer_single",
        "prefer_single_regime", "regime_policy",
        "allow_single_fallback_after_replan",
        "auto_priority", "regime_single_leading_fraction_threshold",
        "single_regime_leading_fraction_threshold",
    }
    removed_nested = {"enabled", "selection_mode", "require_regime_criteria", "prefer_single_regime",
                      "auto_priority", "single_regime_leading_fraction_threshold"}
    paths = ["manual_hyperparams." + key for key in sorted(removed_flat & hp.keys())]
    nested = ((hp.get("gradient_regime_options") or {}).get("clustering") or {})
    for label, values in [("gradient_regime_options.clustering", nested),
                          ("clustering", cluster_options or {})]:
        paths.extend(label + "." + key for key in sorted(removed_nested & values.keys()))
    supplied = hp.get("workflow_plan") or {}
    for label, payload in [("workflow_plan", supplied), ("plan", plan or {})]:
        if isinstance(payload, dict):
            values = payload.get("regime_detection") or {}
            paths.extend(label + ".regime_detection." + key
                         for key in sorted(removed_nested & values.keys()))
    if paths:
        raise ValueError(
            "Unsupported/retired regime controls in v7: " + ", ".join(sorted(set(paths)))
            + ". Use workflow_template and an explicit candidate_cluster_counts list. "
            "Base sample-count, fraction and eigenvalue-ratio gates are mandatory; "
            "the coarsest admissible partition is always selected."
        )


def _validate_regime_configuration(cluster_options, workflow_template):
    """Validate policy fields once while preserving numerical clustering options."""
    from .schemas import RegimeDetectionPlan
    from .information import _pydantic_dump
    _reject_removed_regime_controls(cluster_options=cluster_options)
    result = dict(cluster_options)
    field_names = getattr(RegimeDetectionPlan, "model_fields", {})
    policy = RegimeDetectionPlan(**{key: value for key, value in result.items() if key in field_names})
    result.update(_pydantic_dump(policy))
    counts = list(result["candidate_cluster_counts"])
    if not counts or any(value < 2 for value in counts):
        raise ValueError("candidate_cluster_counts must be non-empty and contain only K >= 2.")
    result["candidate_cluster_counts"] = sorted(set(counts))
    if (workflow_template == "auto_mechanism_adaptive"
            and result.get("n_clusters") is None
            and 2 not in result["candidate_cluster_counts"]):
        raise ValueError("Automatic regime assessment requires K=2 in candidate_cluster_counts.")
    return result


def _select_regime_from_evidence(
    single_baseline,
    candidates,
    workflow_template,
    fixed_k=None,
    direction_cosine_threshold=0.90,
    require_convergence=True,
):
    """Choose from complete candidate evidence; global E never selects a route.

    For unfixed auto, an admissible partition is necessary but not sufficient
    for multiple-regime treatment.  A converged, valid K=2 partition must also
    have distinct FINAL cluster covariance principal directions (|cos| < tau).
    This K=2 support does not require K=2 itself to pass the base gates.
    Prescribed multiple and fixed K retain external authority and base gates.
    """
    if workflow_template not in {
        "single_regime_adaptive", "multi_regime_adaptive", "auto_mechanism_adaptive"
    }:
        raise ValueError("Unknown regime workflow template: " + str(workflow_template))
    if direction_cosine_threshold is None or not 0.0 < float(direction_cosine_threshold) <= 1.0:
        raise ValueError("K=2 direction threshold must be in (0, 1]; it cannot be disabled.")
    ordered = sorted(candidates, key=lambda item: int(item["k"]))
    k2 = next((candidate for candidate in ordered if int(candidate["k"]) == 2), None)
    direction = (k2 or {}).get("direction_separation") or {}
    cosine = direction.get("max_pairwise_absolute_cosine")
    k2_valid_direction = bool(
        k2 is not None and direction.get("invalid_reason") is None
        and cosine is not None and np.isfinite(cosine) and 0.0 <= cosine <= 1.0
    )
    k2_converged = bool((k2 or {}).get("converged", False))
    k2_support_available = bool(k2_valid_direction and k2_converged)
    distinct = bool(k2_support_available and cosine < float(direction_cosine_threshold))

    def admissible(candidate):
        return bool(candidate.get("meets_regime_criteria", False)) and (
            bool(candidate.get("converged", False)) or not require_convergence
        )

    admitted = [candidate for candidate in ordered if admissible(candidate)]
    evidence = {
        "global_leading_fraction_diagnostic": single_baseline.get("leading_fraction"),
        "global_spectrum_role": "diagnostic_only_no_decision_threshold",
        "admissible_cluster_counts": [int(candidate["k"]) for candidate in admitted],
        "k2_available": k2 is not None,
        "k2_actually_converged": k2_converged,
        "k2_base_criteria_passed": bool((k2 or {}).get("meets_regime_criteria", False)),
        "k2_direction_valid": k2_valid_direction,
        "k2_absolute_cosine": cosine,
        "k2_direction_threshold": float(direction_cosine_threshold),
        "k2_directional_support_available": k2_support_available,
        "k2_supports_multiple_regimes": distinct,
        "k2_base_gates_required_for_directional_support": False,
        "all_requested_K_evaluated_before_selection": True,
    }

    def outcome(status, basis, selected=None, action="not_applicable"):
        return {
            "status": status,
            "best_k": 1 if status == "global_analysis_selected" else
                      int(selected["k"]) if selected is not None else None,
            "selection_basis": basis,
            "selected_candidate": selected,
            "k2_direction_action": action,
            "decision_evidence": evidence,
        }

    if workflow_template == "single_regime_adaptive":
        return outcome("global_analysis_selected", "prescribed_single")
    if fixed_k is not None:
        fixed = next((candidate for candidate in ordered if int(candidate["k"]) == int(fixed_k)), None)
        if fixed is not None and admissible(fixed):
            return outcome("success", "prescribed_fixed_K", fixed, "diagnostic_only_fixed_K")
        return outcome("no_admissible_partition", "prescribed_fixed_K_unresolved",
                       action="diagnostic_only_fixed_K")
    if workflow_template == "multi_regime_adaptive":
        selected = admitted[0] if admitted else None
        return outcome("success" if selected is not None else "no_admissible_partition",
                       "prescribed_multiple_coarsest_admissible", selected,
                       "diagnostic_only_prescribed_multiple")
    if not ordered:
        return outcome("regime_evidence_unavailable", "auto_no_usable_partition_evidence",
                       action="unresolved_no_numerical_partition")
    if not admitted:
        return outcome("global_analysis_selected", "auto_no_admissible_partition",
                       action="no_admissible_partition_global")
    if not k2_support_available:
        return outcome("regime_direction_unresolved", "auto_K2_directional_support_unavailable",
                       action="unresolved_K2_missing_invalid_or_not_converged")
    if not distinct:
        return outcome("global_analysis_selected", "auto_K2_directions_aligned",
                       action="K2_absolute_cosine_at_or_above_threshold_global")
    return outcome("success", "auto_K2_supported_coarsest_admissible", admitted[0],
                   action="K2_distinct_directions_support_multiple")


def detect_gradient_regimes(
    X_physical,
    dimensionless_output,
    basis_matrices,
    max_clusters=4,
    min_cluster_fraction=0.05,
    eigenvalue_ratio_threshold=50.0,
    random_state=GLOBAL_RANDOM_SEED,
    max_train_samples=None,
    gradient_regime_options=None,
    *,
    workflow_template="auto_mechanism_adaptive",
    min_cluster_samples=60,
    max_intercluster_cosine=0.90,
):
    """Detect candidate physical regimes from GPR gradients in log-Pi space.

    ``dimensionless_output`` is assumed to be the already dimensionless response
    Y.  The function only detects regimes.  The dominant Pi group in each
    accepted regime is subsequently found by an independent mutual-information
    optimization. ``workflow_template`` preserves explicit single/multiple
    authority. Direct calls default to the auto template. The per-cluster
    sample floor counts informative hard assignments, before optional
    boundary-point filtering, rather than the surrogate training subset.
    Global and every requested K are evaluated before route selection.
    Global E is diagnostic only. Auto requires K=2 directional support
    whenever at least one candidate partition passes the base gates.
    Explicit multiple and fixed K use mandatory base gates and never
    return single merely because their K=2 directions are aligned.
    """
    from .information import _deep_update_dict
    if workflow_template not in {
        "auto_mechanism_adaptive", "multi_regime_adaptive"
    }:
        raise ValueError(
            "detect_gradient_regimes requires the auto_mechanism_adaptive "
            "or multi_regime_adaptive workflow template."
        )
    is_auto = workflow_template == "auto_mechanism_adaptive"
    user_options = _normalize_gradient_regime_options(
        gradient_regime_options
    )
    user_surrogate_options = (
        user_options.get("surrogate", {})
        if isinstance(user_options, dict)
        else {}
    )
    user_cluster_options = (
        user_options.get("clustering", {})
        if isinstance(user_options, dict)
        else {}
    )
    _reject_removed_regime_controls(cluster_options=user_cluster_options)
    options = _deep_update_dict(
        GRADIENT_REGIME_DEFAULT_OPTIONS, user_options
    )
    surrogate_options = options["surrogate"]
    cluster_options = options["clustering"]

    # Scalar arguments supply defaults unless nested options override them.
    if "random_state" not in user_surrogate_options:
        surrogate_options["random_state"] = random_state
    if (
        max_train_samples is not None
        and "max_train_samples" not in user_surrogate_options
    ):
        surrogate_options["max_train_samples"] = max_train_samples
    if "random_state" not in user_cluster_options:
        cluster_options["random_state"] = random_state
    if "min_cluster_fraction" not in user_cluster_options:
        cluster_options["min_cluster_fraction"] = min_cluster_fraction
    if "min_cluster_samples" not in user_cluster_options:
        cluster_options["min_cluster_samples"] = min_cluster_samples
    if "max_intercluster_cosine" not in user_cluster_options:
        cluster_options["max_intercluster_cosine"] = max_intercluster_cosine
    if "eigenvalue_ratio_threshold" not in user_cluster_options:
        cluster_options[
            "eigenvalue_ratio_threshold"
        ] = eigenvalue_ratio_threshold
    if (
        "candidate_cluster_counts" not in user_cluster_options
        and cluster_options.get("n_clusters") is None
    ):
        cluster_options["candidate_cluster_counts"] = list(
            range(2, int(max_clusters) + 1)
        )

    cluster_options = _validate_regime_configuration(cluster_options, workflow_template)
    options["clustering"] = cluster_options
    minimum_cluster_samples = cluster_options["min_cluster_samples"]
    direction_cosine_threshold = cluster_options["max_intercluster_cosine"]

    X = np.asarray(X_physical, dtype=float)
    y = np.asarray(dimensionless_output, dtype=float).reshape(-1, 1)
    B = _schmidt_orthonormalize_basis_rows(
        np.asarray(basis_matrices, dtype=float)
    )

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        return {
            "status": "invalid_data_shape",
            "message": (
                "X_physical must have shape (N, m) and dimensionless_output "
                "must contain N values."
            ),
        }
    if B.ndim != 2 or B.shape[0] < 2:
        return {
            "status": "not_applicable",
            "message": (
                "At least two independent base-Pi coordinates are required "
                "for multi-regime gradient clustering."
            ),
        }
    if B.shape[1] != X.shape[1]:
        return {
            "status": "invalid_basis_shape",
            "message": (
                "The Pi-basis column count must equal the number of physical "
                "input variables."
            ),
        }
    
    ### Previous 
    # if np.any(X <= 0):
    #     return {
    #         "status": "invalid_physical_data",
    #         "message": (
    #             "Power-law Pi construction and log(Pi) require strictly "
    #             "positive physical inputs."
    #         ),
    #     }

    # # log(prod q_i^e_i) = sum_i e_i log(q_i), which avoids overflow from
    # # explicitly constructing the base Pi groups.
    # Z = np.log(X) @ B.T
    # finite_mask = (
    #     np.all(np.isfinite(Z), axis=1)
    #     & np.isfinite(y[:, 0])
    # )
    # valid_indices = np.flatnonzero(finite_mask)
    # Z_valid = Z[finite_mask]
    # y_valid = y[finite_mask]

    ### new version
    # Gradient-regime detection is performed in log(Pi) coordinates, so only
    # samples with finite, strictly positive physical inputs are admissible.
    # Boundary samples such as y=0 are excluded from the gradient analysis
    # rather than causing the entire regime-detection stage to fail.
    finite_input_mask = np.all(np.isfinite(X), axis=1)
    finite_target_mask = np.isfinite(y[:, 0])
    positive_input_mask = np.all(X > 0.0, axis=1)
    positive_target_mask = y[:, 0] > 0.0

    physical_valid_mask = (
        finite_input_mask
        & finite_target_mask
        & positive_input_mask
        & positive_target_mask
    )

    prevalid_indices = np.flatnonzero(physical_valid_mask)

    # log(prod q_i^e_i) = sum_i e_i log(q_i)
    X_prevalid = X[physical_valid_mask]
    y_prevalid = y[physical_valid_mask]

    Z_prevalid = np.log(X_prevalid) @ B.T

    # Retain this secondary finite check for numerical robustness.
    finite_Z_mask = np.all(np.isfinite(Z_prevalid), axis=1)

    valid_indices = prevalid_indices[finite_Z_mask]
    Z_valid = Z_prevalid[finite_Z_mask]
    y_valid = y_prevalid[finite_Z_mask]


    if Z_valid.shape[0] < 30:
        return {
            "status": "insufficient_data",
            "n_samples": int(Z_valid.shape[0]),
        }

    target_transform_requested = str(
        surrogate_options.get("regime_target_transform", "log")
    )
    target_transform_mode = _resolve_gradient_regime_target_transform(
        y_valid,
        target_transform_requested,
    )
    try:
        y_regime = _transform_gradient_regime_target(
            y_valid,
            mode=target_transform_mode,
        )
    except Exception:
        return {
            "status": "invalid_target_transform",
            "message": traceback.format_exc(),
        }

    try:
        gradients, fitted_surrogate = _fit_gpr_gradients(
            Z_valid,
            y_regime,
            random_state=int(
                surrogate_options.get("random_state", random_state)
            ),
            max_train_samples=surrogate_options.get("max_train_samples"),
            gradient_gpr_options=surrogate_options,
        )
    except Exception:
        return {
            "status": "gpr_failed",
            "message": traceback.format_exc(),
        }

    gradient_norm_tolerance = float(
        cluster_options.get("gradient_norm_tolerance", 1e-12)
    )
    gradient_norms = np.linalg.norm(gradients, axis=1)
    informative_mask = gradient_norms > gradient_norm_tolerance
    excluded_directionless_indices = valid_indices[~informative_mask]
    informative_indices = valid_indices[informative_mask]
    informative_feature_coordinates = Z_valid[informative_mask]
    informative_gradients = gradients[informative_mask]

    if informative_gradients.shape[0] < 30:
        return {
            "status": "insufficient_informative_gradients",
            "surrogate": fitted_surrogate,
            "gradient_norm_tolerance": gradient_norm_tolerance,
            "informative_samples": int(informative_gradients.shape[0]),
            "excluded_directionless_samples": int(
                excluded_directionless_indices.size
            ),
        }

    # Collect the global spectrum first; it never stops evidence collection.
    single_labels = np.zeros(informative_gradients.shape[0], dtype=int)
    single_eigenpairs, single_covariances = _cluster_gradient_eigenpairs(
        informative_gradients, single_labels, 1, gradient_norm_tolerance
    )
    single_eigenvalues = np.asarray(single_eigenpairs[0][0], dtype=float)
    single_eigenvectors = np.asarray(single_eigenpairs[0][1], dtype=float)

    spectral_sum = float(np.sum(single_eigenvalues))
    single_leading_fraction = (
        float(single_eigenvalues[0] / max(spectral_sum, 1e-15))
        if single_eigenvalues.size
        else 0.0
    )
    single_baseline = {
        "eigenvalues": single_eigenvalues.tolist(),
        "eigenvectors_in_log_base_pi": single_eigenvectors.tolist(),
        "leading_fraction": single_leading_fraction,
        "criterion": "lambda_1 / sum(lambda)",
        "role": "diagnostic_only_no_decision_threshold",
    }
    acceptance_criteria = {
        "minimum_cluster_samples": minimum_cluster_samples,
        "minimum_cluster_fraction": float(cluster_options["min_cluster_fraction"]),
        "minimum_eigenvalue_ratio": float(cluster_options["eigenvalue_ratio_threshold"]),
        "base_gates_mandatory": True,
        "require_convergence": bool(cluster_options["require_convergence"]),
        "maximum_intercluster_cosine": direction_cosine_threshold,
        "direction_decision_scope": "auto_unfixed_K2_support_for_any_admissible_K",
        "global_spectrum_role": "diagnostic_only_no_decision_threshold",
        "k2_base_gates_required_for_directional_support": False,
    }
    fixed_k = cluster_options.get("n_clusters")
    candidate_counts = ([int(fixed_k)] if fixed_k is not None
                        else list(cluster_options["candidate_cluster_counts"]))

    candidates = []
    clustering_failures = []
    for k in candidate_counts:
        if informative_gradients.shape[0] < k:
            clustering_failures.append({"k": int(k), "status": "insufficient_samples_for_K"})
            continue

        clustering_result = _cluster_by_gradient_similarity(
            informative_gradients,
            k,
            random_state=int(
                cluster_options.get("random_state", random_state)
            ),
            max_iter=int(cluster_options.get("max_iter", 200)),
            n_init=int(cluster_options.get("n_init", 30)),
            initialization=str(
                cluster_options.get(
                    "initialization", "random_eigenpairs"
                )
            ),
            max_initialization_attempts=int(
                cluster_options.get(
                    "max_initialization_attempts", 100
                )
            ),
            require_convergence=bool(
                cluster_options.get("require_convergence", True)
            ),
            gradient_norm_tolerance=gradient_norm_tolerance,
        )
        if (
            clustering_result is None
            or clustering_result.get("status") != "success"
        ):
            clustering_failures.append({
                "k": int(k),
                "status": (
                    None
                    if clustering_result is None
                    else clustering_result.get("status")
                ),
            })
            continue

        labels = np.asarray(
            clustering_result["labels"], dtype=int
        )
        cluster_eigens = clustering_result["cluster_eigens"]
        similarities = np.asarray(
            clustering_result["similarities"], dtype=float
        )
        fractions = [
            float(np.mean(labels == cluster_id))
            for cluster_id in range(k)
        ]

        sample_counts = [
            int(np.count_nonzero(labels == cluster_id))
            for cluster_id in range(k)
        ]
        ratios = []
        cluster_summary = []
        for cluster_id, (eigenvalues, eigenvectors) in enumerate(
            cluster_eigens
        ):
            eigenvalues = np.asarray(eigenvalues, dtype=float)
            eigenvectors = np.asarray(eigenvectors, dtype=float)
            ratio = float(
                eigenvalues[0] / max(eigenvalues[-1], 1e-15)
            )
            ratios.append(ratio)
            physical_directions = (eigenvectors.T @ B).tolist()
            cluster_summary.append({
                "cluster": int(cluster_id),
                "regime_id": int(cluster_id),
                "fraction": fractions[cluster_id],
                "n_samples": sample_counts[cluster_id],
                "eigenvalues": eigenvalues.tolist(),
                "eigenvalue_ratio_max_to_min": ratio,
                "eigenvectors_in_log_base_pi": eigenvectors.tolist(),
                "dimensionless_exponent_directions_in_physical_variables":
                    physical_directions,
                # Diagnostic only: local MI, not this direction, determines
                # the reported dominant Pi group.
                "leading_gradient_direction_in_physical_variables":
                    physical_directions[0],
            })

        # Direction evidence is computed for K=2 only. K>=3 is never
        # rejected because two of its finer subdivisions are aligned.
        direction_evidence = (
            _intercluster_direction_evidence(cluster_eigens, direction_cosine_threshold)
            if int(k) == 2 else {}
        )
        min_samples = int(min(sample_counts))
        sample_count_pass = min_samples >= minimum_cluster_samples
        min_fraction = float(min(fractions))
        min_ratio = float(min(ratios))
        fraction_pass = (
            min_fraction
            >= float(
                cluster_options.get(
                    "min_cluster_fraction", 0.05
                )
            )
        )
        ratio_pass = (
            min_ratio
            >= float(
                cluster_options.get(
                    "eigenvalue_ratio_threshold", 50.0
                )
            )
        )

        base_criteria_met = bool(sample_count_pass and fraction_pass and ratio_pass)
        sorted_similarity = np.sort(similarities, axis=1)
        confidence = (
            sorted_similarity[:, -1] - sorted_similarity[:, -2]
        ) / np.maximum(
            np.abs(sorted_similarity[:, -1]), 1e-15
        )

        candidates.append({
            "k": int(k),
            "labels": labels,
            "assignment_confidence": confidence,
            "fractions": fractions,
            "sample_counts": sample_counts,
            "minimum_cluster_samples": min_samples,
            "passes_sample_count_criterion": bool(sample_count_pass),
            # Retain these evidence keys for existing readers; they no
            # longer determine meets_regime_criteria. K>=3 records None.
            "passes_direction_separation_criterion": direction_evidence.get("passes"),
            "max_intercluster_abs_cosine": direction_evidence.get("max_pairwise_absolute_cosine"),
            "minimum_intercluster_angle_degrees": direction_evidence.get("minimum_pairwise_angle_degrees"),
            "direction_separation": direction_evidence,
            "k2_direction_action": "evidence_collected" if int(k) == 2 else "not_applicable",
            "minimum_cluster_fraction": min_fraction,
            "minimum_eigenvalue_ratio": min_ratio,
            "passes_fraction_criterion": bool(fraction_pass),
            "passes_eigenvalue_ratio_criterion": bool(ratio_pass),
            "meets_regime_criteria": base_criteria_met,
            "clustering_objective": float(
                clustering_result["objective"]
            ),
            "converged": bool(
                clustering_result["converged"]
            ),
            "iterations": int(
                clustering_result["iterations"]
            ),
            "termination_reason": clustering_result[
                "termination_reason"
            ],
            "clusters": cluster_summary,
        })

    common = {
        "method": GRADIENT_REGIME_IMPLEMENTATION,
        "surrogate": fitted_surrogate,
        "gpr": fitted_surrogate,
        "surrogate_mean_mode": fitted_surrogate.get("mean_mode", "constant"),
        "target_transform": target_transform_mode,
        "target_transform_requested": target_transform_requested,
        "target_assumption": f"Y is dimensionless; regime target transform is {target_transform_mode}.",
        "feature_coordinate_system": "log of the Schmidt-orthonormalized Buckingham-Pi screening basis",
        "feature_coordinates": informative_feature_coordinates.tolist(),
        "surrogate_gradients": informative_gradients.tolist(),
        "schmidt_basis": B.tolist(),
        "valid_indices": informative_indices.astype(int).tolist(),
        "excluded_directionless_indices": excluded_directionless_indices.astype(int).tolist(),
        "excluded_directionless_samples": int(excluded_directionless_indices.size),
        "gradient_norm_tolerance": gradient_norm_tolerance,
        "single_regime_baseline": single_baseline,
        "acceptance_criteria": acceptance_criteria,
        "requested_cluster_counts": candidate_counts,
        "clustering_failures": clustering_failures,
        # Display-only trace: preserve evaluated partitions, including rejected K.
        "candidate_partitions": {str(candidate["k"]): {
            "labels": np.asarray(candidate["labels"], dtype=int).tolist(),
            "assignment_confidence": np.asarray(candidate["assignment_confidence"], dtype=float).tolist(),
        } for candidate in candidates},
        "candidate_summary": [],
    }
    if not candidates:
        return {**common, "status": "clustering_failed",
                "message": "No requested K produced a usable partition; regime assessment is unresolved."}

    decision = _select_regime_from_evidence(
        single_baseline, candidates, workflow_template,
        fixed_k=fixed_k,
        direction_cosine_threshold=direction_cosine_threshold,
        require_convergence=bool(cluster_options["require_convergence"]),
    )
    for candidate in candidates:
        if int(candidate["k"]) == 2:
            candidate["k2_direction_action"] = decision["k2_direction_action"]
    common["candidate_summary"] = [
        {key: value for key, value in candidate.items()
         if key not in ("labels", "assignment_confidence")}
        for candidate in candidates
    ]
    selected = decision["selected_candidate"]
    public_decision = {key: value for key, value in decision.items()
                       if key != "selected_candidate"}
    if selected is None:
        return {**common, **public_decision,
                "message": ("Use the global MI/similarity workflow; basis: " + decision["selection_basis"]
                            if decision["status"] == "global_analysis_selected" else
                            "Regime assessment remains unresolved; basis: " + decision["selection_basis"])}
    return {
        **common, **public_decision,
        "meets_regime_criteria": bool(selected["meets_regime_criteria"]),
        "cluster_summary": selected["clusters"],
        "labels": np.asarray(selected["labels"], dtype=int).tolist(),
        "assignment_confidence": np.asarray(selected["assignment_confidence"], dtype=float).tolist(),
        "low_confidence_fraction_at_0.05": float(np.mean(
            np.asarray(selected["assignment_confidence"]) < 0.05)),
        "implementation_notes": {
            "gradient_clustering_reference": "Zhang et al., CMAME 420 (2024), gradient-similarity clustering equations.",
            "agent_extension": "Each accepted regime is analyzed independently by mutual-information/CMA-ES optimization.",
            "cross_k_selection": "The objective compares starts within a fixed K; select the coarsest base-admissible K.",
            "evidence_collection": "Global spectrum and every requested K precede route selection.",
        },
        "next_step": "Optimize one input Pi group independently inside each accepted regime, without output scaling.",
    }
