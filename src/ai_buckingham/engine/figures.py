"""Publication-style notebook visualizations."""
from __future__ import annotations
from .shared import *


def _regime_plot_original_indices(detection: dict, frame: pd.DataFrame) -> np.ndarray:
    """Return CSV row positions aligned with labels and feature diagnostics."""
    labels = np.asarray(detection.get("labels", []), dtype=int)
    original = np.asarray(detection.get("original_csv_indices", []), dtype=int)
    if labels.size == 0:
        raise ValueError("No selected regime labels are available for plotting.")
    if original.size != labels.size:
        raise ValueError(
            "regime_detection.original_csv_indices must be present and have "
            "the same length as labels.  This mapping is created by "
            "execute_compiled_workflow after cleaning the CSV data."
        )
    if np.any(original < 0) or np.any(original >= frame.shape[0]):
        raise ValueError("Regime plot row positions fall outside the CSV frame.")
    return original


def _regime_plot_basis_indices(indices, dimension: int) -> tuple[int, int]:
    """Normalize externally supplied one-based Pi indices."""
    raw = list(indices or [1, 2])
    if len(raw) != 2:
        raise ValueError("regime_plot_feature_pi_indices must contain two indices.")
    selected = tuple(int(value) - 1 for value in raw)
    if selected[0] == selected[1]:
        raise ValueError("The two feature-plot Pi indices must be different.")
    if any(index < 0 or index >= int(dimension) for index in selected):
        raise ValueError(
            f"Requested Pi indices {raw} are outside the available range 1..{dimension}."
        )
    return selected


def _regime_plot_original_basis_coordinates_and_gradients(
    state: dict,
    detection: dict,
    frame_subset: pd.DataFrame,
    screening_gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map screening-basis gradients to the original Buckingham-Pi basis.

    The detector uses a Schmidt-orthonormalized basis ``B_s``.  If the original
    basis is ``B_0`` and ``B_s = A B_0``, then ``z_s = z_0 A^T`` and the row-wise
    gradients transform as ``g_0 = g_s A``.
    """
    input_vars = list(state.get("input_vars") or [])
    B0 = np.asarray(state.get("basis_pi_matrices") or [], dtype=float)
    Bs = np.asarray(detection.get("schmidt_basis") or [], dtype=float)
    if B0.ndim != 2 or Bs.ndim != 2 or B0.shape != Bs.shape:
        raise ValueError(
            "Original and screening Buckingham-Pi bases must be available with matching shapes."
        )
    X = frame_subset[input_vars].to_numpy(dtype=float)
    if np.any(X <= 0.0):
        raise ValueError("Original-basis log(Pi) coordinates require positive inputs.")

    original_coordinates = np.log(X) @ B0.T
    basis_map = Bs @ np.linalg.pinv(B0)
    residual = float(np.linalg.norm(Bs - basis_map @ B0, ord=np.inf))
    original_gradients = screening_gradients @ basis_map
    return original_coordinates, original_gradients, residual


def _select_regime_feature_projection(
    state: dict,
    detection: dict,
    frame_subset: pd.DataFrame,
    hp: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Choose a two-dimensional, non-PCA feature representation.

    Modes
    -----
    auto:
        Use the screening Pi basis when there are exactly two Pi coordinates;
        otherwise use the two leading eigendirections of the global normalized-
        gradient covariance.
    leading_gradient:
        Project onto the first two global gradient-covariance eigendirections.
    screening_basis:
        Plot two externally selected coordinates from the Schmidt basis used by
        the GPR and clustering algorithm.
    original_basis / basis:
        Plot two externally selected coordinates from the original basis made by
        ``calc_basis_tool``.  Gradients are transformed analytically; no PCA is used.
    """
    Z_screen = np.asarray(detection.get("feature_coordinates", []), dtype=float)
    G_screen = np.asarray(detection.get("surrogate_gradients", []), dtype=float)
    labels = np.asarray(detection.get("labels", []), dtype=int)
    if (
        Z_screen.ndim != 2
        or G_screen.ndim != 2
        or Z_screen.shape != G_screen.shape
        or Z_screen.shape[0] != labels.size
        or Z_screen.shape[1] < 2
    ):
        raise ValueError(
            "Regime feature coordinates, gradients, and labels are missing or misaligned."
        )

    requested_mode = str(hp.get("regime_plot_feature_mode", "auto")).strip().lower()
    mode_aliases = {
        "gradient": "leading_gradient",
        "active": "leading_gradient",
        "schmidt": "screening_basis",
        "basis": "original_basis",
    }
    requested_mode = mode_aliases.get(requested_mode, requested_mode)
    if requested_mode == "auto":
        mode = "screening_basis" if Z_screen.shape[1] == 2 else "leading_gradient"
    else:
        mode = requested_mode

    if mode == "leading_gradient":
        norms = np.linalg.norm(G_screen, axis=1)
        good = norms > 1.0e-15
        if np.sum(good) < 2:
            raise ValueError("Too few nonzero gradients for leading-direction projection.")
        normalized = G_screen[good] / norms[good, None]
        covariance = normalized.T @ normalized / normalized.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        projection = eigenvectors[:, order[:2]]
        coordinates = Z_screen @ projection
        gradients = G_screen @ projection
        labels_text = [r"$\xi_1$", r"$\xi_2$"]
        description = (
            "leading global normalized-gradient eigendirections; "
            f"lambda=[{eigenvalues[order[0]]:.4g}, {eigenvalues[order[1]]:.4g}]"
        )
    elif mode == "screening_basis":
        i, j = _regime_plot_basis_indices(
            hp.get("regime_plot_feature_pi_indices", [1, 2]),
            Z_screen.shape[1],
        )
        coordinates = Z_screen[:, [i, j]]
        gradients = G_screen[:, [i, j]]
        labels_text = [rf"$\log(\Pi_{{{i + 1}}})$", rf"$\log(\Pi_{{{j + 1}}})$"]
        description = "selected Schmidt-screening Pi coordinates"
    elif mode == "original_basis":
        Z_original, G_original, residual = (
            _regime_plot_original_basis_coordinates_and_gradients(
                state, detection, frame_subset, G_screen
            )
        )
        i, j = _regime_plot_basis_indices(
            hp.get("regime_plot_feature_pi_indices", [1, 2]),
            Z_original.shape[1],
        )
        coordinates = Z_original[:, [i, j]]
        gradients = G_original[:, [i, j]]
        labels_text = [rf"$\log(\Pi_{{{i + 1}}})$", rf"$\log(\Pi_{{{j + 1}}})$"]
        description = (
            "selected original Buckingham-Pi coordinates; "
            f"basis-map residual={residual:.2e}"
        )
    else:
        raise ValueError(
            "regime_plot_feature_mode must be 'auto', 'leading_gradient', "
            "'screening_basis', or 'original_basis'."
        )

    external_labels = hp.get("regime_plot_feature_labels")
    if external_labels is not None:
        external_labels = list(external_labels)
        if len(external_labels) != 2:
            raise ValueError("regime_plot_feature_labels must contain two labels.")
        labels_text = [str(external_labels[0]), str(external_labels[1])]
    return coordinates, gradients, labels_text, description


def _normalize_regime_axis_spec(spec) -> Optional[dict]:
    """Normalize a physical-panel axis specification."""
    if spec is None:
        return None
    if isinstance(spec, str):
        match = re.fullmatch(r"Pi[_ ]?(\d+)", spec.strip(), flags=re.IGNORECASE)
        if match:
            return {"source": "basis_pi", "index": int(match.group(1))}
        return {"source": "column", "name": spec}
    if not isinstance(spec, dict):
        raise TypeError("A regime physical-axis specification must be a string or dict.")
    return dict(spec)


def _regime_plot_original_log_pi(state: dict, frame_subset: pd.DataFrame) -> np.ndarray:
    input_vars = list(state.get("input_vars") or [])
    basis = np.asarray(state.get("basis_pi_matrices") or [], dtype=float)
    X = frame_subset[input_vars].to_numpy(dtype=float)
    if basis.ndim != 2 or basis.shape[1] != X.shape[1]:
        raise ValueError("The original Buckingham-Pi basis is unavailable or has the wrong shape.")
    if np.any(X <= 0.0):
        raise ValueError("Buckingham-Pi plot coordinates require positive inputs.")
    return np.log(X) @ basis.T


def _resolve_regime_physical_axis(
    spec,
    *,
    state: dict,
    detection: dict,
    frame_subset: pd.DataFrame,
) -> tuple[np.ndarray, str, str]:
    """Resolve a right-panel axis from a column, Pi group, or power product.

    Supported ``source`` values:
      - ``column``: exact CSV column name in ``name``;
      - ``basis_pi`` / ``log_basis_pi``: original basis from calc_basis_tool;
      - ``screening_pi`` / ``log_screening_pi``: Schmidt basis used in screening;
      - ``power_product``: ``powers={column_name: exponent, ...}``.
    """
    axis = _normalize_regime_axis_spec(spec)
    if axis is None:
        raise ValueError("Physical axis is not configured.")
    source = str(axis.get("source", "column")).strip().lower()

    if source == "column":
        name = str(axis.get("name", ""))
        if name not in frame_subset.columns:
            raise ValueError(f"CSV column '{name}' is unavailable for regime plotting.")
        values = frame_subset[name].to_numpy(dtype=float)
        default_label = name
    elif source in {"basis_pi", "log_basis_pi"}:
        log_pi = _regime_plot_original_log_pi(state, frame_subset)
        index = int(axis.get("index", 1)) - 1
        if index < 0 or index >= log_pi.shape[1]:
            raise ValueError(f"Original basis Pi index {index + 1} is unavailable.")
        values = log_pi[:, index] if source.startswith("log_") else np.exp(log_pi[:, index])
        default_label = (
            rf"$\log(\Pi_{{{index + 1}}})$"
            if source.startswith("log_")
            else rf"$\Pi_{{{index + 1}}}$"
        )
    elif source in {"screening_pi", "log_screening_pi"}:
        log_pi = np.asarray(detection.get("feature_coordinates", []), dtype=float)
        index = int(axis.get("index", 1)) - 1
        if index < 0 or index >= log_pi.shape[1]:
            raise ValueError(f"Screening-basis Pi index {index + 1} is unavailable.")
        values = log_pi[:, index] if source.startswith("log_") else np.exp(log_pi[:, index])
        default_label = (
            rf"$\log(\Pi_{{{index + 1}}})$"
            if source.startswith("log_")
            else rf"$\Pi_{{{index + 1}}}$"
        )
    elif source == "power_product":
        powers = dict(axis.get("powers") or {})
        if not powers:
            raise ValueError("power_product axis requires a nonempty 'powers' mapping.")
        values = np.ones(frame_subset.shape[0], dtype=float)
        terms = []
        for name, exponent in powers.items():
            if name not in frame_subset.columns:
                raise ValueError(f"CSV column '{name}' is unavailable for a power product.")
            exponent = float(exponent)
            column = frame_subset[name].to_numpy(dtype=float)
            if np.any(column <= 0.0) and not np.isclose(exponent, round(exponent)):
                raise ValueError(
                    f"Noninteger power {exponent} of nonpositive column '{name}' is invalid."
                )
            values *= column ** exponent
            if abs(exponent) > 1.0e-12:
                terms.append(name if np.isclose(exponent, 1.0) else f"{name}^{exponent:g}")
        default_label = " ".join(terms) if terms else "1"
    else:
        raise ValueError(
            "Physical-axis source must be 'column', 'basis_pi', 'log_basis_pi', "
            "'screening_pi', 'log_screening_pi', or 'power_product'."
        )

    label = str(axis.get("label", default_label))
    scale = str(axis.get("scale", "auto")).strip().lower()
    if scale not in {"auto", "linear", "log", "symlog"}:
        raise ValueError("Physical-axis scale must be auto, linear, log, or symlog.")
    return np.asarray(values, dtype=float), label, scale


def _apply_regime_axis_scale(axis, values: np.ndarray, scale: str, which: str) -> str:
    """Apply an explicit or conservative automatic axis scale."""
    resolved = scale
    finite = values[np.isfinite(values)]
    if scale == "auto":
        resolved = "linear"
        if finite.size and np.all(finite > 0.0):
            decades = float(np.ptp(np.log10(finite)))
            if decades > 1.7:
                resolved = "log"
    if resolved == "log" and (finite.size == 0 or np.any(finite <= 0.0)):
        raise ValueError(f"The requested log {which}-axis contains nonpositive values.")
    if which == "x":
        axis.set_xscale(resolved)
    else:
        axis.set_yscale(resolved)
    return resolved


def _relabel_regime_plot_labels(labels: np.ndarray, ordering_values) -> np.ndarray:
    """Relabel clusters by increasing median of an externally chosen quantity."""
    labels = np.asarray(labels, dtype=int)
    if ordering_values is None:
        return labels.copy()
    values = np.asarray(ordering_values, dtype=float)
    order = sorted(
        np.unique(labels),
        key=lambda cluster_id: float(np.nanmedian(values[labels == cluster_id])),
    )
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _plot_generic_regime_partition(
    state: dict,
    frame: pd.DataFrame,
    save_path: Path,
    hp: dict,
) -> Optional[str]:
    """Plot selected clusters in feature space and a configured physical space."""
    result = state.get("workflow_result", {}) or {}
    detection = result.get("regime_detection", {}) or {}
    if detection.get("status") != "success":
        return None

    labels = np.asarray(detection.get("labels", []), dtype=int)
    row_positions = _regime_plot_original_indices(detection, frame)
    frame_subset = frame.iloc[row_positions].copy()
    feature_xy, feature_gradients, feature_labels, feature_description = (
        _select_regime_feature_projection(state, detection, frame_subset, hp)
    )
    if feature_xy.shape[0] != labels.size:
        raise ValueError("Feature projection and cluster labels are not aligned.")

    physical_x_spec = hp.get("regime_plot_physical_x")
    physical_y_spec = hp.get("regime_plot_physical_y")
    physical_available = physical_x_spec is not None and physical_y_spec is not None
    physical_x = physical_y = None
    physical_x_label = physical_y_label = ""
    physical_x_scale = physical_y_scale = "linear"
    if physical_available:
        physical_x, physical_x_label, physical_x_scale = _resolve_regime_physical_axis(
            physical_x_spec,
            state=state,
            detection=detection,
            frame_subset=frame_subset,
        )
        physical_y, physical_y_label, physical_y_scale = _resolve_regime_physical_axis(
            physical_y_spec,
            state=state,
            detection=detection,
            frame_subset=frame_subset,
        )

    relabel_by = str(hp.get("regime_plot_relabel_by", "physical_y")).strip().lower()
    if relabel_by == "physical_y" and physical_available:
        ordering_values = physical_y
    elif relabel_by == "physical_x" and physical_available:
        ordering_values = physical_x
    elif relabel_by in {"none", "cluster_id"}:
        ordering_values = None
    else:
        ordering_values = None
    plot_labels = _relabel_regime_plot_labels(labels, ordering_values)

    figure_size = tuple(hp.get("regime_plot_figsize", (10.8, 4.1)))
    fig, axes = plt.subplots(1, 2, figsize=figure_size, gridspec_kw={"wspace": 0.28})
    feature_axis, physical_axis = axes

    unique_labels = sorted(np.unique(plot_labels))
    for cluster_id in unique_labels:
        mask = plot_labels == cluster_id
        color = _REGIME_PLOT_COLORS[int(cluster_id) % len(_REGIME_PLOT_COLORS)]
        feature_axis.scatter(
            feature_xy[mask, 0], feature_xy[mask, 1],
            s=float(hp.get("regime_plot_point_size", 18)),
            alpha=float(hp.get("regime_plot_point_alpha", 0.74)),
            color=color, edgecolors="white", linewidths=0.25,
            label=f"Regime {cluster_id + 1}", rasterized=True,
        )
        if physical_available:
            physical_axis.scatter(
                physical_x[mask], physical_y[mask],
                s=float(hp.get("regime_plot_point_size", 18)),
                alpha=float(hp.get("regime_plot_point_alpha", 0.74)),
                color=color, edgecolors="white", linewidths=0.25,
                rasterized=True,
            )

    if bool(hp.get("regime_plot_show_gradient_arrows", True)):
        rng = np.random.default_rng(int(hp.get("regime_plot_random_state", GLOBAL_RANDOM_SEED)))
        maximum_arrows = max(1, int(hp.get("regime_plot_max_arrows", 90)))
        quota = max(1, maximum_arrows // max(len(unique_labels), 1))
        chosen = []
        for cluster_id in unique_labels:
            available = np.flatnonzero(plot_labels == cluster_id)
            if available.size:
                chosen.extend(
                    rng.choice(available, size=min(quota, available.size), replace=False).tolist()
                )
        chosen = np.asarray(sorted(set(chosen)), dtype=int)
        if chosen.size:
            vectors = feature_gradients[chosen]
            norms = np.linalg.norm(vectors, axis=1)
            good = norms > 1.0e-15
            vectors = vectors[good] / norms[good, None]
            origins = feature_xy[chosen[good]]
            arrow_labels = plot_labels[chosen[good]]
            span = np.ptp(feature_xy, axis=0)
            length = float(hp.get(
                "regime_plot_arrow_length",
                0.055 * np.sqrt(max(span[0], 1.0e-12) * max(span[1], 1.0e-12)),
            ))
            for cluster_id in unique_labels:
                mask = arrow_labels == cluster_id
                if not np.any(mask):
                    continue
                color = _REGIME_PLOT_COLORS[int(cluster_id) % len(_REGIME_PLOT_COLORS)]
                feature_axis.quiver(
                    origins[mask, 0], origins[mask, 1],
                    vectors[mask, 0] * length, vectors[mask, 1] * length,
                    angles="xy", scale_units="xy", scale=1.0,
                    width=0.0022, headwidth=3.3, headlength=4.2,
                    color=color, alpha=0.88,
                )

    feature_axis.set_xlabel(feature_labels[0])
    feature_axis.set_ylabel(feature_labels[1])
    feature_axis.set_title("Clustering results in feature space", loc="left", fontsize=10)
    feature_axis.text(
        0.0, -0.20, feature_description,
        transform=feature_axis.transAxes, fontsize=7.2, color="#59636e", va="top",
    )
    feature_axis.legend(frameon=False, fontsize=8, loc="best")

    if physical_available:
        physical_axis.set_xlabel(physical_x_label)
        physical_axis.set_ylabel(physical_y_label)
        _apply_regime_axis_scale(physical_axis, physical_x, physical_x_scale, "x")
        _apply_regime_axis_scale(physical_axis, physical_y, physical_y_scale, "y")
        physical_axis.set_title("Clustering results in physical space", loc="left", fontsize=10)
    else:
        physical_axis.text(
            0.5, 0.5,
            "Physical-space axes are not configured.\nSet regime_plot_physical_x and regime_plot_physical_y.",
            ha="center", va="center", transform=physical_axis.transAxes,
        )
        physical_axis.set_axis_off()

    for axis in axes:
        if axis.axison:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(linewidth=0.5, alpha=0.5)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=8)

    title = str(hp.get(
        "regime_plot_title",
        f"Regime partition from GPR gradients (K={detection.get('best_k')})",
    ))
    fig.suptitle(
        title, x=0.07, y=0.985, ha="left", fontsize=12, fontweight="semibold"
    )
    fig.text(
        0.07, 0.925,
        f"Target transform: {detection.get('target_transform', 'raw')} · display relabeling does not change stored cluster IDs",
        ha="left", va="top", fontsize=7.8, color="#667085",
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.79)

    saved = None
    if SAVE_PLOT:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=int(hp.get("regime_plot_dpi", 320)),
            bbox_inches="tight",
            facecolor="white",
        )
        saved = str(save_path)
    plt.show()
    plt.close(fig)
    return saved


def _local_evidence_stages(local: dict, hyperparameters: Optional[dict] = None) -> List[dict]:
    """Collect every evaluated model-selection stage for one local workflow."""
    hp = dict(hyperparameters or {})
    show_rejected = bool(hp.get("evidence_plot_show_rejected", True))
    stages = []
    for candidate in local.get("dimension_path", []) or []:
        stages.append({"label": f"k={candidate.get('pi_count')}", "kind": "count", "candidate": candidate, "accepted": True})

    sparse = local.get("sparsification") or {}
    sparse_candidate = sparse.get("candidate")
    if isinstance(sparse_candidate, dict) and sparse_candidate.get("status") == "success":
        accepted = sparse.get("status") == "accepted"
        if accepted or show_rejected:
            stages.append({
                "label": "Sparse" if accepted else "Sparse\n(rejected)",
                "kind": "sparse",
                "candidate": sparse_candidate,
                "accepted": accepted,
            })

    rational = local.get("rationalization") or {}
    rational_candidate = rational.get("candidate")
    if isinstance(rational_candidate, dict) and rational_candidate.get("mutual_information") is not None:
        accepted = bool(rational.get("accepted"))
        if accepted or show_rejected:
            stages.append({
                "label": "Rational" if accepted else "Rational\n(rejected)",
                "kind": "rational",
                "candidate": rational_candidate,
                "accepted": accepted,
            })
    return stages


def _candidate_uncertainty(candidate: dict) -> Dict[str, Any]:
    """Return restart, subsampling, and combined one-SD uncertainties for MI and exp(-MI)."""
    uq = candidate.get("mi_score_subsampling_uq") or {}
    restart_mi_std = float(candidate.get("restart_mutual_information_std", 0.0, ) or 0.0)
    restart_score_std = float(candidate.get("restart_information_score_std", 0.0, ) or 0.0)
    subsample_mi_std = float(uq.get("subsample_mi_std", 0.0, ) or 0.0)
    subsample_score_std = float(uq.get("subsample_score_std", 0.0, ) or 0.0)
    return {
        "restart_mi_std": restart_mi_std,
        "subsample_mi_std": subsample_mi_std,
        "combined_mi_std": float(np.hypot(restart_mi_std,subsample_mi_std, )),
        "restart_score_std": restart_score_std,
        "subsample_score_std": subsample_score_std,
        "combined_score_std": float(np.hypot(restart_score_std,subsample_score_std, )),
        "restart_scope": candidate.get("restart_statistics_scope", "current_candidate_cma_restarts",),
        "subsampling_scope": candidate.get("subsampling_statistics_scope", "current_candidate_subsampling",),
    }


def _plot_evidence_bar(ax, stages, metric: str, panel_label: str, show_error_bars: bool = True) -> None:
    labels = [item["label"] for item in stages]
    x = np.arange(len(stages))
    if metric == "mi":
        values = [float(item["candidate"].get("mutual_information", np.nan)) for item in stages]
        errors = [_candidate_uncertainty(item["candidate"])["combined_mi_std"] for item in stages]
        ylabel = "Mutual information"

    else:
        values = [float(item["candidate"].get("normalized_information_score", np.nan)) for item in stages]
        errors = [_candidate_uncertainty(item["candidate"])["combined_score_std"] for item in stages]
        ylabel = r"Information score $S=\exp(-MI)$"

    palette = {"count": "#2f5d73", "sparse": "#6f8f72", "rational": "#b36a3c"}
    colors = [palette.get(item["kind"], "#6b7280") if item["accepted"] else "#c6cbd1" for item in stages]
    plotted_errors = errors if show_error_bars else None
    bars = ax.bar(x, values, yerr=plotted_errors, width=0.62, color=colors, edgecolor="#263238", linewidth=0.45,
                  capsize=2.3 if show_error_bars else 0.0,
                  error_kw={"elinewidth": 0.8, "capthick": 0.8})
    for bar, value, error in zip(bars, values, errors):
        if np.isfinite(value):
            label = f"{value:.4g}"
            if show_error_bars and error > 0:
                label += f"\n±{error:.4g}"
            annotation_error = error if show_error_bars else 0.0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + annotation_error,
                    label, ha="center", va="bottom", fontsize=7.2, color="#263238")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.text(-0.13, 1.08, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, labelsize=8)
    ax.grid(axis="y", color="#d7dce1", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    if metric == "score":
        ax.set_ylim(bottom=0)


def _short_formula_label(formula: str, maximum: int = 74) -> str:
    formula = str(formula or "Pi")
    return formula if len(formula) <= maximum else formula[: maximum - 1] + "…"


def _plot_local_collapse(ax, state, local, frame, X_all, Y_all, panel_label: str) -> None:
    from .information import global_coef_to_pi_mat
    indices = np.asarray(local.get("original_csv_indices", local.get("global_sample_indices", [])), dtype=int)
    valid = indices[(indices >= 0) & (indices < X_all.shape[0])]
    final = local.get("final_solution", {}) or {}
    exponents = np.asarray(final.get("input_physical_exponents", []), dtype=float)
    if valid.size == 0 or exponents.ndim != 2 or exponents.shape[0] == 0:
        ax.text(0.5, 0.5, "Collapse data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    pi_values = np.column_stack([
        global_coef_to_pi_mat(row, X_all[valid], is_log=False).ravel()
        for row in exponents
    ])
    target = Y_all[valid, 0].copy()
    output_label = str(state.get("output_var", "output"))
    if local.get("optimize_output") and final.get("output_physical_exponents") is not None:
        output_scale = global_coef_to_pi_mat(
            np.asarray(final["output_physical_exponents"], dtype=float),
            X_all[valid],
            is_log=False,
        ).ravel()
        target = target * output_scale
        output_label = f"{output_label}*"

    formulas = final.get("input_formulas", []) or []
    if pi_values.shape[1] == 1:
        ax.scatter(pi_values[:, 0], target, s=25, alpha=0.72, color="#2f5d73", edgecolors="white", linewidths=0.35)
        ax.set_xlabel(r"Dominant group  $\Pi_1$", fontsize=9)
        ax.set_ylabel(output_label, fontsize=9)
        xscale = state.get("manual_hyperparams", {}).get("scatter_xscale", "auto")
        yscale = state.get("manual_hyperparams", {}).get("scatter_yscale", "auto")
        if xscale in ("linear", "log"):
            ax.set_xscale(xscale)
        elif np.all(pi_values[:, 0] > 0) and np.ptp(np.log10(pi_values[:, 0])) > 1.7:
            ax.set_xscale("log")
        if yscale in ("linear", "log"):
            ax.set_yscale(yscale)
        elif np.all(target > 0) and np.ptp(np.log10(target)) > 1.7:
            ax.set_yscale("log")
        formula_caption = _short_formula_label(formulas[0] if formulas else "Pi_1")
        ax.text(0.0, -0.23, formula_caption, transform=ax.transAxes, fontsize=7.2, color="#59636e", va="top")
    else:
        scatter = ax.scatter(pi_values[:, 0], pi_values[:, 1], c=target, s=25, alpha=0.78,
                             cmap="viridis", edgecolors="white", linewidths=0.3)
        ax.set_xlabel(r"$\Pi_1$", fontsize=9)
        ax.set_ylabel(r"$\Pi_2$", fontsize=9)
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03)
        cbar.set_label(output_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7, width=0.6)
        captions = " | ".join(_short_formula_label(x, 48) for x in formulas[:2])
        ax.text(0.0, -0.23, captions, transform=ax.transAxes, fontsize=7.0, color="#59636e", va="top")

    gpr = final.get("gpr_validation", {}) or {}
    r2 = gpr.get("test_r2_mean", final.get("gpr_test_r2"))
    subtitle = "Final data collapse" + (f"  ·  held-out R²={float(r2):.3f}" if r2 is not None else "")
    ax.set_title(subtitle, loc="left", fontsize=9.5, pad=7, fontweight="semibold")
    ax.text(-0.13, 1.08, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, labelsize=8)
    ax.grid(color="#d7dce1", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)


def adaptive_visualizer_node(state: AdaptiveWorkflowState):
    """Create regime-partition and local model-evidence figures."""
    from .schemas import AdaptiveWorkflowState
    from .notebook import display_workflow_path, display_workflow_stage
    hp = dict(state.get("manual_hyperparams", {}) or {})
    enabled = bool(hp.get("enable_workflow_visualization", True))
    display_workflow_stage(
        "Stage 7/7 · Data-collapse visualization",
        status="running" if enabled else "skipped",
        summary=(
            "Preparing regime-partition figures, model-selection evidence, "
            "uncertainty bars and final collapse plots."
        ),
        hyperparameters=hp,
    )
    if not enabled:
        return {}

    result = state.get("workflow_result", {}) or {}
    detection = result.get("regime_detection", {}) or {}
    make_regime_plot = bool(
        hp.get("enable_regime_partition_visualization", True)
        and detection.get("status") == "success"
    )
    local_workflows = [
        local
        for local in (result.get("local_workflows", []) or [])
        if local.get("status") == "success"
    ]
    if not make_regime_plot and not local_workflows:
        display_workflow_stage(
            "Stage 7/7 · Data-collapse visualization",
            status="skipped",
            summary="No accepted regime partition or successful local workflow is available for plotting.",
            hyperparameters=hp,
        )
        return {}

    save_dir = Path("./_Workspace_/_Plots_")
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    generated_count = 0
    plot_errors = []

    try:
        frame = pd.read_csv(state["csv_FilePath"])
        X_all = frame[state["input_vars"]].to_numpy(dtype=float)
        Y_all = frame[[state["output_var"]]].to_numpy(dtype=float)
    except Exception as exc:
        display_workflow_stage(
            "Stage 7/7 · Data-collapse visualization",
            status="warning",
            summary=f"Visualization data loading failed: {exc}",
            hyperparameters=hp,
        )
        return {"visualization_files": saved_files}

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with plt.rc_context(rc):
        if make_regime_plot:
            try:
                case_name = Path(state["csv_FilePath"]).stem
                regime_path = save_dir / f"{case_name}_regime_partition.png"
                saved = _plot_generic_regime_partition(
                    state=state,
                    frame=frame,
                    save_path=regime_path,
                    hp=hp,
                )
                generated_count += 1
                if saved:
                    saved_files.append(saved)
            except Exception as exc:
                plot_errors.append(f"regime partition: {exc}")

        for local in local_workflows:
            try:
                stages = _local_evidence_stages(local, hp)
                if not stages:
                    continue
                figure_size = tuple(hp.get("evidence_plot_figsize", (12.8, 3.65)))
                show_error_bars = bool(hp.get("evidence_plot_error_bars", True))
                fig, axes = plt.subplots(
                    1, 3, figsize=figure_size, gridspec_kw={"wspace": 0.34}
                )
                _plot_evidence_bar(
                    axes[0], stages, "mi", "a", show_error_bars=show_error_bars
                )
                _plot_evidence_bar(
                    axes[1], stages, "score", "b", show_error_bars=show_error_bars
                )
                _plot_local_collapse(
                    axes[2], state, local, frame, X_all, Y_all, "c"
                )

                scope = str(local.get("scope_id", "global")).replace("_", " ").title()
                mechanism = str(result.get("mechanism_decision", "single")).replace("_", " ")
                fig.suptitle(
                    f"{scope}: model-selection evidence and selected collapse",
                    x=0.07, ha="left", y=1.04, fontsize=12,
                    fontweight="semibold", color="#17202a",
                )
                fig.text(
                    0.07, 0.975,
                    f"Mechanism decision: {mechanism} · error bars show one standard deviation",
                    ha="left", va="top", fontsize=7.8, color="#667085",
                )
                fig.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.82)

                case_name = Path(state["csv_FilePath"]).stem
                save_path = save_dir / f"{case_name}_{local.get('scope_id', 'scope')}_evidence.png"
                if SAVE_PLOT:
                    fig.savefig(
                        save_path,
                        dpi=int(hp.get("evidence_plot_dpi", 320)),
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    saved_files.append(str(save_path))
                generated_count += 1
                plt.show()
                plt.close(fig)
            except Exception as exc:
                plot_errors.append(
                    f"{local.get('scope_id', 'local workflow')} evidence: {exc}"
                )

    status = "completed" if generated_count > 0 else "warning"
    summary = (
        "Regime-partition and evidence figures were generated."
        if generated_count > 0
        else "No visualization could be generated."
    )
    display_workflow_stage(
        "Stage 7/7 · Data-collapse visualization",
        status=status,
        summary=summary,
        rows=[
            ("Figures generated", generated_count),
            ("Figures saved", len(saved_files)),
            ("Output directory", str(save_dir)),
            ("Plot warnings", plot_errors if plot_errors else "none"),
        ],
        hyperparameters=hp,
    )
    progress_stages = list((state.get("compiled_workflow") or {}).get("stages", []) or [])
    if progress_stages:
        display_workflow_path(
            progress_stages,
            completed_through=(
                "generate_report"
                if "generate_report" in progress_stages
                else progress_stages[-1]
            ),
            hyperparameters=hp,
        )
    return {"visualization_files": saved_files}
