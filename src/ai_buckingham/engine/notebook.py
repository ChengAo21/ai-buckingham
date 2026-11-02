"""Optional notebook presentation and Markdown reporting."""
from __future__ import annotations
from .shared import *


def _inject_agent_editorial_styles() -> None:
    global _AGENT_EDITORIAL_STYLE_LOADED
    if not _AGENT_EDITORIAL_STYLE_LOADED:
        display(HTML(_AGENT_EDITORIAL_CSS))
        _AGENT_EDITORIAL_STYLE_LOADED = True


def _workflow_ui_enabled(hyperparameters: Optional[dict] = None) -> bool:
    hp = dict(hyperparameters or {})
    return bool(hp.get("show_workflow_progress", True))


def _ui_text(value: Any, max_length: int = 220) -> str:
    if value is None:
        result = "—"
    elif isinstance(value, float):
        result = f"{value:.6g}"
    elif isinstance(value, (dict, list, tuple, np.ndarray)):
        result = json.dumps(value, ensure_ascii=False, default=str)
    else:
        result = str(value)
    result = result.replace("\n", " · ").strip()
    return result if len(result) <= max_length else result[: max_length - 1] + "…"


def _stage_identity(title: str) -> tuple[str, str]:
    raw = str(title).strip()
    exact = {
        "Dimensional Analysis Agent": ("Research workflow", "Dimensional Analysis Agent"),
        "Stage 1/7 · Problem and variable analysis": ("Stage 01", "Problem formulation"),
        "Stage 2/7 · Physics consistency audit": ("Stage 02", "Physical consistency"),
        "Stage 3/7 · Buckingham–Pi basis": ("Stage 03", "Dimensional basis"),
        "Stage 4/7 · Workflow planner": ("Stage 04", "Analysis protocol"),
        "Stage 5/7 · Mechanism screening": ("Stage 05A", "Mechanism evidence"),
        "Stage 5/7 · Numerical workflow execution": ("Stage 05B", "Local model selection"),
        "Stage 6/7 · Final summary": ("Stage 06", "Selected relations"),
        "Stage 7/7 · Data-collapse visualization": ("Stage 07", "Evidence figures"),
    }
    if raw in exact:
        return exact[raw]
    if raw.startswith("Local adaptive workflow"):
        return "Local analysis", raw.replace("Local adaptive workflow · ", "")
    if "candidate k=" in raw:
        return "Model evidence", raw
    if "count decision" in raw or "skipped k=" in raw or " failed" in raw:
        return "Model selection", raw
    if raw.endswith("· sparsification"):
        return "Parsimony check", raw.split(" · ")[0]
    if raw.endswith("· rationalization"):
        return "Formula refinement", raw.split(" · ")[0]
    if raw.endswith("· local workflow complete"):
        return "Local result", raw.split(" · ")[0]
    return "Workflow event", raw


def _status_style(status: str) -> tuple[str, str, str]:
    key = str(status).lower()
    mapping = {
        "running": ("info", "Running", "info"),
        "completed": ("success", "Complete", "success"),
        "passed": ("success", "Passed", "success"),
        "warning": ("warning", "Review", "warning"),
        "failed": ("danger", "Failed", "danger"),
        "skipped": ("muted", "Skipped", "info"),
        "info": ("muted", "Info", "info"),
    }
    return mapping.get(key, ("muted", str(status), "info"))


def _details_html(rows=None, bullets=None, code=None) -> str:
    pieces = ["<details class='daa-details'><summary>Methods and diagnostics</summary>"]
    if rows:
        pieces.append("<table>")
        for key, value in rows:
            pieces.append(
                "<tr><td>" + html.escape(_ui_text(key, 80)) + "</td><td>" +
                html.escape(_ui_text(value, 700)) + "</td></tr>"
            )
        pieces.append("</table>")
    if bullets:
        pieces.append("<ul>")
        pieces.extend("<li>" + html.escape(str(item)) + "</li>" for item in bullets)
        pieces.append("</ul>")
    if code:
        pieces.append("<div class='daa-formula'>" + html.escape(str(code).strip()) + "</div>")
    pieces.append("</details>")
    return "".join(pieces)


def display_workflow_stage(
    title: str,
    *,
    status: str = "running",
    summary: Optional[str] = None,
    rows: Optional[List[tuple[str, Any]]] = None,
    bullets: Optional[List[str]] = None,
    code: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> None:
    """Render the single fixed publication-style notebook card."""
    if not _workflow_ui_enabled(hyperparameters):
        return
    _inject_agent_editorial_styles()
    kicker, clean_title = _stage_identity(title)
    tone, status_label, status_class = _status_style(status)

    all_rows = list(rows or [])
    visible_rows = all_rows[:5]
    hidden_rows = all_rows[5:]

    metric_html = ""
    if visible_rows:
        metric_html = "<div class='daa-metrics'>" + "".join(
            "<div class='daa-metric'><div class='daa-metric-key'>" +
            html.escape(_ui_text(key, 70)) + "</div><div class='daa-metric-value'>" +
            html.escape(_ui_text(value, 180)) + "</div></div>"
            for key, value in visible_rows
        ) + "</div>"

    code_lines = [line.strip() for line in str(code or "").splitlines() if line.strip()]
    formula_html = "".join(
        "<div class='daa-formula'>" + html.escape(_ui_text(line, 700)) + "</div>"
        for line in code_lines
    )

    clean_bullets = [
        item
        for item in (bullets or [])
        if item is not None and str(item).strip()
    ]

    note_html = ""
    if clean_bullets:
        note_html = (
            "<div class='daa-note'>"
            + html.escape(str(clean_bullets[0]))
            + "</div>"
        )

    detail_html = ""
    if hidden_rows or len(clean_bullets) > 1:
        detail_html = _details_html(
            hidden_rows,
            clean_bullets[1:],
            None,
        )

    card = f"""
    <div class='daa-shell'>
      <section class='daa-card' data-tone='{tone}'>
        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>{html.escape(kicker)}</div>
            <h3 class='daa-card-title'>{html.escape(clean_title)}</h3>
          </div>
          <span class='daa-status {status_class}'>{html.escape(status_label)}</span>
        </div>
        {f"<div class='daa-summary'>{html.escape(str(summary).strip())}</div>" if summary else ""}
        {metric_html}
        {formula_html}
        {note_html}
        {detail_html}
      </section>
    </div>
    """
    display(HTML(card))


def display_workflow_path(
    stages: List[str],
    *,
    current: Optional[str] = None,
    completed_through: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> None:
    if not _workflow_ui_enabled(hyperparameters) or not stages:
        return
    _inject_agent_editorial_styles()
    stage_list = list(stages)

    try:
        current_index = stage_list.index(current) if current is not None else -1
    except ValueError:
        current_index = -1
    try:
        completed_index = (
            stage_list.index(completed_through)
            if completed_through is not None
            else current_index - 1
            if current_index >= 0
            else -1
        )
    except ValueError:
        completed_index = -1

    cells = []
    for index, stage in enumerate(stage_list, start=1):
        raw = str(stage)
        clean = raw.replace("_", " ").replace("Buckingham-Pi", "Dimensional basis")
        zero_based_index = index - 1
        if zero_based_index <= completed_index:
            state_class = " complete"
        elif zero_based_index == current_index:
            state_class = " current"
        else:
            state_class = ""
        cells.append(
            f"<div class='daa-route-step{state_class}'>"
            f"<div class='daa-route-index'>{index:02d}</div>"
            f"<div class='daa-route-label'>{html.escape(clean.title())}</div></div>"
        )
    display(HTML("<div class='daa-shell'><div class='daa-route'>" + "".join(cells) + "</div></div>"))


def display_agent_run_header(
    problem_prompt: str,
    csv_path: str,
    n_rows: Optional[int],
    columns: List[str],
    hyperparameters: Optional[dict] = None,
) -> None:
    if not _workflow_ui_enabled(hyperparameters):
        return
    _inject_agent_editorial_styles()
    hp = dict(hyperparameters or {})
    template = hp.get("workflow_template", "planner selected")
    abstract = _ui_text(problem_prompt, 540)
    html_block = f"""
    <div class='daa-shell'>
      <header class='daa-masthead'>
        <div class='daa-kicker'>Data-driven dimensional analysis</div>
        <h1 class='daa-title'>Dimensional Analysis Agent</h1>
        <div class='daa-deck'>{html.escape(abstract)}</div>
        <div class='daa-meta'>
          <span><strong>Samples</strong> {html.escape(_ui_text(n_rows))}</span>
          <span><strong>Variables</strong> {len(columns)}</span>
          <span><strong>Protocol</strong> {html.escape(str(template).replace('_', ' '))}</span>
          <span><strong>Version</strong> {html.escape(AGENT_VERSION)}</span>
          <span><strong>Reproducibility</strong> manual parameters override planner suggestions</span>
        </div>
      </header>
    </div>
    """
    display(HTML(html_block))
    display_workflow_path(
        [
            "problem formulation", "physical consistency", "dimensional basis",
            "analysis protocol", "mechanism evidence", "local model selection",
            "selected relations", "evidence figures",
        ],
        current="problem formulation",
        hyperparameters=hyperparameters,
    )


def display_basis_summary(input_vars, output_var, pi_formulas, output_resolution, hyperparameters=None) -> None:
    formulas = [f"{name} = {formula}" for name, formula in (pi_formulas or {}).items()]
    display_workflow_stage(
        "Stage 3/7 · Buckingham–Pi basis",
        status="completed",
        summary=(
            "A full-precision generating basis of the dimensional null space was constructed. "
            "The basis is not unique; familiar physical groups are products and ratios of these generators."
        ),
        rows=[
            ("Input variables", len(input_vars)),
            ("Output", output_var),
            ("Output inference", output_resolution.get("source")),
            ("Basis dimension", len(pi_formulas or {})),
        ],
        bullets=[
            "These are basis generators, not yet the final governing groups selected by MI/CMA-ES."
        ],
        code="\n".join(formulas),
        hyperparameters=hyperparameters,
    )


def display_compiled_plan(
    plan,
    compiled,
    provenance,
    hyperparameters=None,
) -> None:
    """Display the selected protocol and the output-scaling decision."""
    local = dict(compiled.get("local_workflow", {}))
    stages = list(
        compiled.get("stages", plan.get("stages", []))
    )
    rationale = str(plan.get("rationale", "—"))
    if len(rationale) > 260:
        rationale = rationale[:259].rstrip() + "…"

    similarity_type = str(local.get("similarity_type", "first_kind"))
    similarity_is_auto = similarity_type == "auto"
    input_constraints = local.get("fixed_in_power") or {}
    output_constraints = local.get("fixed_out_power") or {}
    display_workflow_stage(
        "Stage 4/7 · Workflow planner",
        status="completed",
        summary=(
            "The macro protocol was selected; the similarity class will be "
            "resolved numerically within each local regime."
            if similarity_is_auto
            else (
                "The macro protocol and similarity class were selected. "
                "Count locking and all acceptance decisions remain numerical."
            )
        ),
        rows=[
            (
                "Protocol",
                str(compiled.get("template", "")).replace("_", " "),
            ),
            (
                "Similarity",
                similarity_type.replace("_", " "),
            ),
            (
                "Output optimization",
                "screening: fixed + joint"
                if similarity_is_auto
                else bool(local.get("optimize_output", False)),
            ),
            (
                "Exponent diagnostic threshold",
                local.get("similarity_screening_output_exponent_threshold")
                if similarity_is_auto else "—",
            ),
            (
                "Similarity decision · relative MI threshold",
                local.get("similarity_screening_max_relative_mi_change")
                if similarity_is_auto else "—",
            ),
            (
                "Input constraints",
                input_constraints if input_constraints else "none",
            ),
            (
                "Output constraints",
                output_constraints if output_constraints else "none",
            ),
            (
                "Scaling confidence",
                "pending numerical screen"
                if similarity_is_auto
                else local.get("output_scaling_confidence", "—"),
            ),
            (
                "Manual overrides",
                len((provenance or {}).get("manual_overrides", {})),
            ),
            (
                "Replanning budget",
                compiled.get("max_replans"),
            ),
        ],
        bullets=[
            str(local.get("output_scaling_reason", "")),
            (
                "Auto screening ignores fixed input/output basis powers and uses "
                "k=1 with the default input-output cosine threshold. Final "
                "constraints are restored only after the similarity class is resolved."
                if similarity_is_auto else ""
            ),
            rationale,
        ],
        hyperparameters=hyperparameters,
    )
    display_workflow_path(
        stages,
        current=stages[0] if stages else None,
        hyperparameters=hyperparameters,
    )


def display_similarity_screening_summary(screening: dict, hyperparameters=None) -> None:
    """Render the numerical Type-I/Type-II screening evidence for one scope."""
    if not _workflow_ui_enabled(hyperparameters):
        return
    status = str(screening.get("status", "unknown"))
    scope_id = str(screening.get("scope_id", "global"))
    if status != "success":
        display_workflow_stage(
            f"Similarity-class screening · {scope_id}",
            status="warning",
            summary=screening.get(
                "message", "The similarity-class screen did not complete."
            ),
            rows=[("Status", status)],
            hyperparameters=hyperparameters,
        )
        return

    selected = str(screening.get("selected_similarity_type", "unknown"))
    exponent_pass = bool(screening.get("output_exponent_negligible", False))
    mi_pass = bool(screening.get("mi_change_negligible", False))
    max_exponent = screening.get("max_abs_output_physical_exponent")
    exponent_threshold = screening.get("output_exponent_threshold")
    relative_mi_change = screening.get("relative_mi_loss_without_output_scaling")
    mi_threshold = screening.get("max_relative_mi_change")
    joint_summary = screening.get("joint_output_candidate", {}) or {}
    fixed_summary = screening.get("fixed_output_candidate", {}) or {}


    screening_formula_lines = []

    fixed_symbolic = list(
        fixed_summary.get("input_basis_coordinate_formulas") or []
    )
    fixed_expanded = list(
        fixed_summary.get("input_basis_expanded_formulas") or []
    )

    for i, symbolic in enumerate(fixed_symbolic):
        expanded = (
            fixed_expanded[i]
            if i < len(fixed_expanded)
            else symbolic
        )
        screening_formula_lines.append(
            f"Fixed-output optimizer basis coordinates PI_{i + 1}: "
            f"{symbolic} = {expanded}"
        )


    joint_symbolic = list(
        joint_summary.get("input_basis_coordinate_formulas") or []
    )
    joint_expanded = list(
        joint_summary.get("input_basis_expanded_formulas") or []
    )

    for i, symbolic in enumerate(joint_symbolic):
        expanded = (
            joint_expanded[i]
            if i < len(joint_expanded)
            else symbolic
        )
        screening_formula_lines.append(
            f"Joint-output optimizer basis coordinates PI_{i + 1}: "
            f"{symbolic} = {expanded}"
        )

    if joint_summary.get("output_basis_coordinate_formula"):
        screening_formula_lines.append(
            "Joint-output optimizer output coordinates: "
            f"{joint_summary.get('output_basis_coordinate_formula')} = "
            f"{joint_summary.get('output_basis_expanded_formula')}"
        )

    if joint_summary.get("output_scaling_formula"):
        screening_formula_lines.append(
            f"Screening output scale: "
            f"{joint_summary.get('output_scaling_formula')}"
        )


    criterion_text = (
        f"MI decision: {'Type I' if mi_pass else 'Type II'}; "
        f"output-exponent diagnostic: "
        f"{'negligible' if exponent_pass else 'non-negligible'}"
    )
    display_workflow_stage(
        f"Similarity-class screening · {scope_id}",
        status="completed",
        summary=(
            "The similarity class is determined by the relative mutual-information loss "
            "incurred when output scaling is removed. Output-exponent magnitude is "
            "reported as diagnostic evidence only."
        ),
        rows=[
            ("Decision", selected.replace("_", " ")),
            ("Screening k", screening.get("screening_k", SIMILARITY_SCREEN_K)),
            ("MI · fixed output", screening.get("fixed_output_mutual_information")),
            ("MI · joint output", screening.get("joint_output_mutual_information")),
            (
                "Relative MI loss without output scaling",
                f"{100.0 * float(relative_mi_change):.3g}%"
                if relative_mi_change is not None else "—",
            ),
            (
                "Relative MI threshold",
                f"≤ {100.0 * float(mi_threshold):.3g}%"
                if mi_threshold is not None else "—",
            ),
            ("max |output physical exponent|", max_exponent),
            ("Output-exponent diagnostic threshold", f"≤ {exponent_threshold}"),
            ("Criteria", criterion_text),
            ("Screening cosine threshold", screening.get("screening_max_cosine_similarity")),
            ("Fixed basis powers during screening", "none"),
            ("Final input constraints", screening.get("final_fixed_in_power") or "none"),
            ("Final output constraints", screening.get("final_fixed_out_power") or "none"),
            ("Decision confidence", screening.get("confidence", "—")),
        ],
        bullets=[
            screening.get("reason", ""),
            (
                "Screening is diagnostic only: the selected local workflow is rerun from "
                "scratch and restores the user-specified basis constraints for final inference."
            ),
            (
                "Type II was selected without an output-side uniqueness constraint; the final "
                "optimization will therefore retain the existing unconstrained Type-II behavior."
                if selected == "second_kind" and not screening.get("final_fixed_out_power")
                else ""
            ),
        ],
        code=(
            "\n".join(screening_formula_lines)
            if screening_formula_lines
            else None
        ),
        hyperparameters=hyperparameters,
    )


def _regime_diag_num(value, digits=5) -> str:
    """Compact numerical formatter for regime-screening diagnostics."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except Exception:
        return html.escape(str(value))
    if np.isnan(number):
        return "—"
    if np.isposinf(number):
        return "∞"
    if np.isneginf(number):
        return "−∞"
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude >= 1.0e4 or magnitude < 1.0e-3):
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def _regime_diag_badge(flag, *, true_text="pass", false_text="reject") -> str:
    if flag is None:
        return "<span class='daa-mini-badge neutral'>—</span>"
    css_class = "pass" if bool(flag) else "fail"
    label = true_text if bool(flag) else false_text
    return (
        f"<span class='daa-mini-badge {css_class}'>"
        + html.escape(str(label))
        + "</span>"
    )


def display_regime_detection_summary(detection: dict, hyperparameters=None) -> None:
    """Render detailed GPR-gradient regime evidence without raw console dumps.

    This is display-only. It does not alter clustering, acceptance thresholds,
    selected K, labels, or downstream local workflows.
    """
    if not _workflow_ui_enabled(hyperparameters):
        return
    _inject_agent_editorial_styles()

    hp = dict(hyperparameters or {})
    status = str(detection.get("status", "unknown"))
    single = detection.get("single_regime_baseline") or {}
    surrogate = detection.get("surrogate") or detection.get("gpr") or {}
    candidates = list(detection.get("candidate_summary", []) or [])
    selected_clusters = list(detection.get("cluster_summary", []) or [])
    criteria = dict(detection.get("acceptance_criteria") or {})

    # Fallbacks are used only for older detection payloads produced before the
    # acceptance_criteria field was added.
    nested_cluster_hp = (
        (hp.get("gradient_regime_options") or {}).get("clustering", {})
        if isinstance(hp.get("gradient_regime_options") or {}, dict)
        else {}
    )
    fraction_threshold = float(
        criteria.get(
            "minimum_cluster_fraction",
            nested_cluster_hp.get("min_cluster_fraction", 0.05),
        )
    )
    ratio_threshold = float(
        criteria.get(
            "minimum_eigenvalue_ratio",
            nested_cluster_hp.get("eigenvalue_ratio_threshold", 50.0),
        )
    )
    require_convergence = bool(
        criteria.get(
            "require_convergence",
            nested_cluster_hp.get("require_convergence", True),
        )
    )

    if status == "not_requested":
        headline = detection.get("message", "Prescribed single-regime analysis; gradient diagnostics not computed")
        tone = "success"
        status_label = "Not requested"
    elif status == "global_analysis_selected":
        headline = "Global analysis selected after collecting regime evidence"
        tone = "success"
        status_label = "Complete"
    elif status == "success":
        headline = "A multi-regime partition passed the evidence gate"
        tone = "success"
        status_label = "Complete"
    else:
        headline = detection.get(
            "message",
            "No admissible multi-regime partition was found",
        )
        tone = "warning"
        status_label = "Review"

    selected_k = detection.get(
        "best_k",
        1 if status == "global_analysis_selected" else None,
    )
    usable_gradients = len(detection.get("valid_indices", []) or [])
    excluded_gradients = detection.get(
        "excluded_directionless_samples",
        len(detection.get("excluded_directionless_indices", []) or []),
    )

    # ------------------------------------------------------------------
    # Top-level evidence metrics
    # ------------------------------------------------------------------
    metric_items = [
        ("Decision", status.replace("_", " ")),
        ("Selected K", selected_k),
        ("Selection basis", detection.get("selection_basis")),
        ("Requested K", detection.get("requested_cluster_counts", "not computed" if status == "not_requested" else [])),
        ("Failed K", [failure.get("k") for failure in detection.get("clustering_failures", [])]),
        ("All requested K attempted", (detection.get("decision_evidence") or {}).get("all_requested_K_evaluated_before_selection")),
        ("Admissible K", (detection.get("decision_evidence") or {}).get("admissible_cluster_counts")),
        ("K=2 directional support available", (detection.get("decision_evidence") or {}).get("k2_directional_support_available")),
        ("Target transform", detection.get("target_transform")),
        ("Usable gradients", usable_gradients),
        ("Excluded gradients", excluded_gradients),
        (
            "Leading spectral fraction",
            single.get("leading_fraction"),
        ),
        ("Fraction gate", f"≥ {fraction_threshold:.1%}"),
        ("Sample-count gate", "≥ " + str(criteria["minimum_cluster_samples"])
         if "minimum_cluster_samples" in criteria else "not recorded"),
        ("Eigenvalue-ratio gate", f"≥ {_regime_diag_num(ratio_threshold)}"),
        ("Auto multiple support requires K=2 |cos| <",
         str(criteria["maximum_intercluster_cosine"])
         if criteria.get("maximum_intercluster_cosine") is not None
         else "disabled" if "maximum_intercluster_cosine" in criteria else "not recorded"),
    ]
    metric_html = "<div class='daa-metrics'>" + "".join(
        "<div class='daa-metric'>"
        "<div class='daa-metric-key'>" + html.escape(str(key)) + "</div>"
        "<div class='daa-metric-value'>" + html.escape(_ui_text(value, 180)) + "</div>"
        "</div>"
        for key, value in metric_items
    ) + "</div>"

    # ------------------------------------------------------------------
    # GPR surrogate summary and restart history
    # ------------------------------------------------------------------
    gpr_rows = [
        ("Mean mode", surrogate.get("mean_mode")),
        ("Training samples", surrogate.get("n_train_samples")),
        ("sigma0²", surrogate.get("sigma0_squared")),
        ("sigma1²", surrogate.get("sigma1_squared")),
        ("sigma_u²", surrogate.get("sigma_u_squared")),
        ("Negative log likelihood", surrogate.get("negative_log_likelihood")),
        ("Optimizer success", surrogate.get("optimizer_success")),
        ("Optimizer iterations", surrogate.get("optimizer_iterations")),
    ]
    gpr_table = (
        "<div class='daa-section-title'>GPR surrogate</div>"
        "<table class='daa-diagnostic-table'><tbody>"
        + "".join(
            "<tr><td>" + html.escape(str(key)) + "</td><td class='num'>"
            + html.escape(
                str(value)
                if isinstance(value, (bool, np.bool_))
                else _regime_diag_num(value)
                if isinstance(value, (int, float, np.number))
                else _ui_text(value, 260)
            )
            + "</td></tr>"
            for key, value in gpr_rows
        )
        + "</tbody></table>"
    )

    restart_rows = []
    for item in surrogate.get("optimizer_restarts", []) or []:
        restart_rows.append(
            "<tr>"
            f"<td>{html.escape(str(int(item.get('restart_id', 0)) + 1))}</td>"
            f"<td class='num'>{html.escape(_regime_diag_num(item.get('negative_log_likelihood')))}</td>"
            f"<td>{_regime_diag_badge(item.get('success'), true_text='success', false_text='stopped')}</td>"
            f"<td class='num'>{html.escape(_regime_diag_num(item.get('iterations'), digits=6))}</td>"
            f"<td class='num'>{html.escape(_ui_text(item.get('message'), 260))}</td>"
            "</tr>"
        )

    restart_html = ""
    if restart_rows:
        restart_html = (
            "<div class='daa-section-title'>Optimizer restarts</div>"
            "<table class='daa-diagnostic-table'>"
            "<thead><tr>"
            "<th>Restart</th>"
            "<th class='num'>NLL</th>"
            "<th>Status</th>"
            "<th class='num'>Iterations</th>"
            "<th class='num'>Messages</th>"
            "</tr></thead><tbody>"
            + "".join(restart_rows)
            + "</tbody></table>"
        )

    gpr_details_html = (
        "<details class='daa-regime-details'>"
        "<summary>GPR surrogate and optimizer diagnostics</summary>"
        + gpr_table
        + restart_html
        + "</details>"
        )

    # ------------------------------------------------------------------
    # Global normalized-gradient covariance (= single-regime baseline)
    # ------------------------------------------------------------------
    global_html = ""
    if single:
        eigenvalues = single.get("eigenvalues", []) or []
        global_html = (
            "<div class='daa-section-title'>Global normalized-gradient covariance</div>"
            "<table class='daa-diagnostic-table'><tbody>"
            "<tr><td>Eigenvalues</td><td>"
            + html.escape("[" + ", ".join(_regime_diag_num(v) for v in eigenvalues) + "]")
            + "</td></tr>"
            "<tr><td>λ1 / Σλ</td><td class='num'>"
            + html.escape(_regime_diag_num(single.get("leading_fraction")))
            + "</td></tr>"
            "<tr><td>Global spectrum role</td><td>"
            + html.escape(str(single.get("role", "diagnostic_only_no_decision_threshold")))
            + "</td></tr>"
            "</tbody></table>"
        )

    # ------------------------------------------------------------------
    # Candidate-K summary. These values already exist in candidate_summary;
    # this block only exposes them in a readable table.
    # ------------------------------------------------------------------
    candidate_html = ""
    candidate_detail_html = ""
    if candidates:
        candidate_rows = []
        details = []
        for candidate in sorted(candidates, key=lambda item: int(item.get("k", 0))):
            k = int(candidate.get("k", 0))
            admissible = bool(candidate.get("meets_regime_criteria", False)) and (
                bool(candidate.get("converged", False)) or not require_convergence
            )
            selected_mark = " · selected" if selected_k is not None and int(selected_k) == k and status == "success" else ""
            candidate_rows.append(
                "<tr>"
                f"<td><strong>K={k}</strong>{html.escape(selected_mark)}</td>"
                f"<td>{_regime_diag_badge(candidate.get('converged'), true_text='yes', false_text='no')}</td>"
                f"<td class='num'>{float(candidate.get('minimum_cluster_fraction', 0.0)):.1%}</td>"
                f"<td>{_regime_diag_badge(candidate.get('passes_fraction_criterion'))}</td>"
                f"<td class='num'>{html.escape(_ui_text(candidate.get('minimum_cluster_samples'), 80))}</td>"
                f"<td>{_regime_diag_badge(candidate.get('passes_sample_count_criterion'))}</td>"
                f"<td class='num'>{html.escape(_regime_diag_num(candidate.get('minimum_eigenvalue_ratio')))}</td>"
                f"<td>{_regime_diag_badge(candidate.get('passes_eigenvalue_ratio_criterion'))}</td>"
                f"<td class='num'>{html.escape(_regime_diag_num(candidate.get('max_intercluster_abs_cosine')))}</td>"
                f"<td class='num'>{html.escape(_regime_diag_num(candidate.get('minimum_intercluster_angle_degrees')))}</td>"
                f"<td>{html.escape(str(candidate.get('k2_direction_action', 'not recorded')).replace('_', ' '))}</td>"
                f"<td>{_regime_diag_badge(admissible, true_text='admissible', false_text='reject')}</td>"
                f"<td class='num'>{html.escape(_regime_diag_num(candidate.get('clustering_objective')))}</td>"
                "</tr>"
            )

            cluster_rows = []
            for cluster in candidate.get("clusters", []) or []:
                regime_id = int(cluster.get("regime_id", cluster.get("cluster", 0))) + 1
                eigenvalues = cluster.get("eigenvalues", []) or []
                cluster_rows.append(
                    "<tr>"
                    f"<td>R{regime_id}</td>"
                    f"<td class='num'>{html.escape(_ui_text(cluster.get('n_samples'), 80))}</td>"
                    f"<td class='num'>{float(cluster.get('fraction', 0.0)):.1%}</td>"
                    f"<td class='num'>{html.escape(_regime_diag_num(cluster.get('eigenvalue_ratio_max_to_min')))}</td>"
                    f"<td>{html.escape('[' + ', '.join(_regime_diag_num(v) for v in eigenvalues) + ']')}</td>"
                    "</tr>"
                )
            direction_pairs = (candidate.get("direction_separation") or {}).get("pairs", [])
            direction_pair_html = ""
            if direction_pairs:
                direction_pair_html = (
                    "<table class='daa-diagnostic-table'><thead><tr>"
                    "<th>Cluster pair</th><th>|cos θ|</th><th>Angle (deg)</th>"
                    "</tr></thead><tbody>"
                    + "".join(
                        f"<tr><td>R{pair['cluster_i'] + 1}–R{pair['cluster_j'] + 1}</td>"
                        f"<td>{html.escape(_regime_diag_num(pair['absolute_cosine']))}</td>"
                        f"<td>{html.escape(_regime_diag_num(pair['angle_degrees']))}</td></tr>"
                        for pair in direction_pairs
                    ) + "</tbody></table>"
                )
            details.append(
                "<details class='daa-regime-details' open>"
                f"<summary>K={k} cluster composition and eigenspectra</summary>"
                "<table class='daa-diagnostic-table'>"
                "<thead><tr><th>Regime</th><th class='num'>Samples</th><th class='num'>Fraction</th>"
                "<th class='num'>λmax / λmin</th><th>Eigenvalues</th></tr></thead><tbody>"
                + ("".join(cluster_rows) if cluster_rows else "<tr><td colspan='5'>No cluster detail recorded.</td></tr>")
                + "</tbody></table>"
                + direction_pair_html
                + "<div class='daa-diagnostic-note'>"
                + html.escape(
                    "Termination: " + str(candidate.get("termination_reason", "—"))
                    + f" · iterations={candidate.get('iterations', '—')}"
                )
                + "</div></details>"
            )

        candidate_html = (
            "<div class='daa-section-title'>Candidate partition evidence</div>"
            "<table class='daa-diagnostic-table'>"
            "<thead><tr><th>K</th><th>Converged</th>"
            "<th class='num'>Min fraction</th><th>Fraction gate</th>"
            "<th class='num'>Min samples</th><th>Sample-count gate</th>"
            "<th class='num'>Min eigenvalue ratio</th><th>Ratio gate</th>"
            "<th class='num'>K=2 |cos|</th><th class='num'>K=2 angle (deg)</th><th>K=2 action</th>"
            "<th>Base admissibility</th><th class='num'>Objective</th></tr></thead><tbody>"
            + "".join(candidate_rows)
            + "</tbody></table>"
        )
        candidate_detail_html = "".join(details)

    failure_rows = []
    for failure in detection.get("clustering_failures", []) or []:
        failure_rows.append(
            "<tr><td>K=" + html.escape(str(failure.get("k", "—"))) + "</td><td>"
            + html.escape(str(failure.get("status") or "No valid candidate partition returned"))
            + "</td></tr>"
        )
    failure_html = (
        "<div class='daa-section-title'>Requested K without usable partition evidence</div>"
        "<table class='daa-diagnostic-table'><thead><tr><th>K</th><th>Status</th>"
        "</tr></thead><tbody>" + "".join(failure_rows) + "</tbody></table>"
        if failure_rows else ""
    )


    card = f"""
    <div class='daa-shell'>
      <section class='daa-card' data-tone='{tone}'>
        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>Stage 05A</div>
            <h3 class='daa-card-title'>Mechanism evidence</h3>
          </div>
          <span class='daa-status {'success' if tone == 'success' else 'warning'}'>{html.escape(status_label)}</span>
        </div>
        <div class='daa-summary'>{html.escape(str(headline))}</div>
        {metric_html}
        {gpr_details_html}
        {global_html}
        {candidate_html}
        {failure_html}
        {candidate_detail_html}
      </section>
    </div>
    """
    display(HTML(card))

    progress_stages = list(hp.get("_workflow_progress_stages", []) or [])
    if progress_stages:
        progress_current = (
            "validate_regime_partition"
            if "validate_regime_partition" in progress_stages
            else "gradient_regime_detection"
            if "gradient_regime_detection" in progress_stages
            else progress_stages[0]
        )
        display_workflow_path(
            progress_stages,
            current=progress_current,
            hyperparameters=hp,
        )


def _candidate_uncertainty_text(candidate: dict) -> str:
    from .figures import _candidate_uncertainty
    uncertainty = _candidate_uncertainty(candidate)
    return (
        f"[ MI = {uncertainty['combined_mi_std']:.3g} Res={uncertainty['restart_mi_std']:.3g}, Sub={uncertainty['subsample_mi_std']:.3g} ]"
        f"[ Score = {uncertainty['combined_score_std']:.3g} Res={uncertainty['restart_score_std']:.3g}, Sub={uncertainty['subsample_score_std']:.3g} ]"
    )


def display_local_candidate(
    scope_id: str,
    candidate: dict,
    hyperparameters=None,
) -> None:
    from .information import _candidate_formula_lines
    validation = candidate.get("gpr_validation") or {}
    stability = candidate.get("restart_stability") or {}
    formula_lines = _candidate_formula_lines(
        candidate,
        output_label=str(candidate.get("output_label", "Y")),
    )
    display_workflow_stage(
        f"{scope_id} · candidate k={candidate.get('pi_count')}",
        status="completed",
        summary=(
            "A candidate representation was optimized and checked "
            "against restart and held-out evidence."
        ),
        rows=[
            ("Mutual information", candidate.get("mutual_information")),
            (
                "Information score",
                candidate.get("normalized_information_score"),
            ),
            ("Uncertainty", _candidate_uncertainty_text(candidate)),
            ("Held-out R²", validation.get("test_r2_mean")),
            ("Successful restarts", candidate.get("restart_count")),            
            (
                "Restart stability",
                stability.get("minimum_principal_cosine"),
            ),
        ],
        bullets=[
            "Physical exponents are display-normalized; basis-coordinate powers are the raw optimizer coefficients."
        ],
        code="\n".join(formula_lines),
        hyperparameters=hyperparameters,
    )


def display_count_decision(
    scope_id: str,
    decision: dict,
    hyperparameters=None,
) -> None:
    """Display only count decisions that affect the executed path."""
    event = decision.get("event")

    if event == "COUNT_TEST_SKIPPED_INSUFFICIENT_SAMPLES":
        display_workflow_stage(
            f"{scope_id} · skipped k={decision.get('pi_count')}",
            status="warning",
            summary=(
                "The candidate was not fitted because the reproducible "
                "sample-size guard was not met."
            ),
            rows=[
                ("Available samples", decision.get("n_samples")),
                ("Required samples", decision.get("minimum_required")),
                ("Guard", decision.get("sample_guard")),
            ],
            hyperparameters=hyperparameters,
        )
    elif event == "COUNT_OPTIMIZATION_FAILED":
        display_workflow_stage(
            f"{scope_id} · k={decision.get('pi_count')} failed",
            status="warning",
            summary=(
                "All CMA-ES restarts failed; the last successful "
                "dimensionality is retained."
            ),
            rows=[("Diagnostic", decision.get("details"))],
            hyperparameters=hyperparameters,
        )


def display_local_final(
    local_result: dict,
    hyperparameters=None,
) -> None:
    from .information import _candidate_formula_lines
    if not _workflow_ui_enabled(hyperparameters):
        return

    _inject_agent_editorial_styles()

    final = local_result.get("final_solution", {}) or {}
    gpr = final.get("gpr_validation", {}) or {}
    sparsification = local_result.get("sparsification", {}) or {}
    rationalization = local_result.get("rationalization", {}) or {}

    formula_lines = _candidate_formula_lines(
        final,
        output_label=str(final.get("output_label", "Y")),
    )

    # --------------------------------------------------------------
    # Count-selection evidence
    # --------------------------------------------------------------
    decisions = list(local_result.get("count_decisions", []) or [])

    count_comparison = next(
        (
            item for item in reversed(decisions)
            if item.get("event") == "COUNT_COMPARISON"
        ),
        None,
    )

    if count_comparison:
        count_evidence = (
            f"k={count_comparison.get('previous_count')} vs "
            f"k={count_comparison.get('current_count')}; "
            f"Δscore={_ui_text(count_comparison.get('absolute_score_drop'))}; "
            f"required ≥ {_ui_text(count_comparison.get('required_drop'))}; "
            f"beyond uncertainty="
            f"{_ui_text(count_comparison.get('exceeds_uq_margin'))}"
        )
    else:
        count_evidence = (
            f"tested={_ui_text(local_result.get('tested_counts'))}; "
            f"{_ui_text(local_result.get('lock_reason'))}"
        )

    # --------------------------------------------------------------
    # Sparsification evidence
    # --------------------------------------------------------------
    sparse_parts = []

    if "dropped_variables" in sparsification:
        sparse_parts.append(
            "dropped=" + _ui_text(
                sparsification.get("dropped_variables", [])
            )
        )

    if sparsification.get("score_increase") is not None:
        sparse_parts.append(
            "Δscore="
            + _ui_text(sparsification.get("score_increase"))
        )

    if sparsification.get("maximum_allowed_score_increase") is not None:
        sparse_parts.append(
            "allowed ≤ "
            + _ui_text(
                sparsification.get("maximum_allowed_score_increase")
            )
        )

    sparse_evidence = "; ".join(sparse_parts) or "—"

    # --------------------------------------------------------------
    # Rationalization evidence
    # --------------------------------------------------------------
    rational_status = (
        "accepted"
        if rationalization.get("accepted")
        else rationalization.get("status", "not adopted")
    )

    rational_parts = []

    if rationalization.get("relative_mi_drop") is not None:
        rational_parts.append(
            "relative MI loss="
            + _ui_text(rationalization.get("relative_mi_drop"))
        )

    if rationalization.get("max_relative_mi_drop") is not None:
        rational_parts.append(
            "allowed ≤ "
            + _ui_text(rationalization.get("max_relative_mi_drop"))
        )

    rational_evidence = "; ".join(rational_parts) or "—"

    # --------------------------------------------------------------
    # Final evidence
    # --------------------------------------------------------------
    final_evidence = (
        f"MI={_ui_text(final.get('mutual_information'))}; "
        f"residual={_ui_text(final.get('normalized_information_score'))}; "
        f"held-out R²="
        f"{_ui_text(gpr.get('test_r2_mean', final.get('gpr_test_r2')))}"
    )

    rows = [
        (
            "Count selection",
            f"k={_ui_text(local_result.get('locked_count'))}",
            count_evidence,
        ),
        (
            "Sparsification",
            _ui_text(sparsification.get("status")),
            sparse_evidence,
        ),
        (
            "Rationalization",
            _ui_text(rational_status),
            rational_evidence,
        ),
        (
            "Final model",
            "complete"
            if local_result.get("status") == "success"
            else "review",
            final_evidence,
        ),
    ]

    table_html = (
        "<table class='daa-diagnostic-table'>"
        "<thead><tr>"
        "<th>Step</th>"
        "<th>Result</th>"
        "<th>Evidence</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td><strong>{html.escape(str(step))}</strong></td>"
            f"<td>{html.escape(str(result))}</td>"
            f"<td>{html.escape(str(evidence))}</td>"
            "</tr>"
            for step, result, evidence in rows
        )
        + "</tbody></table>"
    )

    formula_html = "".join(
        "<div class='daa-formula'>"
        + html.escape(_ui_text(line, 700))
        + "</div>"
        for line in formula_lines
    )

    success = local_result.get("status") == "success"

    card = f"""
    <div class='daa-shell'>
      <section class='daa-card'
               data-tone='{"success" if success else "warning"}'>

        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>Local result</div>
            <h3 class='daa-card-title'>
              {html.escape(str(local_result.get('scope_id')))}
            </h3>
          </div>

          <span class='daa-status {"success" if success else "warning"}'>
            {"Complete" if success else "Review"}
          </span>
        </div>

        <div class='daa-summary'>
          Count selection, parsimony checks, and formula refinement
          are summarized below.
        </div>

        {table_html}

        {formula_html}

        <div class='daa-note'>
          The physical and basis-coordinate forms are
          scale-equivalent representations of the selected group.
        </div>

      </section>
    </div>
    """

    display(HTML(card))

    progress_stages = list(dict(hyperparameters or {}).get("_workflow_progress_stages", []) or [])
    if progress_stages:
        progress_current = (
            "local_dominant_group_search"
            if "local_dominant_group_search" in progress_stages
            else "fan_out_local_workflows"
            if "fan_out_local_workflows" in progress_stages
            else progress_stages[-1]
        )
        display_workflow_path(
            progress_stages,
            current=progress_current,
            hyperparameters=hyperparameters,
        )


def display_final_workflow_summary(
    state: dict,
    deterministic_report: str,
    report_text: str,
) -> None:
    """Display the final deterministic results with paired formula representations."""
    from .information import _candidate_formula_lines
    hp = dict(state.get("manual_hyperparams", {}) or {})
    result = state.get("workflow_result", {}) or {}
    local_results = result.get("local_workflows", []) or []
    output_label = str(state.get("output_var") or "Y")

    _inject_agent_editorial_styles()
    cards = []
    for local in local_results:
        final = local.get("final_solution", {}) or {}
        gpr = final.get("gpr_validation", {}) or {}
        formula_lines = _candidate_formula_lines(
            final,
            output_label=output_label,
        )
        formula_html = "".join(
            "<div class='daa-formula'>" +
            html.escape(_ui_text(line, 900)) +
            "</div>"
            for line in formula_lines
        )

        requested_similarity = str(
            local.get(
                "similarity_type_requested",
                local.get("similarity_type", "first_kind"),
            )
        )

        resolved_similarity = str(
            local.get("similarity_type", "first_kind")
        )

        requested_display = requested_similarity.replace("_", " ")
        resolved_display = resolved_similarity.replace("_", " ")

        if requested_similarity == "auto":
            similarity_display = f"auto → {resolved_display}"
        else:
            similarity_display = f"set → {requested_display}"

        cards.append(
            "<article class='daa-result'>"
            f"<h4>{html.escape(str(local.get('scope_id', 'global')).replace('_', ' ').title())}</h4>"
            f"<p><strong>Selected k:</strong> {html.escape(_ui_text(local.get('locked_count')))}</p>"
            f"<p><strong>Similarity:</strong> {html.escape(_ui_text(similarity_display))}</p>"            
            f"<p><strong>Output optimization:</strong> {html.escape(_ui_text(local.get('optimize_output', False)))}</p>"
            f"<p><strong>MI / information score:</strong> "
            f"{html.escape(_ui_text(final.get('mutual_information')))} / "
            f"{html.escape(_ui_text(final.get('normalized_information_score')))}</p>"
            f"<p><strong>Held-out R²:</strong> "
            f"{html.escape(_ui_text(gpr.get('test_r2_mean', final.get('gpr_test_r2'))))}</p>"
            f"{formula_html or '<div class=\"daa-formula\">No formula</div>'}"
            "</article>"
        )

    block = f"""
    <div class='daa-shell'>
      <section class='daa-card' data-tone='success'>
        <div class='daa-card-head'>
          <div>
            <div class='daa-card-label'>Stage 06</div>
            <h3 class='daa-card-title'>Selected dimensionless relations</h3>
          </div>
          <span class='daa-status success'>
            {html.escape(str(result.get('status', 'complete')).replace('_', ' '))}
          </span>
        </div>
        <div class='daa-summary'>
          Final relations are copied directly from the numerical state.
          Each relation is shown both in physical variables and in the
          optimized Buckingham-Pi basis coordinates.
        </div>
        <div class='daa-metrics'>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Output</div>
            <div class='daa-metric-value'>{html.escape(output_label)}</div>
          </div>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Mechanism decision</div>
            <div class='daa-metric-value'>
              {html.escape(_ui_text(result.get('mechanism_decision')).replace('_', ' '))}
            </div>
          </div>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Protocol</div>
            <div class='daa-metric-value'>
              {html.escape(_ui_text(result.get('executed_template')).replace('_', ' '))}
            </div>
          </div>
          <div class='daa-metric'>
            <div class='daa-metric-key'>Local models</div>
            <div class='daa-metric-value'>{len(local_results)}</div>
          </div>
        </div>
        <div class='daa-result-grid'>{''.join(cards)}</div>
      </section>
    </div>
    """
    display(HTML(block))
    progress_stages = list((state.get("compiled_workflow") or {}).get("stages", []) or [])
    if progress_stages:
        # Aggregation is finished here. Keep Generate Report inactive until
        # the final notebook output (including evidence figures) has completed.
        display_workflow_path(
            progress_stages,
            completed_through=(
                "aggregate_results"
                if "aggregate_results" in progress_stages
                else progress_stages[-2]
                if len(progress_stages) > 1
                else None
            ),
            hyperparameters=hp,
        )

    if (
        hp.get("show_llm_report_in_notebook", True)
        and str(report_text).strip()
        and str(report_text).strip() != str(deterministic_report).strip()
    ):
        interpretation_kb = state.get("interpretation_knowledge_retrieval", {}) or {}
        grounding_note = ""
        rag_evidence_html = ""
        if interpretation_kb.get("status") == "success":
            grounding_note = (
                "<div class='daa-diagnostic-note'>"
                "Knowledge grounding · documented dimensionless relations · "
                + html.escape(_ui_text(interpretation_kb.get("embedding_model"), 80))
                + " · retrieved "
                + html.escape(_ui_text(interpretation_kb.get("n_retrieved_records", 0), 40))
                + " reference records</div>"
            )

            rag_matches = list(interpretation_kb.get("matches", []) or [])[:3]
            if rag_matches:
                rag_rows = []
                for match in rag_matches:
                    rag_rows.append(
                    "<tr>"
                    "<td><strong>"
                    + html.escape(str(match.get("canonical_name", "—")))
                    + "</strong></td>"
                    "<td>"
                    + html.escape(_ui_text(match.get("domain", "—"), 100))
                    + "</td>"
                    "<td class='num'>"
                    + html.escape(
                        f"{float(match.get('retrieval_score', 0.0)):.3f}"
                    )
                    + "</td>"
                    "<td class='num'>"
                    + html.escape(
                        f"{float(match.get('lexical_score', 0.0)):.3f}"
                    )
                    + "</td>"
                    "<td class='num'>"
                    + html.escape(
                        f"{float(match.get('semantic_score', 0.0)):.3f}"
                    )
                    + "</td>"
                    "<td>"
                    + html.escape(
                        str(match.get("retrieval_mode", "—")).replace("_", " ")
                    )
                    + "</td>"
                    "</tr>"
                    )

                rag_evidence_html = (
                    "<details class='daa-details'>"
                    "<summary>"
                    "Retrieved dimensionless knowledge · top 3 matches"
                    "</summary>"

                    "<table class='daa-diagnostic-table'>"

                    "<thead>"
                    "<tr>"
                    "<th>Relation</th>"
                    "<th>Domain</th>"
                    "<th class='num'>Score</th>"
                    "<th class='num'>Lexical</th>"
                    "<th class='num'>Semantic</th>"
                    "<th>Mode</th>"
                    "</tr>"
                    "</thead>"

                    "<tbody>"
                    + "".join(rag_rows)
                    + "</tbody>"

                    "</table>"
                    "</details>"
                )

        display(HTML(
            "<div class='daa-shell'><section class='daa-card' data-tone='muted'>"
            "<div class='daa-card-label'>Physical reading</div>"
            f"<div class='daa-summary'>{html.escape(_ui_text(report_text, 2000))}</div>"
            f"{grounding_note}"
            f"{rag_evidence_html}"
            "</section></div>"
        ))


def save_State_2md(state: dict, save_name: str, if_save=False):
    """Save one compact, human-readable Markdown report instead of dumping the full state."""
    from .figures import _candidate_uncertainty
    case_name = save_name.split("_")[0].upper()
    wk_space = f"./_Workspace_/{case_name}/"
    os.makedirs(wk_space, exist_ok=True)
    save_path = os.path.join(wk_space, f"{save_name}.md")

    if not if_save:
        print(f"⚠️ To path {save_path} State save NOT ON.")
        return save_path

    def md_text(value, max_length=1500):
        if value is None or value == "":
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        value = str(value).replace("\r", "").strip()
        return value if len(value) <= max_length else value[: max_length - 1] + "…"

    def md_cell(value, max_length=240):
        return md_text(value, max_length).replace("|", "\\|").replace("\n", "<br>")

    def write_table(handle, rows):
        handle.write("| Item | Value |\n|---|---|\n")
        for key, value in rows:
            handle.write(f"| {md_cell(key, 80)} | {md_cell(value)} |\n")
        handle.write("\n")

    def md_pct(value):
        if value is None or value == "":
            return "—"
        try:
            return f"{float(value):.1%}"
        except Exception:
            return md_text(value)

    def md_vec(value, digits=6):
        if value is None or value == "":
            return "—"
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return md_text(value, 2000)

        if arr.ndim == 0:
            return f"{float(arr):.{digits}g}"
        if arr.ndim == 1:
            return "[" + ", ".join(f"{float(x):.{digits}g}" for x in arr.tolist()) + "]"
        return json.dumps(arr.tolist(), ensure_ascii=False)

    def write_wide_table(handle, headers, rows):
        handle.write("| " + " | ".join(md_cell(h, 120) for h in headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(md_cell(v, 1200) for v in row) + " |\n")
        handle.write("\n")


    def write_rag_evidence_block(handle, summary: str, evidence: Any) -> None:
        text = str(evidence or "").strip()
        if not text:
            return
        handle.write("<details>\n")
        handle.write(f"<summary>{md_cell(summary, 160)}</summary>\n\n")
        handle.write(text + "\n\n")
        handle.write("</details>\n\n")


    result = state.get("workflow_result", {}) or {}
    compiled = state.get("compiled_workflow", {}) or {}
    local_options = compiled.get("local_workflow", {}) or {}
    local_workflows = result.get("local_workflows", []) or []
    output_var = state.get("output_var") or state.get("analyzed_output_var") or "—"
    input_vars = state.get("input_vars") or state.get("analyzed_input_vars") or []
    regime_detection = result.get("regime_detection", {}) or {}


    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Dimensional Analysis Agent Report\n\n")
        write_table(f, [
            ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Status", result.get("status", state.get("workflow_status", "—"))),
            ("Data", state.get("csv_FilePath", "—")),
            ("Agent version", globals().get("AGENT_VERSION", "—")),
        ])

        f.write("## 1. Problem\n\n")
        f.write(md_text(state.get("prompt_problem", "—"), 5000) + "\n\n")
        write_table(f, [
            ("Dependent variable", output_var),
            ("Independent inputs", input_vars),
            ("Output inference", (state.get("output_resolution") or {}).get("source", state.get("analyzed_output_reasoning", "—"))),
        ])

        f.write("## 2. Dimensional formulation\n\n")
        f.write("### Variables and dimensions\n\n")
        f.write(md_text(state.get("dimension_detect", "—"), 5000) + "\n\n")
        knowledge_retrieval = state.get("knowledge_retrieval", {}) or {}
        write_table(f, [
            ("Expected Pi count", state.get("dimension_numK", "—")),
            ("Physics audit", (state.get("hallucination_decision") or ["—"])[-1]),
            ("Knowledge retrieval", ", ".join(knowledge_retrieval.get("retrieval_modes", []) or []) or knowledge_retrieval.get("status", "—")),
            ("Embedding model", knowledge_retrieval.get("embedding_model", "—")),
            ("Retrieved inputs", knowledge_retrieval.get("retrieved_variables", [])),
        ])

        if knowledge_retrieval:
            f.write("### Retrieval-augmented physical consistency evidence\n\n")
            write_table(f, [
                ("Retrieval status", knowledge_retrieval.get("status", "—")),
                ("Knowledge base", knowledge_retrieval.get("kb_path", "—")),
                ("Knowledge-base SHA256", knowledge_retrieval.get("kb_sha256", "—")),
                ("Embedding provider", knowledge_retrieval.get("embedding_provider", "—")),
                ("Embedding model", knowledge_retrieval.get("embedding_model", "—")),
                ("Embedding base URL", knowledge_retrieval.get("embedding_base_url", "—")),
                ("Embedding cache", knowledge_retrieval.get("embedding_cache_path", "—")),
                ("Embedding status", knowledge_retrieval.get("embedding_status", "—")),
                ("Retrieval mode(s)", knowledge_retrieval.get("retrieval_modes", [])),
                ("Retrieved variables", knowledge_retrieval.get("retrieved_variables", [])),
                ("Retrieved records", knowledge_retrieval.get("n_retrieved_records", 0)),
                ("Top-k per variable", knowledge_retrieval.get("top_k_per_variable", globals().get("KNOWLEDGE_TOP_K_PER_VARIABLE", "—"))),
                ("Maximum evidence rows", knowledge_retrieval.get("max_evidence_rows", globals().get("KNOWLEDGE_MAX_EVIDENCE_ROWS", "—"))),
                ("Lexical weight", knowledge_retrieval.get("lexical_weight", globals().get("KNOWLEDGE_LEXICAL_WEIGHT", "—"))),
                ("Semantic weight", knowledge_retrieval.get("semantic_weight", globals().get("KNOWLEDGE_SEMANTIC_WEIGHT", "—"))),
                ("Minimum hybrid score", knowledge_retrieval.get("min_hybrid_score", globals().get("KNOWLEDGE_MIN_HYBRID_SCORE", "—"))),
                ("Lexical fallback allowed", knowledge_retrieval.get("allow_lexical_fallback", globals().get("KNOWLEDGE_ALLOW_LEXICAL_FALLBACK", "—"))),
            ])

            physical_rag_rows = []
            for variable_name, matches in (knowledge_retrieval.get("matches_by_variable", {}) or {}).items():
                if not matches:
                    physical_rag_rows.append([
                        variable_name, "—", "NOT_FOUND", "—", "—", "—", "—", "—", "—",
                        "No sufficiently relevant knowledge-base record was retrieved.",
                        "—", "—", "—", "—", "—",
                    ])
                    continue
                for rank, match in enumerate(matches, start=1):
                    physical_rag_rows.append([
                        variable_name,
                        rank,
                        match.get("canonical_quantity", "—"),
                        match.get("si_dimension", "—"),
                        match.get("dimension_vector", "—"),
                        match.get("lexical_score", "—"),
                        match.get("semantic_score", "—"),
                        match.get("hybrid_score", "—"),
                        match.get("retrieval_mode", "—"),
                        match.get("definition", "—"),
                        match.get("domain", "—"),
                        match.get("ambiguity_notes", "—"),
                        match.get("source", "—"),
                        match.get("source_identifier", "—"),
                        match.get("confidence", "—"),
                    ])

            if physical_rag_rows:
                write_wide_table(
                    f,
                    headers=[
                        "Variable", "Rank", "Canonical quantity", "SI dimension",
                        "Dimension vector [M,L,T,I,Theta,N,J]",
                        "Lexical score", "Semantic score", "Hybrid score", "Retrieval mode",
                        "Definition", "Domain", "Ambiguity notes", "Source",
                        "Source identifier", "Confidence",
                    ],
                    rows=physical_rag_rows,
                )

            write_rag_evidence_block(
                f,
                "Exact retrieved evidence supplied to the physics-consistency evaluator",
                knowledge_retrieval.get("evidence_markdown", ""),
            )

            audit_history = state.get("hallucination_check") or []
            if audit_history:
                f.write("**Physics-consistency evaluator output**\n\n")
                f.write("```text\n" + md_text(audit_history[-1], 12000) + "\n```\n\n")

        basis = state.get("basis_pi_formulas", {}) or {}
        if basis:
            f.write("### Buckingham–Pi basis\n\n")
            f.write("| Group | Formula |\n|---|---|\n")
            for name, formula in basis.items():
                f.write(f"| {md_cell(name, 80)} | `{md_cell(formula, 500)}` |\n")
            f.write("\n")

        f.write("## 3. Analysis protocol\n\n")
        write_table(f, [
            ("Protocol", str(compiled.get("template", "—")).replace("_", " ")),
            ("Similarity", str(local_options.get("similarity_type", result.get("similarity_type", "—"))).replace("_", " ")),
            ("Output optimization", local_options.get("optimize_output", result.get("output_optimization", False))),
            ("Input constraints", local_options.get("fixed_in_power") or "none"),
            ("Output constraints", local_options.get("fixed_out_power") or "none"),
            ("CMA-ES restarts", local_options.get("cma_restarts", "—")),
            ("Mechanism decision", str(result.get("mechanism_decision", "—")).replace("_", " ")),
        ])

        f.write("## 4. Mechanism evidence\n\n")
        if not regime_detection or regime_detection.get("status") in {None, "not_requested"}:
            f.write("Mechanism screening was not requested.\n\n")
        else:
            detection_status = regime_detection.get("status", "—")
            selected_k = regime_detection.get(
                "best_k",
                1 if detection_status == "global_analysis_selected" else "—"
            )
            single = regime_detection.get("single_regime_baseline", {}) or {}
            acceptance = regime_detection.get("acceptance_criteria", {}) or {}
            candidates = regime_detection.get("candidate_summary", []) or []

            usable_gradients = len(regime_detection.get("valid_indices", []) or [])
            excluded_gradients = regime_detection.get(
                "excluded_directionless_samples",
                len(regime_detection.get("excluded_directionless_indices", []) or [])
            )

            write_table(f, [
                ("Decision", detection_status),
                ("Selected K", selected_k),
                ("Selection basis", regime_detection.get("selection_basis", "—")),
                ("Requested K", regime_detection.get("requested_cluster_counts", [])),
                ("All requested K attempted before selection", (regime_detection.get("decision_evidence") or {}).get("all_requested_K_evaluated_before_selection", "—")),
                ("Admissible K", (regime_detection.get("decision_evidence") or {}).get("admissible_cluster_counts", "—")),
                ("K=2 directional support available", (regime_detection.get("decision_evidence") or {}).get("k2_directional_support_available", "—")),
                ("K=2 actually converged", (regime_detection.get("decision_evidence") or {}).get("k2_actually_converged", "—")),
                ("K=2 direction action", regime_detection.get("k2_direction_action", "—")),
                ("Target transform", regime_detection.get("target_transform", "—")),
                ("Usable gradients", usable_gradients),
                ("Minimum cluster samples required", acceptance.get("minimum_cluster_samples", "—")),
                ("Auto multiple support requires K=2 absolute cosine <", acceptance.get("maximum_intercluster_cosine", "—")),
                ("Excluded gradients", excluded_gradients),
                ("Low-confidence fraction @ 0.05", regime_detection.get("low_confidence_fraction_at_0.05", "—")),
                ("Feature coordinate system", regime_detection.get("feature_coordinate_system", "—")),
                ("Target assumption", regime_detection.get("target_assumption", "—")),
            ])

            if single:
                f.write("### Single-regime baseline\n\n")
                write_table(f, [
                    ("Eigenvalues", md_vec(single.get("eigenvalues", []))),
                    ("λ1 / Σλ", single.get("leading_fraction", "—")),
                    ("Global spectrum role", single.get("role", "diagnostic_only_no_decision_threshold")),
                ])

            failures = regime_detection.get("clustering_failures", []) or []
            if failures:
                f.write("### Requested K without usable partition evidence\n\n")
                write_wide_table(f, ["K", "Status"], [
                    ["K=" + str(failure.get("k", "—")),
                     failure.get("status") or "No valid candidate partition returned"]
                    for failure in failures
                ])

            if candidates:
                f.write("### Candidate partition evidence\n\n")
                require_convergence = bool(acceptance.get("require_convergence", True))
                rows = []
                for candidate in sorted(candidates, key=lambda item: int(item.get("k", 0))):
                    k = int(candidate.get("k", 0))
                    admissible = bool(candidate.get("meets_regime_criteria", False)) and (
                        bool(candidate.get("converged", False)) or not require_convergence
                    )
                    selected_mark = "yes" if regime_detection.get("best_k") == k and detection_status == "success" else ""
                    rows.append([
                        f"K={k}",
                        selected_mark,
                        candidate.get("converged", "—"),
                        md_pct(candidate.get("minimum_cluster_fraction")),
                        candidate.get("passes_fraction_criterion", "—"),
                        candidate.get("minimum_cluster_samples", "—"),
                        candidate.get("passes_sample_count_criterion", "—"),
                        candidate.get("minimum_eigenvalue_ratio", "—"),
                        candidate.get("passes_eigenvalue_ratio_criterion", "—"),
                        candidate.get("max_intercluster_abs_cosine", "—"),
                        candidate.get("minimum_intercluster_angle_degrees", "—"),
                        candidate.get("k2_direction_action", "not recorded"),
                        "admissible" if admissible else "reject",
                        candidate.get("clustering_objective", "—"),
                        candidate.get("termination_reason", "—"),
                        candidate.get("iterations", "—"),
                    ])

                write_wide_table(
                    f,
                    headers=[
                        "K",
                        "Selected",
                        "Converged",
                        "Min fraction",
                        "Fraction gate",
                        "Min samples",
                        "Sample-count gate",
                        "Min λmax/λmin",
                        "Ratio gate",
                        "K=2 |cos θ|",
                        "K=2 angle (deg)",
                        "K=2 action",
                        "Base admissibility",
                        "Objective",
                        "Termination",
                        "Iterations",
                    ],
                    rows=rows,
                )

                for candidate in sorted(candidates, key=lambda item: int(item.get("k", 0))):
                    k = int(candidate.get("k", 0))
                    selected_suffix = " (selected)" if regime_detection.get("best_k") == k and detection_status == "success" else ""
                    f.write(f"#### K={k}{selected_suffix}\n\n")

                    write_table(f, [
                        ("Converged", candidate.get("converged", "—")),
                        ("Minimum cluster fraction", md_pct(candidate.get("minimum_cluster_fraction"))),
                        ("Minimum cluster samples", candidate.get("minimum_cluster_samples", "—")),
                        ("Passes sample-count criterion", candidate.get("passes_sample_count_criterion", "—")),
                        ("K=2 absolute cosine", candidate.get("max_intercluster_abs_cosine", "—")),
                        ("K=2 angle (deg)", candidate.get("minimum_intercluster_angle_degrees", "—")),
                        ("K=2 direction action", candidate.get("k2_direction_action", "not recorded")),
                        ("Minimum eigenvalue ratio", candidate.get("minimum_eigenvalue_ratio", "—")),
                        ("Meets regime criteria", candidate.get("meets_regime_criteria", "—")),
                        ("Objective", candidate.get("clustering_objective", "—")),
                        ("Termination", candidate.get("termination_reason", "—")),
                        ("Iterations", candidate.get("iterations", "—")),
                    ])

                    direction_pairs = (candidate.get("direction_separation") or {}).get("pairs", [])
                    if direction_pairs:
                        write_wide_table(f, ["Cluster pair", "Absolute cosine", "Angle (deg)"], [
                            [f"R{pair['cluster_i'] + 1}–R{pair['cluster_j'] + 1}",
                             pair["absolute_cosine"], pair["angle_degrees"]]
                            for pair in direction_pairs
                        ])
                    clusters = candidate.get("clusters", []) or []
                    if clusters:
                        cluster_rows = []
                        for cluster in clusters:
                            regime_id = int(cluster.get("regime_id", cluster.get("cluster", 0))) + 1
                            eigvecs = cluster.get("eigenvectors_in_log_base_pi", []) or []
                            phys_dirs = cluster.get("dimensionless_exponent_directions_in_physical_variables", []) or []
                            leading_eigvec = np.asarray(eigvecs, dtype=float)[:, 0].tolist() if len(eigvecs) > 0 else None
                            leading_phys = phys_dirs[0] if len(phys_dirs) > 0 else None

                            cluster_rows.append([
                                f"R{regime_id}",
                                cluster.get("n_samples", "—"),
                                md_pct(cluster.get("fraction")),
                                cluster.get("eigenvalue_ratio_max_to_min", "—"),
                                md_vec(cluster.get("eigenvalues")),
                                md_vec(leading_eigvec),
                                md_vec(leading_phys),
                            ])

                        write_wide_table(
                            f,
                            headers=[
                                "Regime",
                                "Samples",
                                "Fraction",
                                "λmax / λmin",
                                "Eigenvalues",
                                "Leading eigenvector (log-Π basis)",
                                "Leading direction (physical vars)",
                            ],
                            rows=cluster_rows,
                        )

                        for cluster in clusters:
                            regime_id = int(cluster.get("regime_id", cluster.get("cluster", 0))) + 1

                            f.write(f"**R{regime_id} full eigenvectors in log-Π basis**\n\n")
                            f.write("```json\n")
                            f.write(json.dumps(
                                cluster.get("eigenvectors_in_log_base_pi", []),
                                ensure_ascii=False,
                                indent=2,
                            ))
                            f.write("\n```\n\n")

                            f.write(f"**R{regime_id} full dimensionless directions in physical variables**\n\n")
                            f.write("```json\n")
                            f.write(json.dumps(
                                cluster.get("dimensionless_exponent_directions_in_physical_variables", []),
                                ensure_ascii=False,
                                indent=2,
                            ))
                            f.write("\n```\n\n")


        f.write("## 5. Selected results\n\n")
        if not local_workflows:
            f.write("No successful local workflow was recorded.\n\n")
        for local in local_workflows:
            scope = str(local.get("scope_id", "global")).replace("_", " ").title()
            final = local.get("final_solution", {}) or {}
            gpr = final.get("gpr_validation", {}) or {}
            f.write(f"### {scope}\n\n")
            similarity_screen = local.get("similarity_screening") or {}
            if similarity_screen.get("status") == "success":
                f.write("**Similarity-class screening**\n\n")
                write_table(f, [
                    ("Requested class", local.get("similarity_type_requested", "auto")),
                    ("Selected class", similarity_screen.get("selected_similarity_type")),
                    ("Screening k", similarity_screen.get("screening_k")),
                    ("MI, fixed output", similarity_screen.get("fixed_output_mutual_information")),
                    ("MI, joint output", similarity_screen.get("joint_output_mutual_information")),
                    ("Relative MI loss without output scaling", similarity_screen.get("relative_mi_loss_without_output_scaling")),
                    ("Maximum allowed relative MI change", similarity_screen.get("max_relative_mi_change")),
                    ("max |output physical exponent|", similarity_screen.get("max_abs_output_physical_exponent")),
                    ("Output-exponent diagnostic threshold", similarity_screen.get("output_exponent_threshold")),
                    ("Screening cosine threshold", similarity_screen.get("screening_max_cosine_similarity")),
                    ("Decision confidence", similarity_screen.get("confidence")),
                ])
            write_table(f, [
                ("Status", local.get("status", "—")),
                ("Selected count", local.get("locked_count", "—")),
                ("Counts tested", local.get("tested_counts", [])),
                ("Similarity", str(local.get("similarity_type", "—")).replace("_", " ")),
                ("Mutual information", final.get("mutual_information", "—")),
                ("Residual score", final.get("normalized_information_score", "—")),
                ("Held-out R²", gpr.get("test_r2_mean", final.get("gpr_test_r2", "—"))),
                ("Sparsification", (local.get("sparsification") or {}).get("status", "—")),
                ("Rationalization", "accepted" if (local.get("rationalization") or {}).get("accepted") else (local.get("rationalization") or {}).get("status", "not adopted")),
            ])

            input_formulas = final.get("input_formulas", []) or []
            basis_formulas = final.get("input_basis_coordinate_formulas_display", []) or final.get("input_basis_expanded_formulas_display", []) or []
            output_formula = final.get("output_dimensionless_formula")
            f.write("**Physical-variable relation(s)**\n\n")
            f.write("```text\n" + ("\n".join(map(str, input_formulas)) if input_formulas else "—") + "\n```\n\n")
            if basis_formulas:
                f.write("**Buckingham–Pi coordinate form(s)**\n\n")
                f.write("```text\n" + "\n".join(map(str, basis_formulas)) + "\n```\n\n")
            if output_formula:
                f.write("**Output scaling**\n\n")
                f.write(f"```text\n{output_formula}\n```\n\n")

            uncertainty = _candidate_uncertainty(final)
            diagnostics = {
                "lock_reason": local.get("lock_reason"),
                "count_decisions": local.get("count_decisions", []),
                "restart_statistics": {
                    "selection_mode": final.get("selection_mode"),
                    "restart_seeds": final.get("restart_seeds", []),
                    "restart_count": final.get("restart_count"),
                    "scope": uncertainty["restart_scope"],
                    "mi_std": uncertainty["restart_mi_std"],
                    "score_std": uncertainty["restart_score_std"],
                },
                "subsampling_statistics": {
                    "scope": uncertainty["subsampling_scope"],
                    "mi_std": uncertainty["subsample_mi_std"],
                    "score_std": uncertainty["subsample_score_std"],
                },
                "combined_uncertainty": {"combination_rule": ("sqrt(restart_std^2 + subsample_std^2)"),
                    "mi_std": uncertainty["combined_mi_std"],
                    "score_std": uncertainty["combined_score_std"],
                },
                "similarity_screening": local.get("similarity_screening", {}),
                "review_flags": local.get("review_flags", {}),
            }
            f.write("<details>\n<summary>Diagnostics</summary>\n\n```json\n")
            f.write(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str))
            f.write("\n```\n\n</details>\n\n")

        final_report = state.get("final_report")
        if final_report:
            f.write("## 6. Physical interpretation\n\n")
            f.write(md_text(final_report, 5000) + "\n\n")

            interpretation_rag = state.get("interpretation_knowledge_retrieval", {}) or {}
            if interpretation_rag:
                f.write("### Dimensionless-relation retrieval provenance\n\n")
                write_table(f, [
                    ("Retrieval status", interpretation_rag.get("status", "—")),
                    ("Knowledge base", interpretation_rag.get("kb_path", "—")),
                    ("Knowledge-base SHA256", interpretation_rag.get("kb_sha256", "—")),
                    ("Embedding provider", interpretation_rag.get("embedding_provider", "—")),
                    ("Embedding model", interpretation_rag.get("embedding_model", "—")),
                    ("Embedding base URL", interpretation_rag.get("embedding_base_url", "—")),
                    ("Embedding cache", interpretation_rag.get("embedding_cache_path", "—")),
                    ("Embedding status", interpretation_rag.get("embedding_status", "—")),
                    ("Retrieval mode", interpretation_rag.get("retrieval_mode", "—")),
                    ("Retrieved records", interpretation_rag.get("n_retrieved_records", 0)),
                    ("Interpretation top-k", interpretation_rag.get("top_k", globals().get("KNOWLEDGE_INTERPRETATION_TOP_K", "—"))),
                    ("Lexical fallback allowed", interpretation_rag.get("allow_lexical_fallback", globals().get("KNOWLEDGE_ALLOW_LEXICAL_FALLBACK", "—"))),
                ])

                interpretation_rows = []
                for rank, match in enumerate(interpretation_rag.get("matches", []) or [], start=1):
                    interpretation_rows.append([
                        rank,
                        match.get("canonical_name", "—"),
                        match.get("symbols", "—"),
                        match.get("aliases", "—"),
                        match.get("defining_expression", "—"),
                        match.get("physical_interpretation", "—"),
                        match.get("domain", "—"),
                        match.get("retrieval_score", "—"),
                        match.get("lexical_score", "—"),
                        match.get("semantic_score", "—"),
                        match.get("retrieval_mode", "—"),
                        match.get("source", "—"),
                        match.get("source_identifier", "—"),
                        match.get("confidence", "—"),
                    ])

                if interpretation_rows:
                    write_wide_table(
                        f,
                        headers=[
                            "Rank", "Canonical name", "Symbols", "Aliases",
                            "Defining expression", "Physical interpretation", "Domain",
                            "Retrieval score", "Lexical score", "Semantic score",
                            "Retrieval mode", "Source", "Source identifier", "Confidence",
                        ],
                        rows=interpretation_rows,
                    )

                write_rag_evidence_block(
                    f,
                    "Exact dimensionless-relation evidence supplied to Physical reading",
                    interpretation_rag.get("evidence_markdown", ""),
                )

        figures = state.get("visualization_files", []) or []
        if figures:
            f.write("## 7. Generated figures\n\n")
            for figure in figures:
                f.write(f"- `{figure}`\n")
            f.write("\n")

    print(f"✅ State saved to markdown: {save_path}")
    return save_path


def load_file_local(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ not found at {filepath}. Returning empty string.")
        return ""
