"""Semantic analysis, planning, bounded replanning and LangGraph orchestration."""
from __future__ import annotations
from .shared import *


def analyze_problem_node(state: dict):
    """Infer the dependent variable, independent inputs, and dimensional structure."""
    from .notebook import display_agent_run_header, display_workflow_stage
    csv_columns: List[str] = []
    data_head_raw = "CSV unavailable."
    n_rows: Optional[int] = None

    if os.path.exists(state["csv_FilePath"]):
        try:
            frame = pd.read_csv(state["csv_FilePath"])
            n_rows = int(frame.shape[0])
            csv_columns = frame.columns.tolist()
            data_head_raw = frame.head().to_string()
        except Exception as exc:
            data_head_raw = f"CSV read failed: {exc}"

    previous_corrections = list(state.get("hallucination_check", []))
    if previous_corrections:
        previous_correction_text = "\n\n".join(
            f"Audit round {index+1}:\n{correction}"
            for index, correction in enumerate(previous_corrections)
        )
    else:
        previous_correction_text = ("No previous audit corrections were recorded. This is the first analysis attempt.")

    if not state.get("dimension_detect"):
        display_agent_run_header(
            problem_prompt=state["prompt_problem"],
            csv_path=state["csv_FilePath"],
            n_rows=n_rows,
            columns=csv_columns,
            hyperparameters=state.get("manual_hyperparams", {}),
        )

    display_workflow_stage(
        "Stage 1/7 · Problem and variable analysis",
        status="running",
        summary="Identifying the dependent variable, independent inputs, and fundamental dimensions.",
        rows=[("Analysis attempt", len(previous_corrections) + 1)],
        hyperparameters=state.get("manual_hyperparams", {}),
    )

    manual_output_override = dict(state.get("manual_hyperparams", {}) or {}).get("output_var")
    response = llm1_w_struOUT.invoke([
        SystemMessage(content=(
            "You are an expert physicist and applied mathematician. Infer the exact dependent/output "
            "CSV column requested by the problem statement. Preserve capitalization and symbols exactly, "
            "distinguish similarly named variables such as Nu and nu, and exclude the output from the "
            "dimensional matrix. Return only the requested structured fields."
        )),
        HumanMessage(content=(
            f"Problem statement:\n{state['prompt_problem']}\n\n"
            f"Authoritative manual output-column override (None means infer it): {manual_output_override}\n\n"
            f"CSV path: {state['csv_FilePath']}\n"
            f"CSV columns: {csv_columns}\n"
            f"CSV preview:\n{data_head_raw}\n\n"
            f"Previous physics-audit feedback:\n"
            f"{previous_correction_text}\n\n"
            "Tasks:\n"
            "1. Identify the exact dependent/output CSV column.\n"
            "2. Identify only the independent physical input columns.\n"
            "3. Give their standard fundamental dimensions.\n"
            "4. Construct the dimensional matrix excluding the output.\n"
            "5. State the expected number of independent Buckingham-Pi generators."
        )),
    ])

    display_workflow_stage(
        "Stage 1/7 · Problem and variable analysis",
        status="completed",
        summary="The variable roles and dimensional structure were inferred from the prompt and CSV schema.",
        rows=[
            ("Dependent variable", response.output_variable_name),
            ("Inference confidence", response.output_variable_confidence),
            ("Independent inputs", list(response.input_variable_names or [])),
            ("Expected Pi count", response.dimension_numK),
        ],
        bullets=[f"Output inference: {response.output_variable_reasoning}"],
        code=(
            "Variables and dimensions:\n" + str(response.dimension_detect)
        ),
        hyperparameters=state.get("manual_hyperparams", {}),
    )

    return {
        "problem_description": state["prompt_problem"],
        "analyzed_output_var": response.output_variable_name,
        "analyzed_output_reasoning": response.output_variable_reasoning,
        "analyzed_output_confidence": response.output_variable_confidence,
        "analyzed_input_vars": list(response.input_variable_names or []),
        "dimension_detect": response.dimension_detect,
        "dimension_mat": response.dimension_mat,
        "dimension_numK": response.dimension_numK,
    }


def physics_consistency_node(state: dict):
    """Audit output exclusion, dimensions, and the Buckingham-Pi count with retrieved KB evidence."""
    from .notebook import display_workflow_stage
    from .knowledge import safe_retrieve_dimension_evidence
    hp = dict(state.get("manual_hyperparams", {}) or {})
    display_workflow_stage(
        "Stage 2/7 · Physics consistency audit",
        status="running",
        summary=(
            "Checking output exclusion, dimensional consistency, and the null-space "
            "dimension with retrieval-grounded physical-quantity evidence."
        ),
        hyperparameters=hp,
    )

    problem_description = state.get(
        "problem_description", state.get("prompt_problem", "")
    )
    input_variable_names = list(state.get("analyzed_input_vars") or [])
    dimension_detect = state.get("dimension_detect", "")

    knowledge_retrieval = safe_retrieve_dimension_evidence(
        problem_description=problem_description,
        input_variable_names=input_variable_names,
        dimension_detect=dimension_detect,
    )
    retrieved_evidence = knowledge_retrieval.get(
        "evidence_markdown",
        "No external dimensional evidence was retrieved.",
    )

    response = llm4_w_struOUT_halluCheck.invoke([
        SystemMessage(content=(
            "You are a physics-consistency evaluator. Verify three requirements:\n"
            "1. The dependent/output variable must be excluded from the input dimensional matrix.\n"
            "2. Every inferred input dimension must be physically correct in standard SI base dimensions.\n"
            "3. The reported Buckingham-Pi count must equal n-r, where n is the number of "
            "independent input physical quantities and r is the dimensional-matrix rank.\n\n"
            "A small set of records has been retrieved from an external dimensional knowledge base. "
            "Use those records only as verification evidence; they do not replace the problem statement "
            "or the Analyzer interpretation. Retrieval rank is a relevance ranking, not a truth score. "
            "When several records share a symbol, resolve the physical meaning from the explicit problem "
            "description and definition.\n\n"
            "For every input variable classify the external evidence as SUPPORTED, CONFLICT, NOT_FOUND, "
            "or AMBIGUOUS. NOT_FOUND alone is not an error. AMBIGUOUS alone is not an error unless the "
            "problem context makes the Analyzer interpretation inconsistent. Issue RETRY only for an "
            "incorrectly included output, a reliable dimensional contradiction, or an incorrect n-r count."
        )),
        HumanMessage(content=(
            f"Problem:\n{problem_description}\n\n"
            f"Inferred output: {state.get('analyzed_output_var')}\n"
            f"Output inference: {state.get('analyzed_output_reasoning')}\n\n"
            f"Independent input variables: {input_variable_names}\n\n"
            f"Input descriptions:\n{dimension_detect}\n\n"
            f"Dimensional matrix:\n{state.get('dimension_mat', '')}\n\n"
            f"Pi-count statement:\n{state.get('dimension_numK', '')}\n\n"
            f"Retrieved dimensional evidence:\n{retrieved_evidence}"
        )),
    ])

    retrieval_modes = knowledge_retrieval.get("retrieval_modes", []) or []
    retrieval_mode_text = (
        ", ".join(retrieval_modes)
        if retrieval_modes
        else knowledge_retrieval.get("status", "unavailable")
    )
    retrieved_inputs = knowledge_retrieval.get(
        "retrieved_variables", input_variable_names
    )

    rag_preview_lines = []
    for variable_name, matches in (
        knowledge_retrieval.get("matches_by_variable", {}) or {}
    ).items():
        
        if not matches:
            rag_preview_lines.append(
                f"{variable_name} → no sufficiently relevant KB record"
            )
            continue
        top = matches[0]
        rag_preview_lines.append(
        f"{variable_name} → "
        f"{top.get('canonical_quantity', '—')} · "
        f"{top.get('si_dimension', '—')} "
        f"[ lexical {float(top.get('lexical_score', 0.0)):.3f} / "
        f"semantic {float(top.get('semantic_score', 0.0)):.3f} ] → "
        f"score {float(top.get('hybrid_score', 0.0)):.3f}"
    )


    display_workflow_stage(
        "Stage 2/7 · Physics consistency audit",
        status="passed" if response.decision == "PASSED" else "warning",
        summary=(
            "The dimensional specification passed the retrieval-grounded consistency audit."
            if response.decision == "PASSED"
            else "The audit identified an issue and the analyzer will be asked to revise its result."
        ),
        rows=[
            ("Decision", response.decision),
            ("Knowledge retrieval", retrieval_mode_text),
            ("Embedding model", knowledge_retrieval.get("embedding_model", KNOWLEDGE_EMBEDDING_MODEL)),
            ("Audited inputs", retrieved_inputs),
            ("Retrieved KB records", knowledge_retrieval.get("n_retrieved_records", 0)),
        ],
        code=(
            "\n".join(rag_preview_lines)
            if rag_preview_lines else "No external KB evidence retrieved."
        ),
        bullets=[
            response.check_reasoning,
            response.correction_msg if response.correction_msg else "No correction requested.",
        ],
        hyperparameters=hp,
    )

    audit_message = (
        f"Decision: {response.decision}\n"
        f"Reasoning: {response.check_reasoning}\n"
        f"Knowledge evidence status: {response.knowledge_evidence_status}\n"
        f"Knowledge evidence reasoning: {response.knowledge_evidence_reasoning}\n"
        f"Correction: {response.correction_msg if response.correction_msg else 'None'}"
    )
    return {
        "hallucination_check": [audit_message],
        "hallucination_decision": [response.decision],
        "knowledge_retrieval": knowledge_retrieval,
    }


from .discovery import _rationalize_local_solution, run_local_dominant_group_workflow
from .basis import calc_basis_tool
from .regimes import detect_gradient_regimes
WORKFLOW_TOOL_REGISTRY = {
    "calc_basis_tool": calc_basis_tool,
    "detect_gradient_regimes": detect_gradient_regimes,
    "run_local_dominant_group_workflow": run_local_dominant_group_workflow,
    "rationalize_local_solution": _rationalize_local_solution,
}


def _infer_workflow_template_from_context(
    problem_prompt: str,
    hp: dict,
) -> str:
    """Infer the canonical workflow template; single regime is the default."""
    explicit = hp.get("workflow_template")
    if explicit in WORKFLOW_TEMPLATE_STAGES:
        return str(explicit)

    prompt = str(problem_prompt).lower()
    auto_markers = (
        "automatic mechanism screening",
        "automatically detect regimes",
        "auto-detect regimes",
        "自动探测机制",
        "自动识别机制",
        "自动聚类机制",
    )
    multi_markers = (
        "multiple regime",
        "multiple-regime",
        "multi-regime",
        "different regime",
        "multiple mechanism",
        "multi-mechanism",
        "different mechanism",
        "transition regime",
        "不同机制",
        "多个机制",
        "多机制",
        "不同区间",
        "多个阶段",
        "不同阶段",
    )
    if any(marker in prompt for marker in auto_markers):
        return "auto_mechanism_adaptive"
    if any(marker in prompt for marker in multi_markers):
        return "multi_regime_adaptive"
    return "single_regime_adaptive"


def _infer_output_optimization_from_context(
    problem_prompt: str,
    hp: dict,
) -> dict:
    """Deterministic fallback for the three-way similarity policy.

    This mirrors _infer_workflow_template_from_context: first_kind is the
    default/null policy; second_kind is selected when output scaling is explicitly
    prescribed; auto is selected only when automatic similarity-class screening
    is explicitly requested. Normal LLM-planner output is not overwritten by
    this fallback; manual hyperparameters remain authoritative.
    """
    prompt = str(problem_prompt).lower()

    if "optimize_output" in hp:
        enabled = bool(hp["optimize_output"])
        return {
            "optimize_output": enabled,
            "similarity_type": "second_kind" if enabled else "first_kind",
            "reason": "Authoritative manual_hyperparams optimize_output override.",
            "confidence": "high",
            "source": "manual_hyperparams",
        }

    manual_similarity = hp.get("similarity_type")
    if manual_similarity in {"auto", "first_kind", "second_kind"}:
        if manual_similarity == "auto":
            return {
                "optimize_output": None,
                "similarity_type": "auto",
                "reason": (
                    "Authoritative manual similarity_type='auto'; runtime will compare "
                    "fixed-output and joint-output k=1 solutions."
                ),
                "confidence": "medium",
                "source": "manual_hyperparams",
            }
        enabled = manual_similarity == "second_kind"
        return {
            "optimize_output": enabled,
            "similarity_type": manual_similarity,
            "reason": "Authoritative manual similarity_type override.",
            "confidence": "high",
            "source": "manual_hyperparams",
        }

    auto_markers = (
        "automatic similarity screening",
        "automatic similarity-class screening",
        "automatically determine similarity type",
        "automatically determine similarity class",
        "auto-detect similarity",
        "auto detect similarity",
        "automatically distinguish type i and type ii",
        "automatically distinguish type i from type ii",
        "automatically determine first- or second-kind similarity",
        "automatically determine first or second kind similarity",
        "自动判断相似性类型",
        "自动识别相似性类型",
        "自动判断第一类和第二类相似性",
        "自动判断第一类或第二类相似性",
        "自动识别第一类和第二类相似性",
        "自动相似性筛选",
    )
    if any(marker in prompt for marker in auto_markers):
        return {
            "optimize_output": None,
            "similarity_type": "auto",
            "reason": (
                "The prompt explicitly requests automatic Type-I/Type-II similarity screening."
            ),
            "confidence": "medium",
            "source": "prompt_rule",
        }

    # If automatic screening was explicitly requested above, any output-side
    # uniqueness constraints are retained for final Type-II inference but do not
    # force the screening result. Without an auto request, fixed_out_power implies
    # an explicit second-kind analysis, preserving the previous behavior.
    if hp.get("fixed_out_power"):
        return {
            "optimize_output": True,
            "similarity_type": "second_kind",
            "reason": (
                "A fixed_out_power constraint requires an output-side "
                "dimensionless scaling group."
            ),
            "confidence": "high",
            "source": "manual_hyperparams",
        }

    negative_markers = (
        "do not optimize the output",
        "output optimization is disabled",
        "output should not be scaled",
        "no output scaling",
        "first-kind similarity",
        "first kind similarity",
        "不优化输出",
        "无需优化输出",
        "输出不缩放",
        "第一类相似性",
    )
    complete_similarity_requested = bool(
        re.search(r"(?<!in)complete similarity", prompt)
        or re.search(r"(?<!不)完全相似", prompt)
    )
    positive_markers = (
        "output variable needs to be scaled",
        "output needs to be scaled",
        "scale the output",
        "scaled dependent variable",
        "output optimization",
        "optimize the output",
        "incomplete similarity",
        "second-kind similarity",
        "second kind similarity",
        "输出变量需要缩放",
        "需要优化输出",
        "输出缩放",
        "第二类相似性",
        "不完全相似",
    )

    if any(marker in prompt for marker in negative_markers) or complete_similarity_requested:
        return {
            "optimize_output": False,
            "similarity_type": "first_kind",
            "reason": "The prompt explicitly disables dependent-variable scaling.",
            "confidence": "high",
            "source": "prompt_rule",
        }
    if any(marker in prompt for marker in positive_markers):
        return {
            "optimize_output": True,
            "similarity_type": "second_kind",
            "reason": (
                "The prompt explicitly requests incomplete-similarity output scaling."
            ),
            "confidence": "high",
            "source": "prompt_rule",
        }

    return {
        "optimize_output": False,
        "similarity_type": "first_kind",
        "reason": (
            "First-kind similarity is the default/null policy because neither output "
            "scaling nor automatic similarity screening was explicitly requested."
        ),
        "confidence": "medium",
        "source": "default",
    }


def _fallback_workflow_plan(problem_prompt: str, hp: dict) -> WorkflowPlan:
    """Construct a deterministic plan when the planner call is unavailable."""
    from .schemas import LocalDominantWorkflowPlan, RegimeDetectionPlan, WorkflowPlan
    from .regimes import _reject_removed_regime_controls
    template = _infer_workflow_template_from_context(problem_prompt, hp)
    output_policy = _infer_output_optimization_from_context(problem_prompt, hp)

    local_payload = {
        "start_count": int(hp.get("workflow_start_count", 1)),
        "max_count": hp.get("workflow_max_count"),
        "absolute_score_drop_threshold": float(
            hp.get("abs_score_drop_threshold", ABS_SCORE_DROP_THRESHOLD)
        ),
        "require_two_counts_when_possible": bool(
            hp.get("workflow_require_two_counts_when_possible", True)
        ),

        "similarity_type": output_policy["similarity_type"],
        "optimize_output": output_policy["optimize_output"],
        "output_scaling_reason": output_policy["reason"],
        "output_scaling_confidence": output_policy["confidence"],
        "similarity_screening_output_exponent_threshold": float(
            hp.get(
                "similarity_screening_output_exponent_threshold",
                SIMILARITY_OUTPUT_EXPONENT_THRESHOLD,
            )
        ),
        "similarity_screening_max_relative_mi_change": float(
            hp.get(
                "similarity_screening_max_relative_mi_change",
                SIMILARITY_MAX_RELATIVE_MI_CHANGE,
            )
        ),
        "similarity_screening_max_cosine_similarity": float(
            hp.get(
                "similarity_screening_max_cosine_similarity",
                SIMILARITY_SCREEN_MAX_COSINE,
            )
        ),

        "mi_method": hp.get("mi_method_used", "auto"),
        "mi_param": hp.get("mi_param_used"),
        "cma_restarts": int(
            hp.get("regime_cma_restarts", hp.get("cma_restarts", 3))
        ),
        "minimum_samples": int(hp.get("regime_min_samples", 20)),
        "minimum_samples_per_mi_dimension": int(
            hp.get("regime_min_samples_per_mi_dimension", 20)
        ),
        "minimum_samples_for_second_count": int(
            hp.get("workflow_minimum_samples_for_second_count", 40)
        ),
        "fixed_in_power": hp.get("fixed_in_power", {}) or {},
        "fixed_out_power": hp.get("fixed_out_power", {}) or {},
        "sparsification_enabled": bool(
            hp.get("workflow_sparsification_enabled", True)
        ),
        "sparsify_exponent_threshold": float(
            hp.get("regime_sparsify_exponent_threshold", SPARSE_THRES)
        ),
        "sparsify_max_score_increase": float(
            hp.get("regime_sparsify_max_score_increase", 0.05)
        ),
        "rationalization_enabled": bool(hp.get("rationalized_on", True)),
        "max_rational_denominator": int(
            hp.get("max_rational_denominator", 5)
        ),
    }

    _reject_removed_regime_controls(hp=hp)
    regime_payload = {}

    return WorkflowPlan(
        template=template,
        rationale=(
            "Deterministic fallback selected from explicit settings and "
            "problem wording."
        ),
        target_is_dimensionless=bool(
            hp.get("target_is_dimensionless", True)
        ),
        stages=WORKFLOW_TEMPLATE_STAGES[template],
        regime_detection=RegimeDetectionPlan(**regime_payload),
        local_analysis=LocalDominantWorkflowPlan(**local_payload),
    )


def _apply_manual_workflow_overrides(
    payload: dict,
    hp: dict,
) -> tuple[dict, dict]:
    """Apply authoritative manual settings after planner suggestions."""
    from .information import _deep_update_dict
    from .regimes import _reject_removed_regime_controls
    _reject_removed_regime_controls(hp=hp, plan=payload)
    if hp.get("workflow_template") is not None and hp["workflow_template"] not in WORKFLOW_TEMPLATE_STAGES:
        raise ValueError("Unknown explicit workflow_template: " + str(hp["workflow_template"]))
    result = _deep_update_dict({}, payload)
    provenance = {
        "precedence": [
            "library_defaults",
            "llm_workflow_plan",
            "manual_hyperparams",
        ],
        "manual_overrides": {},
        "passthrough_manual_hyperparameters": {},
    }

    def record(path: str, value: Any) -> None:
        provenance["manual_overrides"][path] = value

    if hp.get("workflow_template") is not None:
        result["template"] = _infer_workflow_template_from_context("", hp)
        record("template", result["template"])

    if "target_is_dimensionless" in hp:
        result["target_is_dimensionless"] = bool(
            hp["target_is_dimensionless"]
        )
        record(
            "target_is_dimensionless",
            result["target_is_dimensionless"],
        )

    local = dict(result.get("local_analysis", {}))
    local_nested = {}
    if isinstance(hp.get("local_workflow_options"), dict):
        local_nested = _deep_update_dict(
            local_nested,
            hp["local_workflow_options"],
        )
    gradient_nested = hp.get("gradient_regime_options") or {}
    if (
        isinstance(gradient_nested, dict)
        and isinstance(gradient_nested.get("local_workflow"), dict)
    ):
        local_nested = _deep_update_dict(
            local_nested,
            gradient_nested["local_workflow"],
        )
    local = _deep_update_dict(local, local_nested)
    for key, value in local_nested.items():
        record(f"local_analysis.{key}", value)

    local_flat_mapping = {
        "workflow_start_count": "start_count",
        "workflow_max_count": "max_count",
        "abs_score_drop_threshold": "absolute_score_drop_threshold",
        "mi_method_used": "mi_method",
        "mi_param_used": "mi_param",
        "fixed_in_power": "fixed_in_power",
        "fixed_out_power": "fixed_out_power",
        "workflow_sparsification_enabled": "sparsification_enabled",
        "regime_sparsify_exponent_threshold": "sparsify_exponent_threshold",
        "regime_sparsify_max_score_increase": "sparsify_max_score_increase",
        "rationalized_on": "rationalization_enabled",
        "max_rational_denominator": "max_rational_denominator",
        "validate_after_rationalization": "validate_after_rationalization",
        "regime_min_samples": "minimum_samples",
        "regime_min_samples_per_mi_dimension":
            "minimum_samples_per_mi_dimension",
        "workflow_minimum_samples_for_second_count":
            "minimum_samples_for_second_count",
        "workflow_require_two_counts_when_possible":
            "require_two_counts_when_possible",
        "similarity_screening_output_exponent_threshold":
            "similarity_screening_output_exponent_threshold",
        "similarity_screening_max_relative_mi_change":
            "similarity_screening_max_relative_mi_change",
        "similarity_screening_max_cosine_similarity":
            "similarity_screening_max_cosine_similarity",
    }
    for hp_key, plan_key in local_flat_mapping.items():
        if hp_key in hp and hp[hp_key] is not None:
            local[plan_key] = hp[hp_key]
            record(f"local_analysis.{plan_key}", hp[hp_key])

    if hp.get("regime_cma_restarts") is not None:
        local["cma_restarts"] = int(hp["regime_cma_restarts"])
        record("local_analysis.cma_restarts", local["cma_restarts"])
    elif hp.get("cma_restarts") is not None:
        local["cma_restarts"] = int(hp["cma_restarts"])
        record("local_analysis.cma_restarts", local["cma_restarts"])

    if "optimize_output" in hp:
        local["optimize_output"] = bool(hp["optimize_output"])
        local["similarity_type"] = (
            "second_kind" if local["optimize_output"] else "first_kind"
        )
        local["output_scaling_reason"] = (
            "Authoritative manual_hyperparams override."
        )
        local["output_scaling_confidence"] = "high"
        record("local_analysis.optimize_output", local["optimize_output"])
        record("local_analysis.similarity_type", local["similarity_type"])
    elif hp.get("similarity_type") in {"auto", "first_kind", "second_kind"}:
        local["similarity_type"] = str(hp["similarity_type"])
        local["optimize_output"] = (
            None
            if local["similarity_type"] == "auto"
            else local["similarity_type"] == "second_kind"
        )
        local["output_scaling_reason"] = (
            "Authoritative manual similarity_type='auto'; runtime numerical screening required."
            if local["similarity_type"] == "auto"
            else "Authoritative manual similarity_type override."
        )
        local["output_scaling_confidence"] = (
            "medium" if local["similarity_type"] == "auto" else "high"
        )
        record("local_analysis.similarity_type", local["similarity_type"])
        record("local_analysis.optimize_output", local["optimize_output"])

    if (
        local.get("fixed_out_power")
        and "optimize_output" not in hp
        and hp.get("similarity_type") is None
        and str(local.get("similarity_type", "first_kind")) != "auto"
    ):
        local["optimize_output"] = True
        local["similarity_type"] = "second_kind"
        local["output_scaling_reason"] = (
            "A fixed_out_power constraint requires output optimization."
        )
        local["output_scaling_confidence"] = "high"
        record("local_analysis.optimize_output", True)
        record("local_analysis.similarity_type", "second_kind")

    if hp.get("output_scaling_reason") is not None:
        local["output_scaling_reason"] = str(
            hp["output_scaling_reason"]
        )
        record(
            "local_analysis.output_scaling_reason",
            local["output_scaling_reason"],
        )

    result["local_analysis"] = local

    # Regime options are resolved once in compile_workflow_plan.  Keeping
    # the raw plan here prevents nested/flat settings being merged twice.
    regime_flat_mapping = {
        "regime_n_clusters": "n_clusters",
        "regime_candidate_cluster_counts": "candidate_cluster_counts",
        "regime_min_cluster_fraction": "min_cluster_fraction",
        "regime_min_cluster_samples": "min_cluster_samples",
        "regime_max_intercluster_cosine": "max_intercluster_cosine",
        "regime_eigenvalue_ratio_threshold": "eigenvalue_ratio_threshold",
        "regime_require_convergence": "require_convergence",
        "regime_exclude_ambiguous": "exclude_low_confidence_points",
        "regime_assignment_confidence_threshold": "assignment_confidence_threshold",
    }

    consumed = {
        "workflow_plan",
        "workflow_template",
        "target_is_dimensionless",
        "local_workflow_options",
        "gradient_regime_options",
        "regime_cma_restarts",
        "cma_restarts",
        "optimize_output",
        "similarity_type",
        "output_scaling_reason",
        *local_flat_mapping.keys(),
        *regime_flat_mapping.keys(),
    }
    provenance["passthrough_manual_hyperparameters"] = {
        key: value
        for key, value in hp.items()
        if key not in consumed
    }

    result["stages"] = WORKFLOW_TEMPLATE_STAGES[result["template"]]
    return result, provenance


def create_workflow_plan(
    problem_prompt: str,
    dataset_summary: str,
    basis_count: int,
    basis_formulas: Optional[Dict[str, str]] = None,
    manual_hyperparams: Optional[dict] = None,
) -> WorkflowPlan:
    """Ask the LLM for a macro protocol, then apply manual overrides."""
    from .schemas import WorkflowPlan
    from .information import _pydantic_dump, _pydantic_validate
    from .regimes import _reject_removed_regime_controls
    hp = dict(manual_hyperparams or {})
    _reject_removed_regime_controls(hp=hp)
    if hp.get("workflow_template") is not None and hp["workflow_template"] not in WORKFLOW_TEMPLATE_STAGES:
        raise ValueError("Unknown explicit workflow_template: " + str(hp["workflow_template"]))
    supplied_plan = hp.get("workflow_plan")

    if supplied_plan:
        plan = _pydantic_validate(WorkflowPlan, supplied_plan)
    else:
        system_message = SystemMessage(content=(
            "You are the macro-workflow planner for a data-driven dimensional-analysis agent. "
            "Select one workflow template and one similarity-analysis policy. The similarity policy "
            "mirrors the mechanism policy: first_kind is the default/null policy; use second_kind only "
            "when the prompt explicitly requests incomplete similarity or dependent-variable scaling; "
            "use auto only when the prompt explicitly requests automatic Type-I/Type-II similarity "
            "screening. Set optimize_output=False for first_kind, True for second_kind, and null for auto. "
            "In auto mode, runtime resolves the class numerically by comparing fixed-output and jointly "
            "optimized k=1 solutions. Do not decide the final number of dominant Pi groups: runtime must "
            "test k=1,2,... and lock the count numerically using S=exp(-MI). For accepted multiple regimes, "
            "auto similarity screening is performed independently within each regime. Use only fields "
            "from the schema and do not invent tool names."
        ))
        human_message = HumanMessage(content=(
            f"Problem prompt:\n{problem_prompt}\n\n"
            f"Dataset summary:\n{dataset_summary}\n\n"
            f"Available base-Pi coordinates: {basis_count}\n"
            f"Base-Pi generators: {json.dumps(basis_formulas or {}, ensure_ascii=False)}\n\n"
            "Protocol rules:\n"
            "- single_regime_adaptive is the default/null protocol.\n"
            "- multi_regime_adaptive is used only when multiple mechanisms are explicitly expected.\n"
            "- auto_mechanism_adaptive is used only when automatic mechanism screening is explicitly requested.\n"
            "- In auto mode, K=1 is the null hypothesis; K>=2 is accepted only with sufficient evidence.\n"
            "- fixed_in_power and fixed_out_power may contain only valid Pi_1,...,Pi_n keys.\n"
            "- A dimensionless output may still require output optimization under incomplete similarity; "
            "target_is_dimensionless and optimize_output are independent decisions.\n"
            "- first_kind is the default/null similarity policy when neither output scaling nor automatic "
            "similarity screening is explicitly requested.\n"
            "- second_kind is used when incomplete similarity or dependent-variable scaling is explicitly requested.\n"
            "- auto is used only when automatic Type-I/Type-II similarity screening is explicitly requested; "
            "set optimize_output=null in this mode.\n"
            f"- Prefer kraskov for sample size <= {MI_BINNING_SAMPLE_THRESHOLD} and binning otherwise; select mi_param accordingly to sample size and estimator.\n"
        ))
        try:
            plan = llm_workflow_planner.invoke(
                [system_message, human_message]
            )
        except Exception:
            plan = _fallback_workflow_plan(problem_prompt, hp)

    # Match the mechanism-policy precedence exactly: library/schema defaults <
    # LLM planner < manual_hyperparams. Deterministic context inference is used
    # only by _fallback_workflow_plan; normal planner output is not sanitized or
    # overwritten here.
    payload, _ = _apply_manual_workflow_overrides(
        _pydantic_dump(plan),
        hp,
    )
    return _pydantic_validate(WorkflowPlan, payload)


def compile_workflow_plan(
    plan: WorkflowPlan,
    basis_count: int,
    manual_hyperparams=None,
) -> dict:
    """Validate planner output and compile it into an executable policy."""
    from .schemas import RegimeDetectionPlan, WorkflowPlan
    from .information import _deep_update_dict, _pydantic_dump
    from .regimes import _normalize_gradient_regime_options, _reject_removed_regime_controls, _validate_regime_configuration
    hp = dict(manual_hyperparams or {})
    payload = _pydantic_dump(plan)
    _reject_removed_regime_controls(hp=hp, plan=payload)
    template = payload["template"]
    if template not in WORKFLOW_TEMPLATE_STAGES:
        raise ValueError(f"Unknown workflow template: {template}")

    local = dict(payload["local_analysis"])
    local["start_count"] = max(1, int(local.get("start_count", 1)))
    if local.get("max_count") is None:
        local["max_count"] = int(basis_count)
    else:
        local["max_count"] = max(
            local["start_count"],
            min(int(local["max_count"]), int(basis_count)),
        )

    local["absolute_score_drop_threshold"] = max(
        0.0,
        float(local["absolute_score_drop_threshold"]),
    )
    local["cma_restarts"] = max(
        1,
        int(local.get("cma_restarts", 3)),
    )
    local["minimum_samples"] = max(
        10,
        int(local.get("minimum_samples", 20)),
    )
    local["minimum_samples_per_mi_dimension"] = max(
        5,
        int(local.get("minimum_samples_per_mi_dimension", 20)),
    )
    local["minimum_samples_for_second_count"] = max(
        10,
        int(local.get("minimum_samples_for_second_count", 40)),
    )

    similarity_type = str(local.get("similarity_type", "first_kind"))
    if similarity_type not in {"auto", "first_kind", "second_kind"}:
        raise ValueError(f"Unknown similarity_type: {similarity_type}")
    local["similarity_type"] = similarity_type
    # Canonical three-way policy: False = explicit/default Type I, True =
    # explicit Type II, None = unresolved auto screening. Runtime never treats
    # None as Type I; it triggers the fixed-vs-joint k=1 similarity screen.
    local["optimize_output"] = (
        None
        if similarity_type == "auto"
        else similarity_type == "second_kind"
    )
    local["similarity_screening_output_exponent_threshold"] = max(
        0.0,
        float(
            local.get(
                "similarity_screening_output_exponent_threshold",
                SIMILARITY_OUTPUT_EXPONENT_THRESHOLD,
            )
        ),
    )
    local["similarity_screening_max_relative_mi_change"] = max(
        0.0,
        float(
            local.get(
                "similarity_screening_max_relative_mi_change",
                SIMILARITY_MAX_RELATIVE_MI_CHANGE,
            )
        ),
    )
    local["similarity_screening_max_cosine_similarity"] = min(
        1.0,
        max(
            0.0,
            float(
                local.get(
                    "similarity_screening_max_cosine_similarity",
                    SIMILARITY_SCREEN_MAX_COSINE,
                )
            ),
        ),
    )
    if not str(local.get("output_scaling_reason", "")).strip():
        local["output_scaling_reason"] = (
            "Similarity class will be resolved by numerical screening."
            if similarity_type == "auto"
            else "Dependent-variable scaling enabled."
            if local["optimize_output"]
            else "Dependent-variable scaling not requested."
        )
    if local.get("output_scaling_confidence") not in {
        "high",
        "medium",
        "low",
    }:
        local["output_scaling_confidence"] = "medium"

    allowed_pi_keys = {
        f"Pi_{index + 1}" for index in range(int(basis_count))
    }
    for constraint_name in ("fixed_in_power", "fixed_out_power"):
        constraints = dict(local.get(constraint_name, {}) or {})
        invalid_keys = sorted(set(constraints) - allowed_pi_keys)
        if invalid_keys:
            raise ValueError(
                f"{constraint_name} contains unavailable base-Pi keys: "
                f"{invalid_keys}; allowed keys are "
                f"{sorted(allowed_pi_keys)}."
            )
        local[constraint_name] = {
            key: float(value)
            for key, value in constraints.items()
        }

    if local["fixed_out_power"] and local["similarity_type"] == "first_kind":
        raise ValueError(
            "fixed_out_power is incompatible with explicit first-kind similarity. "
            "For similarity_type='auto', output-side constraints are ignored during "
            "screening and restored only if Type II is selected."
        )

    # Resolve regime configuration ONCE: defaults < plan < nested < flat.
    gradient_manual = _normalize_gradient_regime_options(hp.get("gradient_regime_options"))
    clustering = _deep_update_dict(
        GRADIENT_REGIME_DEFAULT_OPTIONS["clustering"],
        dict(payload["regime_detection"]),
    )
    clustering = _deep_update_dict(clustering, gradient_manual["clustering"])
    flat_aliases = {
        "regime_n_clusters": "n_clusters",
        "regime_candidate_cluster_counts": "candidate_cluster_counts",
        "regime_min_cluster_fraction": "min_cluster_fraction",
        "regime_min_cluster_samples": "min_cluster_samples",
        "regime_max_intercluster_cosine": "max_intercluster_cosine",
        "regime_eigenvalue_ratio_threshold": "eigenvalue_ratio_threshold",
        "regime_require_convergence": "require_convergence",
        "regime_exclude_ambiguous": "exclude_low_confidence_points",
        "regime_assignment_confidence_threshold": "assignment_confidence_threshold",
    }
    resolved_manual = {}
    for flat_key, policy_key in flat_aliases.items():
        if flat_key in hp and (hp[flat_key] is not None
                              or policy_key in {"n_clusters", "max_intercluster_cosine"}):
            clustering[policy_key] = hp[flat_key]
            resolved_manual[policy_key] = hp[flat_key]
    surrogate = _deep_update_dict(
        GRADIENT_REGIME_DEFAULT_OPTIONS["surrogate"], gradient_manual["surrogate"]
    )
    shared_seed = hp.get("regime_random_state", hp.get("random_state"))
    if shared_seed is not None:
        if "random_state" not in gradient_manual["clustering"]:
            clustering["random_state"] = int(shared_seed)
        if "random_state" not in gradient_manual["surrogate"]:
            surrogate["random_state"] = int(shared_seed)
    clustering = _validate_regime_configuration(clustering, template)
    policy_names = set(_pydantic_dump(RegimeDetectionPlan()))
    regime = {key: value for key, value in clustering.items() if key in policy_names}
    resolved_gradient_options = {
        "surrogate": surrogate,
        "clustering": clustering,
    }

    tool_ids = [
        "calc_basis_tool",
        "run_local_dominant_group_workflow",
        "rationalize_local_solution",
    ]
    if template != "single_regime_adaptive":
        tool_ids.insert(1, "detect_gradient_regimes")
    unavailable_tools = [
        tool_id
        for tool_id in tool_ids
        if tool_id not in WORKFLOW_TOOL_REGISTRY
    ]
    if unavailable_tools:
        raise ValueError(
            f"Workflow references unavailable tools: {unavailable_tools}"
        )

    return {
        "status": "compiled",
        "template": template,
        "rationale": payload.get("rationale", ""),
        "target_is_dimensionless": bool(
            payload.get("target_is_dimensionless", True)
        ),
        "stages": list(WORKFLOW_TEMPLATE_STAGES[template]),
        "tool_ids": tool_ids,
        "regime_detection": regime,
        "resolved_gradient_regime_options": resolved_gradient_options,
        "regime_configuration_precedence": ["library_defaults", "workflow_plan", "nested_manual", "flat_manual"],
        "resolved_manual_regime_overrides": resolved_manual,
        "local_workflow": local,
        "replanning_triggers": payload.get(
            "replanning_triggers",
            [],
        ),
        "max_replans": int(hp.get("workflow_max_replans", 2)),
    }


def execute_compiled_workflow(
    compiled_workflow: dict,
    data_csv: str,
    input_vars: List[str],
    output_var: str,
    basis_matrices,
    manual_hyperparams=None,
) -> dict:
    """Execute one compiled macro workflow; runtime transitions are numerical."""
    from .regimes import _normalize_gradient_regime_options, _reject_removed_regime_controls, detect_gradient_regimes
    from .notebook import display_regime_detection_summary, display_workflow_stage
    from .discovery import optimize_regimes_with_mutual_information, run_local_workflow_with_similarity_resolution
    hp = dict(manual_hyperparams or {})
    _reject_removed_regime_controls(hp=hp)
    hp["_resolved_output_var"] = str(output_var)
    hp["_workflow_progress_stages"] = list(compiled_workflow.get("stages", []) or [])
    frame = pd.read_csv(data_csv)
    missing = [name for name in input_vars + [output_var] if name not in frame.columns]
    if missing:
        return {"status": "missing_dataset_columns", "missing": missing}

    data = frame[input_vars + [output_var]]
    values = data.to_numpy(dtype=float)
    valid_mask = (np.all(np.isfinite(values), axis=1) & np.all(np.abs(values) >= 1e-15, axis=1))
    clean = data.loc[valid_mask]
    X = clean[input_vars].to_numpy(dtype=float)
    Y = clean[[output_var]].to_numpy(dtype=float)
    original_indices = clean.index.to_numpy(dtype=int)

    if X.shape[0] == 0:
        return {"status": "empty_dataset_after_cleaning"}

    local_options = dict(compiled_workflow["local_workflow"])
    planned_mi_method = str(local_options.get("mi_method", "auto"))
    mi_method = (
        str(hp.get("mi_method_used"))
        if hp.get("mi_method_used") is not None
        else ("kraskov" if X.shape[0] < MI_BINNING_SAMPLE_THRESHOLD else "binning")
        if planned_mi_method == "auto"
        else planned_mi_method
    )
    planned_mi_param = local_options.get("mi_param")
    mi_param = int(
        hp.get("mi_param_used")
        if hp.get("mi_param_used") is not None
        else planned_mi_param
        if planned_mi_param is not None
        else 5 if mi_method == "kraskov"
        else 40
    )
    fixed_in_power = hp.get("fixed_in_power", local_options.get("fixed_in_power") or {})
    fixed_out_power = hp.get("fixed_out_power", local_options.get("fixed_out_power") or {})
    hp.setdefault("regime_cma_restarts", int(local_options.get("cma_restarts", 3)))
    optimize_output = local_options.get("optimize_output")
    template = compiled_workflow["template"]

    if template == "single_regime_adaptive":
        prescribed_single_summary = {
            "status": "not_requested", "best_k": 1, "selection_basis": "prescribed_single",
            "single_regime_baseline": {"diagnostic_status": "not_computed_prescribed_single"},
            "candidate_summary": [],
            "message": "Prescribed single-regime analysis: E and candidate K diagnostics were not computed.",
        }
        display_regime_detection_summary(prescribed_single_summary, hp)
        local_result = run_local_workflow_with_similarity_resolution(
            X=X,
            Y=Y,
            input_vars=input_vars,
            basis_matrices=basis_matrices,
            mi_method=mi_method,
            mi_param=mi_param,
            optimize_output=optimize_output,
            fixed_in_power=fixed_in_power,
            fixed_out_power=fixed_out_power,
            manual_hyperparams=hp,
            local_workflow_options=local_options,
            scope_id="global",
        )
        local_result["global_sample_indices"] = original_indices.tolist()
        return {
            "status": local_result.get("status"),
            "executed_template": template,
            "mechanism_decision": "single",
            "similarity_type": local_result.get(
                "similarity_type", local_options.get("similarity_type")
            ),
            "output_optimization": local_result.get(
                "optimize_output", optimize_output
            ),
            "similarity_screening": local_result.get("similarity_screening"),
            "regime_detection": prescribed_single_summary,
            "local_workflows": [local_result],
        }

    if not bool(compiled_workflow.get("target_is_dimensionless", True)):
        return {
            "status": "DIMENSIONLESS_TARGET_REQUIRED_FOR_GRADIENT_CLUSTERING",
            "executed_template": template,
            "mechanism_decision": "unresolved",
            "message": (
                "Gradient-regime detection requires the response used by the surrogate to be dimensionless. "
                "Provide a dimensionless target or use the single-regime workflow with explicit output scaling."
            ),
            "local_workflows": [],
        }

    regime_options = dict(compiled_workflow["regime_detection"])
    # Compilation resolved policy and numerical options; do not merge raw hp again.
    normalized_options = _normalize_gradient_regime_options(
        compiled_workflow["resolved_gradient_regime_options"]
    )
    clustering_options = normalized_options["clustering"]
    display_workflow_stage(
        "Mechanism screening",
        status="running",
        summary=(
            "Collect global diagnostics and all requested K evidence, then verify K=2 directional support."
            if template == "auto_mechanism_adaptive"
            else "Collect the explicitly requested partition evidence before selection."
        ),
        rows=[
            ("Candidate K", clustering_options.get("candidate_cluster_counts")),
            ("Fixed K", clustering_options.get("n_clusters")),
            ("Global E", "diagnostic only; no decision threshold"),
            ("Minimum fraction", clustering_options.get("min_cluster_fraction")),
            ("Minimum cluster samples", clustering_options.get("min_cluster_samples")),
            ("Auto multiple support requires K=2 |cos| <", clustering_options["max_intercluster_cosine"]),
        ],
        hyperparameters=hp,
    )
    detection = detect_gradient_regimes(
        X_physical=X,
        dimensionless_output=Y,
        basis_matrices=basis_matrices,
        max_clusters=max(clustering_options["candidate_cluster_counts"]),
        min_cluster_fraction=float(clustering_options["min_cluster_fraction"]),
        eigenvalue_ratio_threshold=float(clustering_options["eigenvalue_ratio_threshold"]),
        gradient_regime_options=normalized_options,
        workflow_template=template,
    )
    display_regime_detection_summary(detection, hp)

    if detection.get("status") == "global_analysis_selected":
        local_result = run_local_workflow_with_similarity_resolution(
            X=X,
            Y=Y,
            input_vars=input_vars,
            basis_matrices=basis_matrices,
            mi_method=mi_method,
            mi_param=mi_param,
            optimize_output=optimize_output,
            fixed_in_power=fixed_in_power,
            fixed_out_power=fixed_out_power,
            manual_hyperparams=hp,
            local_workflow_options=local_options,
            scope_id="global",
        )
        local_result["global_sample_indices"] = original_indices.tolist()
        return {
            "status": local_result.get("status"),
            "executed_template": template,
            "mechanism_decision": detection.get("selection_basis", "global_analysis_selected"),
            "similarity_type": local_result.get(
                "similarity_type", local_options.get("similarity_type")
            ),
            "output_optimization": local_result.get(
                "optimize_output", optimize_output
            ),
            "similarity_screening": local_result.get("similarity_screening"),
            "regime_detection": detection,
            "local_workflows": [local_result],
        }

    if detection.get("status") == "success":
        # Map clean-frame positions back to original CSV indices for reporting.
        detection_valid = np.asarray(detection.get("valid_indices", []), dtype=int)
        detection["original_csv_indices"] = original_indices[detection_valid].astype(int).tolist()
        local_options["exclude_low_confidence_points"] = bool(
            regime_options.get("exclude_low_confidence_points", False)
        )
        local_options["assignment_confidence_threshold"] = float(
            regime_options.get("assignment_confidence_threshold", 0.05)
        )
        regime_results = optimize_regimes_with_mutual_information(
            X=X,
            Y=Y,
            input_vars=input_vars,
            basis_matrices=basis_matrices,
            regime_detection=detection,
            mi_method=mi_method,
            mi_param=mi_param,
            optimize_output=optimize_output,
            fixed_in_power=fixed_in_power,
            fixed_out_power=fixed_out_power,
            manual_hyperparams=hp,
            local_workflow_options=local_options,
        )
        for local in regime_results.get("regimes", []):
            local_indices = np.asarray(local.get("global_sample_indices", []), dtype=int)
            if local_indices.size:
                local["original_csv_indices"] = original_indices[local_indices].astype(int).tolist()
        resolved_regimes = regime_results.get("regimes", [])
        successful_types = sorted({
            str(local.get("similarity_type"))
            for local in resolved_regimes
            if local.get("status") == "success" and local.get("similarity_type")
        })
        top_similarity = (
            successful_types[0]
            if len(successful_types) == 1
            else "mixed"
            if len(successful_types) > 1
            else local_options.get("similarity_type")
        )
        output_flags = [
            bool(local.get("optimize_output", False))
            for local in resolved_regimes
            if local.get("status") == "success"
        ]
        top_output_optimization = (
            output_flags[0]
            if output_flags and all(flag == output_flags[0] for flag in output_flags)
            else None
        )
        return {
            "status": regime_results.get("status"),
            "executed_template": template,
            "mechanism_decision": "multiple",
            "similarity_type": top_similarity,
            "output_optimization": top_output_optimization,
            "similarity_type_requested": local_options.get("similarity_type"),
            "regime_detection": detection,
            "local_workflows": resolved_regimes,
            "regime_workflow_summary": regime_results,
        }

    # Only an explicit selector outcome may choose the global route.
    # Numerical/data/surrogate failures remain unresolved for review/replanning.
    return {
        "status": "NO_ADMISSIBLE_REGIME_PARTITION"
        if detection.get("status") == "no_admissible_partition"
        else detection.get("status", "REGIME_DETECTION_FAILED"),
        "executed_template": template,
        "mechanism_decision": "multiple_requested_but_not_resolved"
        if template == "multi_regime_adaptive" else "automatic_assessment_unresolved",
        "regime_detection": detection,
        "local_workflows": [],
    }


def _normalize_variable_token(value: Any) -> str:
    """Normalize a variable label for deterministic alias matching."""
    text = str(value or "").strip()
    text = text.replace("*", "star").replace("+", "plus")
    text = text.replace("−", "-").replace("–", "-")
    return re.sub(r"[^0-9A-Za-z]+", "", text).lower()


def _is_obvious_metadata_column(name: str) -> bool:
    token = _normalize_variable_token(name)
    raw = str(name).strip().lower()
    return (
        raw.startswith("unnamed:")
        or token in {
            "index", "row", "rowid", "id", "sample", "sampleid",
            "case", "caseid", "group", "label",
        }
    )


def _match_requested_column(requested: Optional[str], columns: List[str], excluded: set[str]) -> Optional[str]:
    """Match an analyzer/manual variable name to a real CSV column."""
    if requested is None:
        return None
    requested = str(requested).strip().strip("`\"'")
    available = [column for column in columns if column not in excluded]
    if requested in available:
        return requested

    normalized = _normalize_variable_token(requested)
    exact_aliases = [
        column for column in available
        if _normalize_variable_token(column) == normalized
    ]
    if len(exact_aliases) == 1:
        return exact_aliases[0]

    # Conservative containment fallback for common spellings such as u* -> u_star.
    containment = [
        column for column in available
        if normalized
        and (
            normalized in _normalize_variable_token(column)
            or _normalize_variable_token(column) in normalized
        )
    ]
    if len(containment) == 1:
        return containment[0]
    return None


def _prompt_output_symbols(problem_prompt: str) -> List[str]:
    """Extract likely dependent-variable symbols from common prompt phrasings."""
    prompt = str(problem_prompt or "")
    patterns = [
        r"denoted\s+(?:as|by)\s+([^\s,;\)]+)",
        r"represented\s+(?:as|by)\s+([^\s,;\)]+)",
        r"记为\s*([^\s，。；、\)]+)",
        r"表示为\s*([^\s，。；、\)]+)",
    ]
    symbols = []
    for pattern in patterns:
        for match in re.findall(pattern, prompt, flags=re.IGNORECASE):
            symbol = str(match).strip().strip("`\"'()[]{}")
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def resolve_output_variable(
    columns: List[str],
    input_vars: List[str],
    analyzed_output_var: Optional[str],
    problem_prompt: str,
    manual_hyperparams: Optional[dict] = None,
) -> dict:
    """Resolve the dependent CSV column with explicit and reproducible precedence.

    Precedence:
      1. Explicit manual override, when supplied.
      2. Structured Analyzer inference.
      3. Symbol parsed deterministically from the problem prompt.
      4. The sole non-input, non-metadata CSV column.

    The function never silently chooses among multiple unresolved candidates.
    """
    hp = dict(manual_hyperparams or {})
    columns = [str(column) for column in columns]
    excluded = set(str(value) for value in input_vars)
    candidates = [
        column for column in columns
        if column not in excluded and not _is_obvious_metadata_column(column)
    ]

    explicit_key = next(
        (
            key for key in ("output_var", "target_var", "dependent_var")
            if hp.get(key) is not None
        ),
        None,
    )
    if explicit_key is not None:
        requested = hp[explicit_key]
        matched = _match_requested_column(requested, columns, excluded)
        if matched is None:
            return {
                "status": "invalid_manual_output_variable",
                "requested": str(requested),
                "source": f"manual_hyperparams.{explicit_key}",
                "candidates": candidates,
                "columns": columns,
            }
        return {
            "status": "success",
            "output_var": matched,
            "source": f"manual_hyperparams.{explicit_key}",
            "requested": str(requested),
            "candidates": candidates,
        }

    matched = _match_requested_column(analyzed_output_var, columns, excluded)
    if matched is not None:
        return {
            "status": "success",
            "output_var": matched,
            "source": "structured_analyzer",
            "requested": str(analyzed_output_var),
            "candidates": candidates,
        }

    for symbol in _prompt_output_symbols(problem_prompt):
        matched = _match_requested_column(symbol, columns, excluded)
        if matched is not None:
            return {
                "status": "success",
                "output_var": matched,
                "source": "deterministic_prompt_symbol",
                "requested": symbol,
                "candidates": candidates,
            }

    if len(candidates) == 1:
        return {
            "status": "success",
            "output_var": candidates[0],
            "source": "sole_non_input_column",
            "requested": None,
            "candidates": candidates,
        }

    return {
        "status": "output_variable_ambiguous",
        "analyzer_requested": analyzed_output_var,
        "prompt_symbols": _prompt_output_symbols(problem_prompt),
        "candidates": candidates,
        "columns": columns,
    }


def adaptive_basis_decider_node(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    from .notebook import display_workflow_stage
    display_workflow_stage(
        "Stage 3/7 · Buckingham–Pi basis",
        status="running",
        summary="The basis builder is calling calc_basis_tool with the audited dimensional matrix.",
        hyperparameters=state.get("manual_hyperparams", {}),
    )
    frame = pd.read_csv(state["csv_FilePath"])
    system_message = SystemMessage(content=(
        "You only construct the Buckingham-Pi basis. Call calc_basis_tool exactly once. "
        "Use only independent input variables; never include the target/output column in D_in. "
        "Read the dimensional matrix and variable order from the audited analysis."
    ))
    human_message = HumanMessage(content=(
        f"Problem:\n{state['prompt_problem']}\n\n"
        f"Audited dimensional descriptions:\n{state.get('dimension_detect', '')}\n\n"
        f"Audited dimensional matrix:\n{state.get('dimension_mat', '')}\n\n"
        f"Expected Pi count:\n{state.get('dimension_numK', '')}\n\n"
        f"Structured inferred output column (must be excluded): {state.get('analyzed_output_var')}\n"
        f"CSV columns: {frame.columns.tolist()}"
    ))
    response = llm_basis_builder.invoke([system_message, human_message])
    return {"basis_tool_messages": [response]}


def adaptive_basis_executor_node(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    from .basis import calc_basis_tool
    from .notebook import display_basis_summary
    messages = state.get("basis_tool_messages", [])
    if not messages:
        return {"workflow_status": "basis_tool_not_called"}
    last = messages[-1]
    calls = getattr(last, "tool_calls", [])
    if not calls:
        return {"workflow_status": "basis_tool_not_called"}

    history = []
    for call in calls:
        if call.get("name") != "calc_basis_tool":
            continue
        args = dict(call.get("args", {}))
        result = calc_basis_tool.invoke(args)
        history.append({"args": args, "result": result})
        if result.get("status") != "success":
            return {"basis_tool_history": history, "workflow_status": "basis_failed"}

        input_vars = list(args.get("var_names", []))
        frame = pd.read_csv(state["csv_FilePath"], nrows=1)
        hp = dict(state.get("manual_hyperparams", {}))
        output_resolution = resolve_output_variable(
            columns=frame.columns.tolist(),
            input_vars=input_vars,
            analyzed_output_var=state.get("analyzed_output_var"),
            problem_prompt=state.get("prompt_problem", ""),
            manual_hyperparams=hp,
        )
        if output_resolution.get("status") != "success":
            return {
                "basis_tool_history": history,
                "workflow_status": str(output_resolution.get("status")),
                "output_resolution": output_resolution,
                "output_candidates": output_resolution.get("candidates", []),
            }
        output_var = str(output_resolution["output_var"])
        if hp.get("_benchmark_case_id"):
            # Keep fixed-power constraints in the SI coordinate convention.
            from ..catalog import CASES
            benchmark = CASES[hp["_benchmark_case_id"]]
            expected_vars = benchmark["inputs"]
            if set(input_vars) != set(expected_vars):
                return {"basis_tool_history": history, "workflow_status": "benchmark_input_set_mismatch"}
            column_order = [input_vars.index(name) for name in expected_vars]
            audited_D = np.asarray(args["D_in"], dtype=float)[:, column_order]
            si_basis = np.asarray(benchmark["basis"], dtype=float)
            if (len(result["basis_matrices"]) != len(si_basis)
                    or not np.allclose(audited_D @ si_basis.T, 0, atol=1e-9)):
                return {"basis_tool_history": history, "workflow_status": "benchmark_dimensional_subspace_mismatch"}
            history[-1]["original_tool_result"] = dict(result)
            history[-1]["basis_alignment"] = "Audited null space verified; re-expressed in SI convention for fixed-power constraints."
            result = {**result, "basis_matrices": si_basis.tolist(),
                      "basis_matrices_display": si_basis.tolist(),
                      "pi_formulas": {f"Pi_{i+1}": label for i,label in enumerate(benchmark["basis_labels"])}}
            input_vars = list(expected_vars)
            history[-1]["result"] = result
        display_basis_summary(
            input_vars=input_vars,
            output_var=output_var,
            pi_formulas=result["pi_formulas"],
            output_resolution=output_resolution,
            hyperparameters=state.get("manual_hyperparams", {}),
        )
        return {
            "basis_tool_history": history,
            "basis_pi_formulas": result["pi_formulas"],
            "basis_pi_matrices": result["basis_matrices"],
            "input_vars": input_vars,
            "output_var": output_var,
            "output_resolution": output_resolution,
            "workflow_status": "basis_ready",
        }
    return {"basis_tool_history": history, "workflow_status": "basis_failed"}


def workflow_planner_node(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    from .information import _pydantic_dump
    from .notebook import display_compiled_plan
    frame = pd.read_csv(state["csv_FilePath"])
    summary = (
        f"rows={frame.shape[0]}, columns={frame.columns.tolist()}, "
        f"input_vars={state.get('input_vars')}, output_var={state.get('output_var')}, "
        f"output_resolution={state.get('output_resolution')}"
    )
    plan = create_workflow_plan(
        problem_prompt=state["prompt_problem"],
        dataset_summary=summary,
        basis_count=len(state.get("basis_pi_matrices") or []),
        basis_formulas=state.get("basis_pi_formulas", {}),
        manual_hyperparams=state.get("manual_hyperparams", {}),
    )
    compiled = compile_workflow_plan(
        plan,
        basis_count=len(state.get("basis_pi_matrices") or []),
        manual_hyperparams=state.get("manual_hyperparams", {}),
    )
    manual_hp = dict(state.get("manual_hyperparams", {}))
    _, provenance = _apply_manual_workflow_overrides(_pydantic_dump(plan), manual_hp)
    resolved_snapshot = {
        "output_variable": state.get("output_var"),
        "output_resolution": state.get("output_resolution"),
        "manual_hyperparams": manual_hp,
        "effective_workflow": compiled,
        "policy": "library defaults < LLM planner < manual_hyperparams",
    }
    display_compiled_plan(
        plan=_pydantic_dump(plan),
        compiled=compiled,
        provenance=provenance,
        hyperparameters=manual_hp,
    )
    return {
        "workflow_plan": _pydantic_dump(plan),
        "compiled_workflow": compiled,
        "resolved_hyperparameters": resolved_snapshot,
        "hyperparameter_provenance": provenance,
        "workflow_status": "workflow_compiled",
    }


def workflow_executor_node(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    from .notebook import display_workflow_stage
    display_workflow_stage(
        "Stage 5/7 · Numerical workflow execution",
        status="running",
        rows=[
            ("Template", (state.get("compiled_workflow") or {}).get("template")),
            ("Output", state.get("output_var")),
            ("Inputs", state.get("input_vars")),
        ],
        hyperparameters=state.get("manual_hyperparams", {}),
    )
    result = execute_compiled_workflow(
        compiled_workflow=state["compiled_workflow"],
        data_csv=state["csv_FilePath"],
        input_vars=list(state["input_vars"]),
        output_var=str(state["output_var"]),
        basis_matrices=state["basis_pi_matrices"],
        manual_hyperparams=state.get("manual_hyperparams", {}),
    )
    display_workflow_stage(
        "Stage 5/7 · Numerical workflow execution",
        status="completed" if str(result.get("status")) == "success" else "warning",
        rows=[
            ("Execution status", result.get("status")),
            ("Mechanism decision", result.get("mechanism_decision")),
            ("Local workflows", len(result.get("local_workflows", []) or [])),
        ],
        hyperparameters=state.get("manual_hyperparams", {}),
    )
    return {
        "workflow_result": result,
        "workflow_attempt_history": [result],
        "workflow_trace": [{
            "event": "WORKFLOW_EXECUTION_COMPLETE",
            "status": result.get("status"),
            "template": result.get("executed_template"),
            "mechanism_decision": result.get("mechanism_decision"),
            "local_workflow_count": len(result.get("local_workflows", []) or []),
        }],
        "workflow_status": str(result.get("status", "unknown")),
    }


def _workflow_failure_event(result: dict) -> Optional[str]:
    status = str(result.get("status", ""))
    direct = {
        "NO_ADMISSIBLE_REGIME_PARTITION",
        "DIMENSIONLESS_TARGET_REQUIRED_FOR_GRADIENT_CLUSTERING",
        "local_count_search_failed",
        "no_successful_local_workflow",
    }
    if status in direct:
        return status
    detection_status = str((result.get("regime_detection") or {}).get("status", ""))
    if detection_status == "gpr_failed":
        return "SURROGATE_FIT_FAILED"
    if any(local.get("status") == "insufficient_local_samples" for local in result.get("local_workflows", [])):
        return "LOCAL_SAMPLE_SIZE_INSUFFICIENT"
    return None


def workflow_replanner_node(state: AdaptiveWorkflowState):
    """Perform a bounded macro replan only after a recognized workflow failure."""
    from .schemas import AdaptiveWorkflowState, WorkflowPlan
    from .information import _pydantic_dump, _pydantic_validate
    from .notebook import display_workflow_stage
    display_workflow_stage(
        "Runtime replanning",
        status="warning",
        summary="A supported failure event was detected; the planner may adjust the macro template.",
        rows=[("Current replans", state.get("replan_count", 0))],
        hyperparameters=state.get("manual_hyperparams", {}),
    )
    hp = dict(state.get("manual_hyperparams", {}))
    current = dict(state.get("workflow_plan") or {})
    failure_event = _workflow_failure_event(state.get("workflow_result", {}))
    replan_count = int(state.get("replan_count", 0)) + 1

    # Retry the explicitly selected macro policy; a numerical failure is not
    # evidence for a single regime.  Only the evidence selector changes route.
    if not current:
        current = _pydantic_dump(_fallback_workflow_plan(state.get("prompt_problem", ""), hp))
    current, provenance = _apply_manual_workflow_overrides(current, hp)
    current["rationale"] = (
        str(current.get("rationale", ""))
        + f" Runtime retry {replan_count}: retained the requested policy after {failure_event}."
    )
    plan = _pydantic_validate(WorkflowPlan, current)

    compiled = compile_workflow_plan(
        plan,
        basis_count=len(state.get("basis_pi_matrices") or []),
        manual_hyperparams=hp,
    )
    resolved_snapshot = {
        "output_variable": state.get("output_var"),
        "output_resolution": state.get("output_resolution"),
        "manual_hyperparams": hp,
        "effective_workflow": compiled,
        "policy": "library defaults < LLM planner < manual_hyperparams",
    }
    return {
        "workflow_plan": _pydantic_dump(plan),
        "compiled_workflow": compiled,
        "resolved_hyperparameters": resolved_snapshot,
        "hyperparameter_provenance": provenance,
        "replan_count": replan_count,
        "workflow_status": "workflow_replanned",
    }


def route_after_workflow_execution(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    result = state.get("workflow_result", {})
    event = _workflow_failure_event(result)
    compiled = state.get("compiled_workflow", {})
    triggers = set(compiled.get("replanning_triggers", []))
    maximum = int(compiled.get("max_replans", 0))
    current = int(state.get("replan_count", 0))
    return "replan" if event and event in triggers and current < maximum else "report"


def workflow_report_node(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    from .notebook import display_final_workflow_summary
    from .knowledge import safe_retrieve_dimensionless_interpretation_evidence
    result = state.get("workflow_result", {})
    local_summaries = []
    for local in result.get("local_workflows", []):
        if local.get("status") != "success":
            local_summaries.append(
                f"- {local.get('scope_id', local.get('regime', 'scope'))}: {local.get('status')}"
            )
            continue
        final = local.get("final_solution", {})
        similarity_screen = local.get("similarity_screening") or {}
        screening_text = ""
        if similarity_screen.get("status") == "success":
            screening_text = (
                f" similarity_screen[max_output_exponent="
                f"{similarity_screen.get('max_abs_output_physical_exponent')}, "
                f"relative_mi_loss_without_scaling="
                f"{similarity_screen.get('relative_mi_loss_without_output_scaling')}];"
            )
        local_summaries.append(
            f"- {local.get('scope_id')}: locked k={local.get('locked_count')}; "
            f"similarity={local.get('similarity_type')}; "
            f"{screening_text} "
            f"physical_formulas={final.get('input_formulas')}; "
            f"basis_formulas={final.get('input_basis_expanded_formulas')}; "
            f"output_formula={final.get('output_dimensionless_formula')}; "
            f"score={final.get('normalized_information_score')}; "
            f"GPR R2={(final.get('gpr_validation') or {}).get('test_r2_mean', final.get('gpr_test_r2'))}; "
            f"sparsification={local.get('sparsification', {}).get('status')}; "
            f"rationalized={local.get('rationalization', {}).get('accepted')}"
        )
    deterministic_report = (
        f"Output variable: {state.get('output_var')}\n"
        f"Output resolution: {state.get('output_resolution')}\n"
        f"Workflow template: {result.get('executed_template')}\n"
        f"Mechanism decision: {result.get('mechanism_decision')}\n"
        f"Status: {result.get('status')}\n"
        + "\n".join(local_summaries)
    )

    hp = dict(state.get("manual_hyperparams", {}))
    if not hp.get("enable_llm_workflow_report", True):
        display_final_workflow_summary(state, deterministic_report, deterministic_report)
        return {"final_report": deterministic_report}

    # RAG is used here only to ground terminology/physical interpretation of
    # already-selected numerical relations. It cannot alter formulas, k, regimes,
    # similarity type, MI values, GPR validation, or any numerical decision.
    interpretation_knowledge_retrieval = safe_retrieve_dimensionless_interpretation_evidence(
        problem_description=state.get("prompt_problem", ""),
        deterministic_report=deterministic_report,
    )
    interpretation_evidence = interpretation_knowledge_retrieval.get(
        "evidence_markdown",
        "No external dimensionless-relation evidence is available.",
    )

    max_words = max(30, int(hp.get("workflow_report_max_words", 70)))
    try:
        response = chatgpt_model5.invoke([
            SystemMessage(content=(            
                "Retrieved documented dimensionless relations are grounding evidence for the physical interpretation. "
                "You must explicitly relate the discovered dimensionless relation to the most relevant retrieved "
                "knowledge-base record(s). Do not merely restate the deterministic result. "

                "If the discovered expression is algebraically equivalent to a retrieved defining expression, "
                "name that documented dimensionless quantity and briefly state its physical interpretation. "

                "If there is no exact algebraic identity, explicitly say that no exact classical identity is established, "
                "then use the closest retrieved relation(s) only as physical analogues or contextual references. "
                "Briefly explain what physical competition or mechanism the retrieved evidence suggests, while clearly "
                "distinguishing analogy from identity. "

                "Do not change any formula, locked count, regime decision, similarity decision, or validation value. "

                "Return exactly two concise bullets: "
                "(1) Physical interpretation of the discovered dominant relation and mechanism; "
                "(2) RAG grounding: the most relevant retrieved documented relation(s), whether the match is "
                "exact/equivalent or contextual only, and what physical meaning they contribute. "
                "Add the validation caveat briefly to either bullet when needed."
            )),
            HumanMessage(content=(
                f"Maximum words: {max_words}\n"
                f"Problem: {state['prompt_problem']}\n\n"
                f"Deterministic result summary:\n{deterministic_report}\n\n"
                f"Retrieved documented dimensionless relations for interpretation only:\n"
                f"{interpretation_evidence}"
            )),
        ])
        report_text = str(getattr(response, "content", str(response))).strip()
        words = report_text.split()
        if len(words) > max_words:
            report_text = " ".join(words[:max_words]).rstrip(" ,;:") + "…"
    except Exception:
        report_text = deterministic_report

    display_state = dict(state)
    display_state["interpretation_knowledge_retrieval"] = interpretation_knowledge_retrieval
    display_final_workflow_summary(display_state, deterministic_report, report_text)
    return {
        "final_report": report_text,
        "interpretation_knowledge_retrieval": interpretation_knowledge_retrieval,
    }


def route_after_basis(state: AdaptiveWorkflowState):
    from .schemas import AdaptiveWorkflowState
    return "ready" if state.get("workflow_status") == "basis_ready" else "failed"


def route_after_check(state: dict):
    decisions = state.get("hallucination_decision") or []
    return "retry" if decisions and decisions[-1] == "RETRY" else "passed"


def build_adaptive_agent_graph():
    """Build the default planner-driven, runtime-adaptive LangGraph agent."""
    from .schemas import AdaptiveWorkflowState
    from .figures import adaptive_visualizer_node
    initialize_models()
    builder = StateGraph(AdaptiveWorkflowState)
    builder.add_node("Analyzer", analyze_problem_node)
    builder.add_node("Hallu_Checker", physics_consistency_node)
    builder.add_node("Basis_Decider", adaptive_basis_decider_node)
    builder.add_node("Basis_Executor", adaptive_basis_executor_node)
    builder.add_node("Workflow_Planner", workflow_planner_node)
    builder.add_node("Workflow_Executor", workflow_executor_node)
    builder.add_node("Workflow_Replanner", workflow_replanner_node)
    builder.add_node("Workflow_Report", workflow_report_node)
    builder.add_node("Visualizer", adaptive_visualizer_node)

    builder.add_edge(START, "Analyzer")
    builder.add_edge("Analyzer", "Hallu_Checker")
    builder.add_conditional_edges(
        "Hallu_Checker",
        route_after_check,
        {"retry": "Analyzer", "passed": "Basis_Decider"},
    )
    builder.add_edge("Basis_Decider", "Basis_Executor")
    builder.add_conditional_edges(
        "Basis_Executor",
        route_after_basis,
        {"ready": "Workflow_Planner", "failed": END},
    )
    builder.add_edge("Workflow_Planner", "Workflow_Executor")
    builder.add_conditional_edges(
        "Workflow_Executor",
        route_after_workflow_execution,
        {"replan": "Workflow_Replanner", "report": "Workflow_Report"},
    )
    builder.add_edge("Workflow_Replanner", "Workflow_Executor")
    builder.add_edge("Workflow_Report", "Visualizer")
    builder.add_edge("Visualizer", END)
    return builder.compile()

def initialize_models():
    """Create clients only when the API-backed workflow is explicitly launched."""
    global chatgpt_model, chatgpt_model5, llm1_w_struOUT, llm4_w_struOUT_halluCheck, llm_workflow_planner, llm_basis_builder
    from .schemas import Buckingham_sp1, Hallucination_check_sp1, WorkflowPlan
    from .basis import calc_basis_tool
    chatgpt_model = setup_environment(os.getenv("ANALYZER_MODEL", "gpt-5-mini"))
    chatgpt_model5 = setup_environment(os.getenv("PLANNER_MODEL", "gpt-5-mini"))
    llm1_w_struOUT = chatgpt_model.with_structured_output(Buckingham_sp1, method="function_calling")
    llm4_w_struOUT_halluCheck = chatgpt_model5.with_structured_output(Hallucination_check_sp1, method="function_calling")
    llm_workflow_planner = chatgpt_model5.with_structured_output(WorkflowPlan, method="function_calling")
    llm_basis_builder = chatgpt_model5.bind_tools([calc_basis_tool])
