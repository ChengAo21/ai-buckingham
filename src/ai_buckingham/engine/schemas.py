"""Validated dimensional-analysis and workflow contracts."""
from __future__ import annotations
from .shared import *


class Buckingham_sp1(BaseModel):
    output_variable_name: str = Field(
        description=(
            "The exact CSV column name of the dependent/output variable inferred from the problem statement and CSV preview. "
            "Preserve capitalization and symbols exactly. This variable must not appear in the dimensional matrix."
        )
    )
    output_variable_reasoning: str = Field(
        description=(
            "Briefly explain why this CSV column is the requested dependent variable and distinguish it from similarly named inputs."
        )
    )
    output_variable_confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence in the exact output-column identification."
    )
    input_variable_names: List[str] = Field(
        default_factory=list,
        description=(
            "Exact CSV column names of the independent physical quantities included in the dimensional matrix, in matrix-column order."
        )
    )
    dimension_detect: str = Field(
        description=
        "Concise descriptions of each input variable, its physical meanings, and its fundamental dimensions.")
    dimension_mat: str = Field(
        description=
        "A structured table of formal dimensional matrix."
        "Rows: dimensions RELEVANT to the specific problem (e.g., L, M, T; add I for electromagnetism, K for thermal, etc. as needed)."
        "Columns: physical variables.")
    dimension_numK: str = Field(
        description=
        "State the number of variables (n) and the rank of the dimensional matrix (r), calculate the expected number of base Pi groups (k = n - r).")


class Hallucination_check_sp1(BaseModel):
    check_reasoning: str = Field(
        description="Checklist:\n"
        "1. Is the target (output) variable excluded from the dimensional matrix D? If the output appears anywhere in D, this check FAILS."
        "2. Cross-reference the retrieved dimensional-knowledge evidence to verify that all dimensions are strictly correct in standard SI base units."
        "3. Is the count of base dimensionless groups (k = n - r) correctly inferred?")
    decision: Literal["PASSED", "RETRY"] = Field(
        description="PASSED: all checks passed; RETRY: has errors/hallucination.")
    correction_msg: str = Field(
        description="If RETRY, list specific errors. If PASSED, leave empty.")
    knowledge_evidence_status: Dict[
        str, Literal["SUPPORTED", "CONFLICT", "NOT_FOUND", "AMBIGUOUS"]
    ] = Field(
        default_factory=dict,
        description=(
            "For each audited input variable, classify the retrieved external evidence. "
            "NOT_FOUND or AMBIGUOUS alone must not trigger RETRY."
        ),
    )
    knowledge_evidence_reasoning: str = Field(
        default="",
        description=(
            "Concise explanation of how retrieved dimensional evidence was used. "
            "Do not claim KB support for NOT_FOUND variables."
        ),
    )


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


class RegimeDetectionPlan(BaseModel):
    """Validated regime-selection policy; numerical outcomes are runtime evidence."""

    class Config:
        extra = "forbid"

    n_clusters: Optional[int] = Field(default=None, ge=2, strict=True)
    candidate_cluster_counts: List[int] = Field(default_factory=lambda: [2, 3, 4])
    min_cluster_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    min_cluster_samples: int = Field(default=60, ge=1, strict=True)
    max_intercluster_cosine: float = Field(
        default=0.90, gt=0.0, le=1.0, strict=True
    )
    eigenvalue_ratio_threshold: float = Field(default=50.0, gt=0.0)
    require_convergence: bool = True
    exclude_low_confidence_points: bool = False
    assignment_confidence_threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class LocalDominantWorkflowPlan(BaseModel):
    """Planner policy for one single-mechanism data subset."""

    start_count: int = 1
    max_count: Optional[int] = None
    absolute_score_drop_threshold: float = ABS_SCORE_DROP_THRESHOLD
    require_two_counts_when_possible: bool = True

    # Similarity policy mirrors the regime policy:
    # first_kind is the default/null policy; second_kind is explicit output
    # scaling; auto is requested only when numerical Type-I/Type-II screening
    # is desired.
    similarity_type: Literal["auto", "first_kind", "second_kind"] = "first_kind"
    # None means the similarity policy has not yet been numerically resolved
    # (only valid for similarity_type="auto"). Compilation canonicalizes this
    # to False for first_kind, True for second_kind, and None for auto.
    optimize_output: Optional[bool] = None
    output_scaling_reason: str = (
        "First-kind similarity is the default/null policy unless output scaling "
        "or automatic similarity screening is explicitly requested."
    )
    output_scaling_confidence: Literal["high", "medium", "low"] = "medium"
    similarity_screening_output_exponent_threshold: float = (
        SIMILARITY_OUTPUT_EXPONENT_THRESHOLD
    )
    similarity_screening_max_relative_mi_change: float = (
        SIMILARITY_MAX_RELATIVE_MI_CHANGE
    )
    similarity_screening_max_cosine_similarity: float = (
        SIMILARITY_SCREEN_MAX_COSINE
    )

    mi_method: Literal["auto", "kraskov", "binning"] = "auto"
    mi_param: Optional[int] = None
    cma_restarts: int = 3
    minimum_samples: int = 20
    minimum_samples_per_mi_dimension: int = 20
    minimum_samples_for_second_count: int = 40

    fixed_in_power: Dict[str, float] = Field(default_factory=dict)
    fixed_out_power: Dict[str, float] = Field(default_factory=dict)

    sparsification_enabled: bool = True
    sparsify_exponent_threshold: float = SPARSE_THRES
    sparsify_max_score_increase: float = 0.05
    rationalization_enabled: bool = True
    max_rational_denominator: int = 5
    validate_after_rationalization: bool = True


class WorkflowPlan(BaseModel):
    """A macro policy graph selected by the LLM planner.

    The planner chooses a template and parameter policy.  It does not decide
    numerical outcomes such as the final number of dominant Pi groups, which
    remain runtime decisions made from MI-score trends and validation results.
    """

    template: WorkflowTemplateName
    rationale: str
    target_is_dimensionless: bool = True
    stages: List[WorkflowStageId] = Field(default_factory=list)
    regime_detection: RegimeDetectionPlan = Field(default_factory=RegimeDetectionPlan)
    local_analysis: LocalDominantWorkflowPlan = Field(default_factory=LocalDominantWorkflowPlan)
    replanning_triggers: List[str] = Field(default_factory=lambda: [
        "SURROGATE_FIT_FAILED",
        "NO_ADMISSIBLE_REGIME_PARTITION",
        "LOCAL_SAMPLE_SIZE_INSUFFICIENT",
        "OPTIMIZATION_UNSTABLE",
        "RATIONALIZATION_VALIDATION_FAILED",
        "NO_ADMISSIBLE_REGIME_PARTITION",
        "DIMENSIONLESS_TARGET_REQUIRED_FOR_GRADIENT_CLUSTERING",
        "local_count_search_failed",
        "no_successful_local_workflow",
    ])


class AdaptiveWorkflowState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    prompt_problem: str
    csv_FilePath: str
    manual_hyperparams: dict

    problem_description: str
    dimension_detect: str
    dimension_mat: str
    dimension_numK: str
    analyzed_output_var: Optional[str]
    analyzed_output_reasoning: Optional[str]
    analyzed_output_confidence: Optional[str]
    analyzed_input_vars: Optional[List[str]]
    hallucination_decision: Annotated[list, operator.add]
    hallucination_check: Annotated[list, operator.add]
    knowledge_retrieval: Optional[dict]

    basis_tool_messages: Annotated[list, add_messages]
    basis_tool_history: Annotated[list, operator.add]
    basis_pi_formulas: Optional[Dict[str, str]]
    basis_pi_matrices: Optional[List[List[float]]]
    input_vars: Optional[List[str]]
    output_var: Optional[str]
    output_resolution: Optional[dict]

    workflow_plan: Optional[dict]
    resolved_hyperparameters: Optional[dict]
    hyperparameter_provenance: Optional[dict]
    compiled_workflow: Optional[dict]
    workflow_result: Optional[dict]
    workflow_attempt_history: Annotated[list, operator.add]
    workflow_trace: Annotated[list, operator.add]
    replan_count: int
    final_report: Optional[str]
    interpretation_knowledge_retrieval: Optional[dict]
    visualization_files: Optional[List[str]]
    workflow_status: Optional[str]
