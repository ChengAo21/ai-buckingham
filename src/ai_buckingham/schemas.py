"""Schemas extracted from the author implementation."""
from __future__ import annotations

import re
from typing import Annotated, List, Literal, Dict, Optional
import operator
from fractions import Fraction
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from . import config as _config


class FrontendContractError(ValueError):
    pass


def rational_string(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
        raise ValueError("SI exponent must be a finite rational value or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("SI exponent must be finite")
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?\d+)?", text):
        raise ValueError("SI exponent must be a numeric rational, not an expression")
    try:
        f = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid SI rational exponent") from exc
    return str(f)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SIVector(StrictModel):
    M: str | None = Field(...)
    L: str | None = Field(...)
    T: str | None = Field(...)
    I: str | None = Field(...)
    Theta: str | None = Field(...)
    N: str | None = Field(...)
    J: str | None = Field(...)

    @field_validator(*_config.BASE_DIMENSIONS, mode="before")
    @classmethod
    def normalize_exponent(cls, value):
        return rational_string(value)


def _semantic_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Physical meaning must be nonempty text")
    text = value.strip()
    # Reject obvious assigned SI claims; do not rewrite/guess scientific meaning.
    patterns = [r"(?i)si[_ -]?(?:dimension|exponent|vector)|dimension_vector|si_exponents",
                r"(?i)\bdimensions?\s*(?:=|:|is\b|are\b)?\s*(?:M|L|T|I|Theta|K|N|J)\b",
                r"\b(?:M|L|T|I|Theta|K|N|J)\s*(?:\^|\*\*)",
                r"\[\s*[+-]?\d+(?:\s*,\s*[+-]?\d+){2,}\s*\]",
                r"\[\s*(?:M|L|T|I|Theta|K)(?:\s|\]|\^)"]
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError("Meaning/query fields must not contain assigned SI exponents or vectors")
    return text


class VariableDimension(StrictModel):
    csv_name: str = Field(min_length=1)
    quantity_name: str = Field(min_length=1)
    physical_definition: str = Field(min_length=1)
    si_exponents: SIVector

    @field_validator("csv_name")
    @classmethod
    def exact_name(cls, value):
        if value != value.strip():
            raise ValueError("CSV name must match exactly, including case; no surrounding whitespace")
        return value

    @field_validator("quantity_name", "physical_definition", mode="before")
    @classmethod
    def no_assigned_dimensions_in_query_text(cls, value):
        return _semantic_text(value)


class AnalyzerDimensionLedger(StrictModel):
    output_variable_name: str = Field(min_length=1)
    output_variable_reasoning: str = Field(min_length=1)
    output_variable_confidence: Literal["high", "medium", "low"]
    input_variable_names: list[str] = Field(min_length=1)
    variables: list[VariableDimension] = Field(min_length=1)


class OutputColumnSelection(StrictModel):
    column_index: int | None = Field(..., strict=True, ge=0)


class VariableAudit(StrictModel):
    csv_name: str = Field(min_length=1)
    before_si_exponents: SIVector
    decision: Literal["PASSED", "CONFLICT", "UNCERTAIN"]
    suggested_si_exponents: SIVector | None = Field(...)
    support_mode: Literal["KB_RECORD", "INTERNAL_REASONING", "UNCERTAIN"]
    evidence_entry_ids: list[str]
    reason: str = Field(min_length=1)


class CheckerDimensionAudit(StrictModel):
    ledger_sha256: str = Field(min_length=1)
    variables: list[VariableAudit] = Field(min_length=1)


def model_dict(value):
    return value.model_dump() if isinstance(value, BaseModel) else value


def validate_ledger(value):
    return AnalyzerDimensionLedger.model_validate(model_dict(value)).model_dump()


class RegimeDetectionPlan(BaseModel):
    """Validated regime-selection policy; numerical outcomes are runtime evidence."""

    model_config = ConfigDict(extra="forbid")

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
    absolute_score_drop_threshold: float = _config.ABS_SCORE_DROP_THRESHOLD
    require_two_counts_when_possible: bool = True
    similarity_type: Literal['auto', 'first_kind', 'second_kind'] = 'first_kind'
    optimize_output: Optional[bool] = None
    output_scaling_reason: str = 'First-kind similarity is the default/null policy unless output scaling or automatic similarity screening is explicitly requested.'
    output_scaling_confidence: Literal['high', 'medium', 'low'] = 'medium'
    similarity_screening_max_relative_mi_change: float = _config.SIMILARITY_MAX_RELATIVE_MI_CHANGE
    similarity_screening_max_cosine_similarity: float = _config.SIMILARITY_SCREEN_MAX_COSINE
    mi_method: Literal['auto', 'kraskov', 'binning'] = 'auto'
    mi_param: Optional[int] = None
    cma_restarts: int = 3
    minimum_samples: int = 20
    minimum_samples_per_mi_dimension: int = 20
    minimum_samples_for_second_count: int = 40
    fixed_in_power: Dict[str, float] = Field(default_factory=dict)
    fixed_out_power: Dict[str, float] = Field(default_factory=dict)
    sparsification_enabled: bool = True
    sparsify_exponent_threshold: float = _config.SPARSE_THRES
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

    @model_validator(mode="before")
    @classmethod
    def normalize_planner_structure(cls, value):
        """Preserve recognized constraints and normalize single-regime metadata."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        local = dict(payload.get("local_analysis") or {})
        for name in ("fixed_in_power", "fixed_out_power"):
            if name in payload:
                misplaced = payload.pop(name)
                if local.get(name) and local[name] != misplaced:
                    raise ValueError(f"Conflicting {name} at two plan levels")
                if not local.get(name):
                    local[name] = misplaced
        payload["local_analysis"] = local
        if payload.get("template") in ("single_regime_adaptive", "auto_mechanism_adaptive"):
            regime = dict(payload.get("regime_detection") or {})
            if regime.get("n_clusters") == 1:
                regime["n_clusters"] = None
            counts = regime.get("candidate_cluster_counts")
            if counts is not None and 1 in counts:
                regime["candidate_cluster_counts"] = [k for k in counts if k >= 2] or [2, 3, 4]
            payload["regime_detection"] = regime
        return payload

    template: _config.WorkflowTemplateName
    rationale: str
    target_is_dimensionless: bool = True
    stages: List[_config.WorkflowStageId] = Field(default_factory=list)
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
    analyzer_raw_ledger: Optional[dict]
    input_order_binding: Optional[dict]
    dimension_ledger: Optional[dict]
    dimension_ledger_sha256: Optional[str]
    dimension_contract: Optional[dict]
    dimension_audit: Optional[dict]
    dimension_audit_history: Annotated[list, operator.add]
    dimension_query: Optional[dict]
    frontend_rag_enabled: Optional[bool]
    frontend_expected_inputs: Optional[List[str]]
    frontend_expected_output: Optional[str]
    basis_request_ledger_sha256: Optional[str]
    basis_request: Optional[dict]

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
    run_provenance: Optional[dict]


def _pydantic_dump(model: BaseModel) -> dict:
    """Support compatible Pydantic APIs without version branching elsewhere."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _pydantic_validate(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)
