"""Workflow extracted from the author implementation."""
from __future__ import annotations

import re
import json
import hashlib
from io import BytesIO
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
import math
import sympy as sp
from . import config as _config
from . import dimensions as _dimensions
from . import knowledge as _knowledge
from . import models as _models
from . import optimization as _optimization
from . import presentation as _presentation
from . import regimes as _regimes
from . import schemas as _schemas
from . import validation as _validation


def _infer_workflow_template_from_context(
    problem_prompt: str,
    hp: dict,
) -> str:
    """Infer the canonical workflow template; single regime is the default."""
    explicit = hp.get("workflow_template")
    if explicit in _config.WORKFLOW_TEMPLATE_STAGES:
        return str(explicit)

    prompt = str(problem_prompt).lower()
    auto_markers = (
        "automatic mechanism screening",
        "automatically detect regimes",
        "auto-detect regimes",
        '\u81ea\u52a8\u63a2\u6d4b\u673a\u5236',
        '\u81ea\u52a8\u8bc6\u522b\u673a\u5236',
        '\u81ea\u52a8\u805a\u7c7b\u673a\u5236',
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
        '\u4e0d\u540c\u673a\u5236',
        '\u591a\u4e2a\u673a\u5236',
        '\u591a\u673a\u5236',
        '\u4e0d\u540c\u533a\u95f4',
        '\u591a\u4e2a\u9636\u6bb5',
        '\u4e0d\u540c\u9636\u6bb5',
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
        '\u81ea\u52a8\u5224\u65ad\u76f8\u4f3c\u6027\u7c7b\u578b',
        '\u81ea\u52a8\u8bc6\u522b\u76f8\u4f3c\u6027\u7c7b\u578b',
        '\u81ea\u52a8\u5224\u65ad\u7b2c\u4e00\u7c7b\u548c\u7b2c\u4e8c\u7c7b\u76f8\u4f3c\u6027',
        '\u81ea\u52a8\u5224\u65ad\u7b2c\u4e00\u7c7b\u6216\u7b2c\u4e8c\u7c7b\u76f8\u4f3c\u6027',
        '\u81ea\u52a8\u8bc6\u522b\u7b2c\u4e00\u7c7b\u548c\u7b2c\u4e8c\u7c7b\u76f8\u4f3c\u6027',
        '\u81ea\u52a8\u76f8\u4f3c\u6027\u7b5b\u9009',
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
        '\u4e0d\u4f18\u5316\u8f93\u51fa',
        '\u65e0\u9700\u4f18\u5316\u8f93\u51fa',
        '\u8f93\u51fa\u4e0d\u7f29\u653e',
        '\u7b2c\u4e00\u7c7b\u76f8\u4f3c\u6027',
    )
    complete_similarity_requested = bool(
        re.search(r"(?<!in)complete similarity", prompt)
        or re.search('(?<!\u4e0d)\u5b8c\u5168\u76f8\u4f3c', prompt)
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
        '\u8f93\u51fa\u53d8\u91cf\u9700\u8981\u7f29\u653e',
        '\u9700\u8981\u4f18\u5316\u8f93\u51fa',
        '\u8f93\u51fa\u7f29\u653e',
        '\u7b2c\u4e8c\u7c7b\u76f8\u4f3c\u6027',
        '\u4e0d\u5b8c\u5168\u76f8\u4f3c',
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


def _fallback_workflow_plan(problem_prompt: str, hp: dict) -> _schemas.WorkflowPlan:
    """Construct a deterministic plan when the planner call is unavailable."""
    template = _infer_workflow_template_from_context(problem_prompt, hp)
    output_policy = _infer_output_optimization_from_context(problem_prompt, hp)
    local_payload = {'start_count': int(hp.get('workflow_start_count', 1)), 'max_count': hp.get('workflow_max_count'), 'absolute_score_drop_threshold': float(hp.get('abs_score_drop_threshold', _config.ABS_SCORE_DROP_THRESHOLD)), 'require_two_counts_when_possible': bool(hp.get('workflow_require_two_counts_when_possible', True)), 'similarity_type': output_policy['similarity_type'], 'optimize_output': output_policy['optimize_output'], 'output_scaling_reason': output_policy['reason'], 'output_scaling_confidence': output_policy['confidence'], 'similarity_screening_max_relative_mi_change': float(hp.get('similarity_screening_max_relative_mi_change', _config.SIMILARITY_MAX_RELATIVE_MI_CHANGE)), 'similarity_screening_max_cosine_similarity': float(hp.get('similarity_screening_max_cosine_similarity', _config.SIMILARITY_SCREEN_MAX_COSINE)), 'mi_method': hp.get('mi_method_used', 'auto'), 'mi_param': hp.get('mi_param_used'), 'cma_restarts': int(_optimization._cma_hyperparameter(hp, 'cma_restarts', 3)), 'minimum_samples': int(hp.get('regime_min_samples', 20)), 'minimum_samples_per_mi_dimension': int(hp.get('regime_min_samples_per_mi_dimension', 20)), 'minimum_samples_for_second_count': int(hp.get('workflow_minimum_samples_for_second_count', 40)), 'fixed_in_power': hp.get('fixed_in_power', {}) or {}, 'fixed_out_power': hp.get('fixed_out_power', {}) or {}, 'sparsification_enabled': bool(hp.get('workflow_sparsification_enabled', True)), 'sparsify_exponent_threshold': float(hp.get('regime_sparsify_exponent_threshold', _config.SPARSE_THRES)), 'sparsify_max_score_increase': float(hp.get('regime_sparsify_max_score_increase', 0.05)), 'rationalization_enabled': bool(hp.get('rationalized_on', True)), 'max_rational_denominator': int(hp.get('max_rational_denominator', 5))}
    _regimes._reject_removed_regime_controls(hp=hp)
    regime_payload = {}
    return _schemas.WorkflowPlan(template=template, rationale='Deterministic fallback selected from explicit settings and problem wording.', target_is_dimensionless=bool(hp.get('target_is_dimensionless', True)), stages=_config.WORKFLOW_TEMPLATE_STAGES[template], regime_detection=_schemas.RegimeDetectionPlan(**regime_payload), local_analysis=_schemas.LocalDominantWorkflowPlan(**local_payload))


def _apply_manual_workflow_overrides(payload: dict, hp: dict) -> tuple[dict, dict]:
    """Apply authoritative manual settings after planner suggestions."""
    _regimes._reject_removed_regime_controls(hp=hp, plan=payload)
    if hp.get('workflow_template') is not None and hp['workflow_template'] not in _config.WORKFLOW_TEMPLATE_STAGES:
        raise ValueError('Unknown explicit workflow_template: ' + str(hp['workflow_template']))
    result = _validation._deep_update_dict({}, payload)
    provenance = {'precedence': ['library_defaults', 'llm_workflow_plan', 'manual_hyperparams'], 'manual_overrides': {}, 'passthrough_manual_hyperparameters': {}}

    def record(path: str, value: Any) -> None:
        provenance['manual_overrides'][path] = value
    if hp.get('workflow_template') is not None:
        result['template'] = _infer_workflow_template_from_context('', hp)
        record('template', result['template'])
    if 'target_is_dimensionless' in hp:
        result['target_is_dimensionless'] = bool(hp['target_is_dimensionless'])
        record('target_is_dimensionless', result['target_is_dimensionless'])
    local = dict(result.get('local_analysis', {}))
    local_nested = {}
    if isinstance(hp.get('local_workflow_options'), dict):
        local_nested = _validation._deep_update_dict(local_nested, hp['local_workflow_options'])
    gradient_nested = hp.get('gradient_regime_options') or {}
    if isinstance(gradient_nested, dict) and isinstance(gradient_nested.get('local_workflow'), dict):
        local_nested = _validation._deep_update_dict(local_nested, gradient_nested['local_workflow'])
    local = _validation._deep_update_dict(local, local_nested)
    for key, value in local_nested.items():
        record(f'local_analysis.{key}', value)
    local_flat_mapping = {'workflow_start_count': 'start_count', 'workflow_max_count': 'max_count', 'abs_score_drop_threshold': 'absolute_score_drop_threshold', 'mi_method_used': 'mi_method', 'mi_param_used': 'mi_param', 'fixed_in_power': 'fixed_in_power', 'fixed_out_power': 'fixed_out_power', 'workflow_sparsification_enabled': 'sparsification_enabled', 'regime_sparsify_exponent_threshold': 'sparsify_exponent_threshold', 'regime_sparsify_max_score_increase': 'sparsify_max_score_increase', 'rationalized_on': 'rationalization_enabled', 'max_rational_denominator': 'max_rational_denominator', 'validate_after_rationalization': 'validate_after_rationalization', 'regime_min_samples': 'minimum_samples', 'regime_min_samples_per_mi_dimension': 'minimum_samples_per_mi_dimension', 'workflow_minimum_samples_for_second_count': 'minimum_samples_for_second_count', 'workflow_require_two_counts_when_possible': 'require_two_counts_when_possible', 'similarity_screening_max_relative_mi_change': 'similarity_screening_max_relative_mi_change', 'similarity_screening_max_cosine_similarity': 'similarity_screening_max_cosine_similarity'}
    for hp_key, plan_key in local_flat_mapping.items():
        if hp_key in hp and hp[hp_key] is not None:
            local[plan_key] = hp[hp_key]
            record(f'local_analysis.{plan_key}', hp[hp_key])
    cma_restarts = _optimization._cma_hyperparameter(hp, 'cma_restarts', None)
    if cma_restarts is not None:
        local['cma_restarts'] = int(cma_restarts)
        record('local_analysis.cma_restarts', local['cma_restarts'])
    if 'optimize_output' in hp:
        local['optimize_output'] = bool(hp['optimize_output'])
        local['similarity_type'] = 'second_kind' if local['optimize_output'] else 'first_kind'
        local['output_scaling_reason'] = 'Authoritative manual_hyperparams override.'
        local['output_scaling_confidence'] = 'high'
        record('local_analysis.optimize_output', local['optimize_output'])
        record('local_analysis.similarity_type', local['similarity_type'])
    elif hp.get('similarity_type') in {'auto', 'first_kind', 'second_kind'}:
        local['similarity_type'] = str(hp['similarity_type'])
        local['optimize_output'] = None if local['similarity_type'] == 'auto' else local['similarity_type'] == 'second_kind'
        local['output_scaling_reason'] = "Authoritative manual similarity_type='auto'; runtime numerical screening required." if local['similarity_type'] == 'auto' else 'Authoritative manual similarity_type override.'
        local['output_scaling_confidence'] = 'medium' if local['similarity_type'] == 'auto' else 'high'
        record('local_analysis.similarity_type', local['similarity_type'])
        record('local_analysis.optimize_output', local['optimize_output'])
    if local.get('fixed_out_power') and 'optimize_output' not in hp and (hp.get('similarity_type') is None) and (str(local.get('similarity_type', 'first_kind')) != 'auto'):
        local['optimize_output'] = True
        local['similarity_type'] = 'second_kind'
        local['output_scaling_reason'] = 'A fixed_out_power constraint requires output optimization.'
        local['output_scaling_confidence'] = 'high'
        record('local_analysis.optimize_output', True)
        record('local_analysis.similarity_type', 'second_kind')
    if hp.get('output_scaling_reason') is not None:
        local['output_scaling_reason'] = str(hp['output_scaling_reason'])
        record('local_analysis.output_scaling_reason', local['output_scaling_reason'])
    result['local_analysis'] = local
    regime_flat_mapping = {'regime_n_clusters': 'n_clusters', 'regime_candidate_cluster_counts': 'candidate_cluster_counts', 'regime_min_cluster_fraction': 'min_cluster_fraction', 'regime_min_cluster_samples': 'min_cluster_samples', 'regime_max_intercluster_cosine': 'max_intercluster_cosine', 'regime_eigenvalue_ratio_threshold': 'eigenvalue_ratio_threshold', 'regime_require_convergence': 'require_convergence', 'regime_exclude_ambiguous': 'exclude_low_confidence_points', 'regime_assignment_confidence_threshold': 'assignment_confidence_threshold'}
    consumed = {'workflow_plan', 'workflow_template', 'target_is_dimensionless', 'local_workflow_options', 'gradient_regime_options', 'regime_cma_restarts', 'cma_restarts', 'optimize_output', 'similarity_type', 'output_scaling_reason', *local_flat_mapping.keys(), *regime_flat_mapping.keys()}
    provenance['passthrough_manual_hyperparameters'] = {key: value for key, value in hp.items() if key not in consumed}
    result['stages'] = _config.WORKFLOW_TEMPLATE_STAGES[result['template']]
    return (result, provenance)


def _validate_explicit_power_constraints(problem_prompt: str, payload: dict) -> None:
    """Do not execute an under-constrained plan after a failed LLM response."""
    local = payload.get("local_analysis") or {}
    for side, field in (("input", "fixed_in_power"), ("output", "fixed_out_power")):
        requested = re.search(
            rf"(?:power|exponent).{{0,180}}constrain.{{0,180}}\b{side}\b",
            str(problem_prompt), flags=re.IGNORECASE | re.DOTALL,
        )
        if requested and not local.get(field):
            raise ValueError(f"Explicit {side} exponent constraint is missing from {field}")


def create_workflow_plan(
    problem_prompt: str,
    dataset_summary: str,
    basis_count: int,
    basis_formulas: Optional[Dict[str, str]] = None,
    manual_hyperparams: Optional[dict] = None,
) -> _schemas.WorkflowPlan:
    """Ask the LLM for a macro protocol, then apply manual overrides."""
    hp = dict(manual_hyperparams or {})
    _regimes._reject_removed_regime_controls(hp=hp)
    if hp.get("workflow_template") is not None and hp["workflow_template"] not in _config.WORKFLOW_TEMPLATE_STAGES:
        raise ValueError("Unknown explicit workflow_template: " + str(hp["workflow_template"]))
    supplied_plan = hp.get("workflow_plan")

    if supplied_plan:
        plan = _schemas._pydantic_validate(_schemas.WorkflowPlan, supplied_plan)
    else:
        _models._require_model_ready(_models.planner_llm, "Planner")
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
            "- Put both fixed-power dictionaries inside local_analysis and retain every explicit exponent constraint from the problem.\n"
            "- For a single mechanism use regime_detection.n_clusters=null, never 1; candidate counts must be >=2.\n"
            "- A dimensionless output may still require output optimization under incomplete similarity; "
            "target_is_dimensionless and optimize_output are independent decisions.\n"
            "- first_kind is the default/null similarity policy when neither output scaling nor automatic "
            "similarity screening is explicitly requested.\n"
            "- second_kind is used when incomplete similarity or dependent-variable scaling is explicitly requested.\n"
            "- auto is used only when automatic Type-I/Type-II similarity screening is explicitly requested; "
            "set optimize_output=null in this mode.\n"
            f"- Prefer kraskov for sample size <= {_config.MI_BINNING_SAMPLE_THRESHOLD} and binning otherwise; select mi_param accordingly to sample size and estimator.\n"
        ))
        plan = None
        feedback = []
        for attempt in range(max(1, int(hp.get("workflow_planner_max_attempts", 3)))):
            try:
                candidate = _models._require_model_ready(_models.planner_llm, "Planner").invoke(
                    [system_message, human_message, *feedback]
                )
                effective, _ = _apply_manual_workflow_overrides(_schemas._pydantic_dump(candidate), hp)
                _validate_explicit_power_constraints(problem_prompt, effective)
                plan = candidate
                break
            except Exception as exc:
                feedback = [HumanMessage(content=(
                    f"The previous plan failed validation: {type(exc).__name__}: {exc}. "
                    "Return a corrected plan. Keep explicit fixed powers inside local_analysis "
                    "and use null instead of n_clusters=1."
                ))]
        if plan is None:
            plan = _fallback_workflow_plan(problem_prompt, hp)

    # Match the mechanism-policy precedence exactly: library/schema defaults <
    # LLM planner < manual_hyperparams. Deterministic context inference is used
    # only by _fallback_workflow_plan; normal planner output is not sanitized or
    # overwritten here.
    payload, _ = _apply_manual_workflow_overrides(
        _schemas._pydantic_dump(plan),
        hp,
    )
    _validate_explicit_power_constraints(problem_prompt, payload)
    return _schemas._pydantic_validate(_schemas.WorkflowPlan, payload)


def compile_workflow_plan(plan: _schemas.WorkflowPlan, basis_count: int, manual_hyperparams=None) -> dict:
    """Validate planner output and compile it into an executable policy."""
    hp = dict(manual_hyperparams or {})
    payload = _schemas._pydantic_dump(plan)
    _regimes._reject_removed_regime_controls(hp=hp, plan=payload)
    template = payload['template']
    if template not in _config.WORKFLOW_TEMPLATE_STAGES:
        raise ValueError(f'Unknown workflow template: {template}')
    local = dict(payload['local_analysis'])
    local['start_count'] = max(1, int(local.get('start_count', 1)))
    if local.get('max_count') is None:
        local['max_count'] = int(basis_count)
    else:
        local['max_count'] = max(local['start_count'], min(int(local['max_count']), int(basis_count)))
    local['absolute_score_drop_threshold'] = max(0.0, float(local['absolute_score_drop_threshold']))
    local['cma_restarts'] = max(1, int(_optimization._cma_hyperparameter(hp, 'cma_restarts', local.get('cma_restarts', 3))))
    local['minimum_samples'] = max(10, int(local.get('minimum_samples', 20)))
    local['minimum_samples_per_mi_dimension'] = max(5, int(local.get('minimum_samples_per_mi_dimension', 20)))
    local['minimum_samples_for_second_count'] = max(10, int(local.get('minimum_samples_for_second_count', 40)))
    similarity_type = str(local.get('similarity_type', 'first_kind'))
    if similarity_type not in {'auto', 'first_kind', 'second_kind'}:
        raise ValueError(f'Unknown similarity_type: {similarity_type}')
    local['similarity_type'] = similarity_type
    local['optimize_output'] = None if similarity_type == 'auto' else similarity_type == 'second_kind'
    local['similarity_screening_max_relative_mi_change'] = max(0.0, float(local.get('similarity_screening_max_relative_mi_change', _config.SIMILARITY_MAX_RELATIVE_MI_CHANGE)))
    local['similarity_screening_max_cosine_similarity'] = min(1.0, max(0.0, float(local.get('similarity_screening_max_cosine_similarity', _config.SIMILARITY_SCREEN_MAX_COSINE))))
    if not str(local.get('output_scaling_reason', '')).strip():
        local['output_scaling_reason'] = 'Similarity class will be resolved by numerical screening.' if similarity_type == 'auto' else 'Dependent-variable scaling enabled.' if local['optimize_output'] else 'Dependent-variable scaling not requested.'
    if local.get('output_scaling_confidence') not in {'high', 'medium', 'low'}:
        local['output_scaling_confidence'] = 'medium'
    allowed_pi_keys = {f'Pi_{index + 1}' for index in range(int(basis_count))}
    for constraint_name in ('fixed_in_power', 'fixed_out_power'):
        constraints = dict(local.get(constraint_name, {}) or {})
        invalid_keys = sorted(set(constraints) - allowed_pi_keys)
        if invalid_keys:
            raise ValueError(f'{constraint_name} contains unavailable base-Pi keys: {invalid_keys}; allowed keys are {sorted(allowed_pi_keys)}.')
        local[constraint_name] = {key: float(value) for key, value in constraints.items()}
    if local['fixed_out_power'] and local['similarity_type'] == 'first_kind':
        raise ValueError("fixed_out_power is incompatible with explicit first-kind similarity. For similarity_type='auto', output-side constraints are ignored during screening and restored only if Type II is selected.")
    gradient_manual = _regimes._normalize_gradient_regime_options(hp.get('gradient_regime_options'))
    clustering = _validation._deep_update_dict(_config.GRADIENT_REGIME_DEFAULT_OPTIONS['clustering'], dict(payload['regime_detection']))
    clustering = _validation._deep_update_dict(clustering, gradient_manual['clustering'])
    flat_aliases = {'regime_n_clusters': 'n_clusters', 'regime_candidate_cluster_counts': 'candidate_cluster_counts', 'regime_min_cluster_fraction': 'min_cluster_fraction', 'regime_min_cluster_samples': 'min_cluster_samples', 'regime_max_intercluster_cosine': 'max_intercluster_cosine', 'regime_eigenvalue_ratio_threshold': 'eigenvalue_ratio_threshold', 'regime_require_convergence': 'require_convergence', 'regime_exclude_ambiguous': 'exclude_low_confidence_points', 'regime_assignment_confidence_threshold': 'assignment_confidence_threshold'}
    resolved_manual = {}
    for flat_key, policy_key in flat_aliases.items():
        if flat_key in hp and (hp[flat_key] is not None or policy_key in {'n_clusters', 'max_intercluster_cosine'}):
            clustering[policy_key] = hp[flat_key]
            resolved_manual[policy_key] = hp[flat_key]
    surrogate = _validation._deep_update_dict(_config.GRADIENT_REGIME_DEFAULT_OPTIONS['surrogate'], gradient_manual['surrogate'])
    shared_seed = hp.get('regime_random_state', hp.get('random_state'))
    if shared_seed is not None:
        if 'random_state' not in gradient_manual['clustering']:
            clustering['random_state'] = int(shared_seed)
        if 'random_state' not in gradient_manual['surrogate']:
            surrogate['random_state'] = int(shared_seed)
    clustering = _regimes._validate_regime_configuration(clustering, template)
    policy_names = set(_schemas._pydantic_dump(_schemas.RegimeDetectionPlan()))
    regime = {key: value for key, value in clustering.items() if key in policy_names}
    resolved_gradient_options = {'surrogate': surrogate, 'clustering': clustering}
    tool_ids = ['calc_basis_tool', 'run_local_dominant_group_workflow', 'rationalize_local_solution']
    if template != 'single_regime_adaptive':
        tool_ids.insert(1, 'detect_gradient_regimes')
    unavailable_tools = [tool_id for tool_id in tool_ids if tool_id not in WORKFLOW_TOOL_REGISTRY]
    if unavailable_tools:
        raise ValueError(f'Workflow references unavailable tools: {unavailable_tools}')
    return {'status': 'compiled', 'template': template, 'rationale': payload.get('rationale', ''), 'target_is_dimensionless': bool(payload.get('target_is_dimensionless', True)), 'stages': list(_config.WORKFLOW_TEMPLATE_STAGES[template]), 'tool_ids': tool_ids, 'regime_detection': regime, 'resolved_gradient_regime_options': resolved_gradient_options, 'regime_configuration_precedence': ['library_defaults', 'workflow_plan', 'nested_manual', 'flat_manual'], 'resolved_manual_regime_overrides': resolved_manual, 'local_workflow': local, 'replanning_triggers': payload.get('replanning_triggers', []), 'max_replans': int(hp.get('workflow_max_replans', 2))}


def _resolve_dimensionless_input_preprocessing(config, input_vars, dimension_contract) -> dict:
    """Validate explicit transformations against the audited input dimensions."""
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("dimensionless_input_preprocessing must be a mapping of input columns to transforms.")
    if not config:
        return {}

    contract = dimension_contract or {}
    audited_names = list(contract.get("var_names") or [])
    exact_dimensions = contract.get("D_exact") or []
    if (
        contract.get("status") != "complete"
        or len(audited_names) != len(input_vars)
        or set(audited_names) != set(input_vars)
        or not exact_dimensions
        or any(len(row) != len(audited_names) for row in exact_dimensions)
    ):
        raise ValueError("dimensionless_input_preprocessing requires a complete dimension contract for the selected inputs.")
    column_indices = {name: index for index, name in enumerate(audited_names)}
    resolved = {}
    for name, spec in config.items():
        if name not in column_indices:
            raise ValueError(f"{name!r} is not a selected input column.")
        column = column_indices[name]
        if any(sp.Rational(row[column]) != 0 for row in exact_dimensions):
            raise ValueError(f"Input column {name!r} must be dimensionless to use preprocessing.")
        if not isinstance(spec, dict) or spec.get("transform") not in ("shift", "exp"):
            raise ValueError(f"Input column {name!r} requires a 'shift' or 'exp' transform.")
        if spec["transform"] == "exp" and set(spec) == {"transform"}:
            resolved[name] = {"transform": "exp"}
            continue
        if spec["transform"] == "shift" and set(spec) == {"transform", "offset"}:
            offset = spec["offset"]
            if isinstance(offset, (int, float, np.integer, np.floating)) and not isinstance(offset, bool):
                try:
                    offset = float(offset)
                except OverflowError:
                    offset = math.inf
                if math.isfinite(offset):
                    resolved[name] = {"transform": "shift", "offset": offset}
                    continue
        raise ValueError(f"Invalid preprocessing options for input {name!r}. 'shift' requires a finite numeric offset; 'exp' takes no options.")
    return resolved


def _apply_dimensionless_input_preprocessing(frame: pd.DataFrame, preprocessing: dict) -> pd.DataFrame:
    """Transform configured inputs on a working copy, preserving missing values and row positions."""
    if not preprocessing:
        return frame
    numerical_data = frame.copy()
    for name, spec in preprocessing.items():
        values = frame[name].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(values)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if spec["transform"] == "shift":
                values[finite] += spec["offset"]
            else:
                values[finite] = np.exp(values[finite])
        invalid = finite & (~np.isfinite(values) | (values <= 0.0))
        if invalid.any():
            rows = frame.index[invalid].tolist()
            raise ValueError(
                f"Preprocessing {name!r} must yield finite, strictly positive values; "
                f"invalid zero-based CSV row positions: {rows[:10]} ({len(rows)} total)."
            )
        numerical_data[name] = values
    return numerical_data


def _format_preprocessed_input_terms(preprocessing: dict) -> dict:
    """Use the physical input expression as the base of each displayed power."""
    formula_terms = {}
    for name, spec in preprocessing.items():
        if spec["transform"] == "exp":
            formula_terms[name] = f"exp({name})"
        else:
            offset = spec["offset"]
            sign = "-" if offset < 0 else "+"
            magnitude = repr(abs(offset)).removesuffix(".0")
            formula_terms[name] = f"({name} {sign} {magnitude})"
    return formula_terms


def execute_compiled_workflow(
    compiled_workflow: dict,
    data_csv: str,
    input_vars: List[str],
    output_var: str,
    basis_matrices,
    manual_hyperparams=None,
    *,
    dimension_contract=None,
) -> dict:
    """Prepare one set of numerical inputs for the workflow and its plots."""
    hp = dict(manual_hyperparams or {})
    _regimes._reject_removed_regime_controls(hp=hp)
    gpr_options = _validation.resolve_gpr_validation_options(hp)
    hp['_resolved_gpr_validation_options'] = gpr_options
    hp['_resolved_output_var'] = str(output_var)
    hp['_workflow_progress_stages'] = list(compiled_workflow.get('stages', []) or [])
    hp.pop('_preprocessed_input_formula_terms', None)
    preprocessing = _resolve_dimensionless_input_preprocessing(
        hp.get('dimensionless_input_preprocessing'), input_vars, dimension_contract
    )
    if preprocessing:
        source_bytes = Path(data_csv).read_bytes()
        frame = pd.read_csv(BytesIO(source_bytes), float_precision="round_trip")
        hp['_preprocessed_input_formula_terms'] = _format_preprocessed_input_terms(preprocessing)
    else:
        frame = pd.read_csv(data_csv)
    missing = [name for name in input_vars + [output_var] if name not in frame.columns]
    if missing:
        return {'status': 'missing_dataset_columns', 'missing': missing}
    hp.pop('_gpr_fixed_split', None)
    if sum((hp.get('gpr_split') is not None, hp.get('gpr_fixed_split') is not None,
            gpr_options['gpr_stratify_column'] is not None)) > 1:
        raise ValueError("Use only one GPR split configuration.")
    if hp.get('gpr_split') is not None:
        hp['_gpr_fixed_split'] = _validation._resolve_gpr_validation_split(frame, hp['gpr_split'])
    elif hp.get('gpr_fixed_split') is not None:
        hp['_gpr_fixed_split'] = _validation._resolve_gpr_fixed_split(frame, hp['gpr_fixed_split'])
    elif gpr_options['gpr_stratify_column'] is not None:
        hp['_gpr_fixed_split'] = _validation._resolve_gpr_validation_split(
            frame, stratify_column=gpr_options['gpr_stratify_column'],
            test_size=gpr_options['gpr_test_size'],
            random_state=gpr_options['gpr_random_state'],
        )
    numerical_data = _apply_dimensionless_input_preprocessing(frame[input_vars + [output_var]], preprocessing)
    result = _execute_numerical_workflow(compiled_workflow, numerical_data, input_vars, output_var, basis_matrices, hp)
    result['gpr_validation_options'] = dict(gpr_options)
    if preprocessing:
        result['dimensionless_input_preprocessing'] = preprocessing
        result['preprocessed_input_snapshot'] = {
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'columns': {name: numerical_data[name].tolist() for name in preprocessing},
        }
    return result


def _execute_numerical_workflow(compiled_workflow, numerical_data, input_vars, output_var, basis_matrices, hp) -> dict:
    """Execute one compiled macro workflow using prepared inputs."""
    values = numerical_data.to_numpy(dtype=float)
    template = compiled_workflow['template']
    use_unfiltered_regime_rows = False
    if template != 'single_regime_adaptive':
        normalized_options = _regimes._normalize_gradient_regime_options(compiled_workflow.get('resolved_gradient_regime_options'))
        clustering_options = normalized_options['clustering']
        use_unfiltered_regime_rows = bool(_regimes._signed_gradient_mode(clustering_options)) and bool(
            np.any(np.isfinite(values[:, :-1]) & (values[:, :-1] <= 0.0))
        )
    valid_mask = _optimization._power_product_sample_mask(values[:, :-1], values[:, -1])
    if hp.get('_gpr_fixed_split') is not None and not np.all(valid_mask):
        raise ValueError("A fixed GPR split requires every published sample; numerical filtering would change it.")
    mi_samples = numerical_data.loc[valid_mask]
    X = mi_samples[input_vars].to_numpy(dtype=float)
    Y = mi_samples[[output_var]].to_numpy(dtype=float)
    original_indices = mi_samples.index.to_numpy(dtype=int)
    if X.shape[0] == 0 and not use_unfiltered_regime_rows:
        return {'status': 'empty_dataset_after_cleaning'}
    local_options = dict(compiled_workflow['local_workflow'])
    planned_mi_method = str(local_options.get('mi_method', 'auto'))
    mi_method = str(hp.get('mi_method_used')) if hp.get('mi_method_used') is not None else ('kraskov' if X.shape[0] < _config.MI_BINNING_SAMPLE_THRESHOLD else 'binning') if planned_mi_method == 'auto' else planned_mi_method
    planned_mi_param = local_options.get('mi_param')
    mi_param = int(hp.get('mi_param_used') if hp.get('mi_param_used') is not None else planned_mi_param if planned_mi_param is not None else 5 if mi_method == 'kraskov' else 40)
    fixed_in_power = hp.get('fixed_in_power', local_options.get('fixed_in_power') or {})
    fixed_out_power = hp.get('fixed_out_power', local_options.get('fixed_out_power') or {})
    optimize_output = local_options.get('optimize_output')
    if template == 'single_regime_adaptive':
        prescribed_single_summary = {'status': 'not_requested', 'best_k': 1, 'selection_basis': 'prescribed_single', 'single_regime_baseline': {'diagnostic_status': 'not_computed_prescribed_single'}, 'candidate_summary': [], 'message': 'Prescribed single-regime analysis: E and candidate K diagnostics were not computed.'}
        _presentation.display_regime_detection_summary(prescribed_single_summary, hp)
        local_result = _optimization.run_local_workflow_with_similarity_resolution(X=X, Y=Y, input_vars=input_vars, basis_matrices=basis_matrices, mi_method=mi_method, mi_param=mi_param, optimize_output=optimize_output, fixed_in_power=fixed_in_power, fixed_out_power=fixed_out_power, manual_hyperparams=hp, local_workflow_options=local_options, scope_id='global')
        local_result['global_sample_indices'] = original_indices.tolist()
        return {'status': local_result.get('status'), 'executed_template': template, 'mechanism_decision': 'single', 'similarity_type': local_result.get('similarity_type', local_options.get('similarity_type')), 'output_optimization': local_result.get('optimize_output', optimize_output), 'similarity_screening': local_result.get('similarity_screening'), 'regime_detection': prescribed_single_summary, 'local_workflows': [local_result]}
    if not bool(compiled_workflow.get('target_is_dimensionless', True)):
        return {'status': 'DIMENSIONLESS_TARGET_REQUIRED_FOR_GRADIENT_CLUSTERING', 'executed_template': template, 'mechanism_decision': 'unresolved', 'message': 'Gradient-regime detection requires the response used by the surrogate to be dimensionless. Provide a dimensionless target or use the single-regime workflow with explicit output scaling.', 'local_workflows': []}
    regime_options = dict(compiled_workflow['regime_detection'])
    mechanism_display = _presentation._display_mechanism_running(clustering_options, template, hp)
    detection_X = values[:, :-1] if use_unfiltered_regime_rows else X
    detection_Y = values[:, -1] if use_unfiltered_regime_rows else Y
    detection = _regimes.detect_gradient_regimes(X_physical=detection_X, dimensionless_output=detection_Y, basis_matrices=basis_matrices, max_clusters=max(clustering_options['candidate_cluster_counts']), min_cluster_fraction=float(clustering_options['min_cluster_fraction']), eigenvalue_ratio_threshold=float(clustering_options['eigenvalue_ratio_threshold']), gradient_regime_options=normalized_options, workflow_template=template)
    _presentation.display_regime_detection_summary(detection, hp, display_handle=mechanism_display)
    if detection.get('status') == 'global_analysis_selected':
        local_result = _optimization.run_local_workflow_with_similarity_resolution(X=X, Y=Y, input_vars=input_vars, basis_matrices=basis_matrices, mi_method=mi_method, mi_param=mi_param, optimize_output=optimize_output, fixed_in_power=fixed_in_power, fixed_out_power=fixed_out_power, manual_hyperparams=hp, local_workflow_options=local_options, scope_id='global')
        local_result['global_sample_indices'] = original_indices.tolist()
        return {'status': local_result.get('status'), 'executed_template': template, 'mechanism_decision': detection.get('selection_basis', 'global_analysis_selected'), 'similarity_type': local_result.get('similarity_type', local_options.get('similarity_type')), 'output_optimization': local_result.get('optimize_output', optimize_output), 'similarity_screening': local_result.get('similarity_screening'), 'regime_detection': detection, 'local_workflows': [local_result]}
    if detection.get('status') == 'success':
        detection_valid = np.asarray(detection.get('valid_indices', []), dtype=int)
        if use_unfiltered_regime_rows:
            detection['original_csv_indices'] = detection_valid.tolist()
            # Align clustering labels with the finite rows used by MI.
            keep = valid_mask[detection_valid]
            mi_detection = {
                **detection,
                'valid_indices': np.searchsorted(original_indices, detection_valid[keep]).tolist(),
                'labels': np.asarray(detection['labels'])[keep].tolist(),
                'assignment_confidence': np.asarray(detection['assignment_confidence'])[keep].tolist(),
                'mi_expected_regime_ids': sorted(set(detection['labels'])),
            }
            detection['excluded_from_mi_samples'] = int(np.count_nonzero(~keep))
            detection['mi_samples_by_regime'] = {
                int(label): int(np.count_nonzero(keep & (np.asarray(detection['labels']) == label)))
                for label in set(detection['labels'])
            }
        else:
            detection['original_csv_indices'] = original_indices[detection_valid].astype(int).tolist()
            mi_detection = detection
        local_options['exclude_low_confidence_points'] = bool(regime_options.get('exclude_low_confidence_points', False))
        local_options['assignment_confidence_threshold'] = float(regime_options.get('assignment_confidence_threshold', 0.05))
        regime_results = _optimization.optimize_regimes_with_mutual_information(X=X, Y=Y, input_vars=input_vars, basis_matrices=basis_matrices, regime_detection=mi_detection, mi_method=mi_method, mi_param=mi_param, optimize_output=optimize_output, fixed_in_power=fixed_in_power, fixed_out_power=fixed_out_power, manual_hyperparams=hp, local_workflow_options=local_options)
        for local in regime_results.get('regimes', []):
            local_indices = np.asarray(local.get('global_sample_indices', []), dtype=int)
            if local_indices.size:
                local['original_csv_indices'] = original_indices[local_indices].astype(int).tolist()
        resolved_regimes = regime_results.get('regimes', [])
        successful_types = sorted({str(local.get('similarity_type')) for local in resolved_regimes if local.get('status') == 'success' and local.get('similarity_type')})
        top_similarity = successful_types[0] if len(successful_types) == 1 else 'mixed' if len(successful_types) > 1 else local_options.get('similarity_type')
        output_flags = [bool(local.get('optimize_output', False)) for local in resolved_regimes if local.get('status') == 'success']
        top_output_optimization = output_flags[0] if output_flags and all((flag == output_flags[0] for flag in output_flags)) else None
        return {'status': regime_results.get('status'), 'executed_template': template, 'mechanism_decision': 'multiple', 'similarity_type': top_similarity, 'output_optimization': top_output_optimization, 'similarity_type_requested': local_options.get('similarity_type'), 'regime_detection': detection, 'local_workflows': resolved_regimes, 'regime_workflow_summary': regime_results}
    return {'status': 'NO_ADMISSIBLE_REGIME_PARTITION' if detection.get('status') == 'no_admissible_partition' else detection.get('status', 'REGIME_DETECTION_FAILED'), 'executed_template': template, 'mechanism_decision': 'multiple_requested_but_not_resolved' if template == 'multi_regime_adaptive' else 'automatic_assessment_unresolved', 'regime_detection': detection, 'local_workflows': []}


def workflow_planner_node(state: _schemas.AdaptiveWorkflowState):
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
    _, provenance = _apply_manual_workflow_overrides(_schemas._pydantic_dump(plan), manual_hp)
    resolved_snapshot = {
        "output_variable": state.get("output_var"),
        "output_resolution": state.get("output_resolution"),
        "manual_hyperparams": manual_hp,
        "effective_workflow": compiled,
        "policy": "library defaults < LLM planner < manual_hyperparams",
    }
    _presentation.display_compiled_plan(
        plan=_schemas._pydantic_dump(plan),
        compiled=compiled,
        provenance=provenance,
        hyperparameters=manual_hp,
    )
    return {
        "workflow_plan": _schemas._pydantic_dump(plan),
        "compiled_workflow": compiled,
        "resolved_hyperparameters": resolved_snapshot,
        "hyperparameter_provenance": provenance,
        "workflow_status": "workflow_compiled",
    }


def workflow_executor_node(state: _schemas.AdaptiveWorkflowState):
    _presentation.display_workflow_stage(
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
        dimension_contract=state.get("dimension_contract"),
    )
    _presentation.display_workflow_stage(
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


def workflow_replanner_node(state: _schemas.AdaptiveWorkflowState):
    """Perform a bounded macro replan only after a recognized workflow failure."""
    _presentation.display_workflow_stage(
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
        current = _schemas._pydantic_dump(_fallback_workflow_plan(state.get("prompt_problem", ""), hp))
    current, provenance = _apply_manual_workflow_overrides(current, hp)
    current["rationale"] = (
        str(current.get("rationale", ""))
        + f" Runtime retry {replan_count}: retained the requested policy after {failure_event}."
    )
    plan = _schemas._pydantic_validate(_schemas.WorkflowPlan, current)

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
        "workflow_plan": _schemas._pydantic_dump(plan),
        "compiled_workflow": compiled,
        "resolved_hyperparameters": resolved_snapshot,
        "hyperparameter_provenance": provenance,
        "replan_count": replan_count,
        "workflow_status": "workflow_replanned",
    }


def route_after_workflow_execution(state: _schemas.AdaptiveWorkflowState):
    result = state.get("workflow_result", {})
    event = _workflow_failure_event(result)
    compiled = state.get("compiled_workflow", {})
    triggers = set(compiled.get("replanning_triggers", []))
    maximum = int(compiled.get("max_replans", 0))
    current = int(state.get("replan_count", 0))
    return "replan" if event and event in triggers and current < maximum else "report"


def workflow_report_node(state: _schemas.AdaptiveWorkflowState):
    result = state.get('workflow_result', {})
    local_summaries = []
    for local in result.get('local_workflows', []):
        if local.get('status') != 'success':
            local_summaries.append(f"- {local.get('scope_id', local.get('regime', 'scope'))}: {local.get('status')}")
            continue
        final = local.get('final_solution', {})
        similarity_screen = local.get('similarity_screening') or {}
        screening_text = ''
        if similarity_screen.get('status') == 'success':
            screening_text = f" similarity_screen[relative_mi_loss_without_scaling={similarity_screen.get('relative_mi_loss_without_output_scaling')}];"
        local_summaries.append(f"- {local.get('scope_id')}: locked k={local.get('locked_count')}; similarity={local.get('similarity_type')}; {screening_text} physical_formulas={final.get('input_formulas')}; basis_formulas={final.get('input_basis_expanded_formulas')}; output_formula={final.get('output_dimensionless_formula')}; score={final.get('normalized_information_score')}; GPR R2={(final.get('gpr_validation') or {}).get('test_r2_mean', final.get('gpr_test_r2'))}; sparsification={local.get('sparsification', {}).get('status')}; rationalized={local.get('rationalization', {}).get('accepted')}")
    deterministic_report = f"Output variable: {state.get('output_var')}\nOutput resolution: {state.get('output_resolution')}\nWorkflow template: {result.get('executed_template')}\nMechanism decision: {result.get('mechanism_decision')}\nStatus: {result.get('status')}\n" + '\n'.join(local_summaries)
    preprocessing = result.get('dimensionless_input_preprocessing', {})
    if preprocessing:
        deterministic_report += f"\nDimensionless input preprocessing: {_format_preprocessed_input_terms(preprocessing)}"
    hp = dict(state.get('manual_hyperparams', {}))
    if not hp.get('enable_llm_workflow_report', True):
        _presentation.display_final_workflow_summary(state, deterministic_report, deterministic_report)
        return {'final_report': deterministic_report}
    interpretation_knowledge_retrieval = _knowledge.safe_retrieve_dimensionless_interpretation_evidence(problem_description=state.get('prompt_problem', ''), deterministic_report=deterministic_report)
    interpretation_evidence = interpretation_knowledge_retrieval.get('evidence_markdown', 'No external dimensionless-relation evidence is available.')
    max_words = max(30, int(hp.get('workflow_report_max_words', 70)))
    try:
        response = _models._require_model_ready(_models.interpretation_llm, 'interpretation').invoke([SystemMessage(content='Retrieved documented dimensionless relations are grounding evidence for the physical interpretation. You must explicitly relate the discovered dimensionless relation to the most relevant retrieved knowledge-base record(s). Do not merely restate the deterministic result. If the discovered expression is algebraically equivalent to a retrieved defining expression, name that documented dimensionless quantity and briefly state its physical interpretation. If there is no exact algebraic identity, explicitly say that no exact classical identity is established, then use the closest retrieved relation(s) only as physical analogues or contextual references. Briefly explain what physical competition or mechanism the retrieved evidence suggests, while clearly distinguishing analogy from identity. Do not change any formula, locked count, regime decision, similarity decision, or validation value. Return exactly two concise bullets: (1) Physical interpretation of the discovered dominant relation and mechanism; (2) RAG grounding: the most relevant retrieved documented relation(s), whether the match is exact/equivalent or contextual only, and what physical meaning they contribute. Add the validation caveat briefly to either bullet when needed.'), HumanMessage(content=f"Maximum words: {max_words}\nProblem: {state['prompt_problem']}\n\nDeterministic result summary:\n{deterministic_report}\n\nRetrieved documented dimensionless relations for interpretation only:\n{interpretation_evidence}")])
        report_text = str(getattr(response, 'content', str(response))).strip()
        words = report_text.split()
    except Exception:
        report_text = deterministic_report
    display_state = dict(state)
    display_state['interpretation_knowledge_retrieval'] = interpretation_knowledge_retrieval
    _presentation.display_final_workflow_summary(display_state, deterministic_report, report_text)
    return {'final_report': report_text, 'interpretation_knowledge_retrieval': interpretation_knowledge_retrieval}


from collections.abc import Mapping as _Mapping

def _workflow_tools():
    return {
        "calc_basis_tool": _dimensions.calc_basis_tool,
        "detect_gradient_regimes": _regimes.detect_gradient_regimes,
        "run_local_dominant_group_workflow": _optimization.run_local_dominant_group_workflow,
        "rationalize_local_solution": _optimization._rationalize_local_solution,
    }

class _WorkflowToolRegistry(_Mapping):
    """Resolve callable values lazily, avoiding import-order coupling."""
    def __getitem__(self, key):
        return _workflow_tools()[key]
    def __iter__(self):
        return iter(("calc_basis_tool", "detect_gradient_regimes", "run_local_dominant_group_workflow", "rationalize_local_solution"))
    def __len__(self):
        return 4

WORKFLOW_TOOL_REGISTRY = _WorkflowToolRegistry()
