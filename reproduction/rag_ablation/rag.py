"""AI Buckingham RAG module for physical-quantity retrieval and dimensional checking.

The supplied agent source has SHA-256 59ed37f40af4791040e6aac079d467347837d593c5132e0975860dbee0c964dd.
Source identities and scientific-definition comparisons are in reference/source_identity.json.
Defaults point to the reproduction data. Replay injects a score source.
The optional embedding adapter loads the OpenAI SDK on demand and omits
the full agent's HTTP recorder.
"""

import ast
import hashlib
import json
import math
import os
import re
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIRECTORY = PACKAGE_DIR / "reference"
CACHE_DIR = PACKAGE_DIR / "results" / "cache"
PHYSICAL_KNOWLEDGE_BASE_FILEPATH = str(DATA_DIRECTORY / "Data_csv" / "dimensional_physical_quantities.md")
KNOWLEDGE_INDEX_FILEPATH = str(
    Path(os.getenv("DIMAGENT_KNOWLEDGE_INDEX_FILEPATH", str(DATA_DIRECTORY / "physical_quantity_index.npz"))).resolve()
)



KNOWLEDGE_EMBEDDING_MODEL = "GLM-Embedding-3"


PARATERA_DEFAULT_BASE_URL = "https://llmapi.paratera.com/v1"
PARATERA_BASE_URL = os.getenv("PARATERA_BASE_URL", PARATERA_DEFAULT_BASE_URL)


PARATERA_API_KEY_ENV = "PARATERA_API_KEY"  # Set this environment variable


KNOWLEDGE_TOP_K_PER_VARIABLE = 3


KNOWLEDGE_MAX_EVIDENCE_ROWS = 30


KNOWLEDGE_LEXICAL_WEIGHT = 0.65


KNOWLEDGE_SEMANTIC_WEIGHT = 0.35


KNOWLEDGE_MIN_HYBRID_SCORE = 0.22


KNOWLEDGE_ALLOW_LEXICAL_FALLBACK = False


CANONICAL_DIMENSION_ORDER = ("M", "L", "T", "I", "Theta", "N", "J")


PHYSICAL_QUANTITY_REQUIRED_COLUMNS = (
    "entry_id",
    "canonical_quantity",
    "symbols",
    "aliases",
    "definition",
    "si_dimension",
    "dimension_vector",
    "si_unit",
    "domain",
    "ambiguity_notes",
    "source",
    "source_identifier",
    "confidence",
)


_GREEK_NORMALIZATION = {
    "μ": "mu", "ν": "nu", "ρ": "rho", "η": "eta", "σ": "sigma",
    "κ": "kappa", "λ": "lambda", "θ": "theta", "Θ": "theta",
    "ε": "epsilon", "α": "alpha", "β": "beta", "γ": "gamma",
}


_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "with", "variable", "quantity",
    "physical", "input", "parameter",
}


_DIMENSIONAL_KB_RETRIEVER = None


BASE_DIMENSIONS = ("M", "L", "T", "I", "Theta", "N", "J")


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

    @field_validator(*BASE_DIMENSIONS, mode="before")
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


def ledger_sha256(value):
    ledger = validate_ledger(value)
    return hashlib.sha256(json.dumps(ledger, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def render_retrieval_interpretation(ledger):
    data = validate_ledger(ledger)
    return {v["csv_name"]: f"exact_column: {v['csv_name']}; quantity_name: {v['quantity_name']}; physical_definition: {v['physical_definition']}" for v in data["variables"]}


def format_evidence_records(matches, max_rows=30):
    lines, visible = [], {name: [] for name in matches}
    count = 0
    for name, records in matches.items():
        lines.append(f"### Input variable `{name}`")
        if not records:
            lines.append("No sufficiently relevant record found; absence alone is not a contradiction.")
        for record in records:
            if count >= max_rows:
                break
            count += 1
            visible[name].append(str(record.get("entry_id", "")))
            lines.append("- " + "; ".join(f"{key}: {record.get(key, '')}" for key in (
                "entry_id", "canonical_quantity", "symbols", "aliases", "definition", "si_dimension", "dimension_vector", "si_unit", "domain", "ambiguity_notes", "source", "source_identifier", "confidence")))
    return "\n".join(lines), visible


def record_vector(record):
    value = record.get("dimension_vector")
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise FrontendContractError("Referenced KB vector must contain exactly seven named coordinates")
    converted = [rational_string(x) for x in value]
    if any(x is None for x in converted):
        raise FrontendContractError("Referenced KB vector is incomplete")
    return dict(zip(BASE_DIMENSIONS, converted))


def gate_checker_audit(ledger, audit, retrieval):
    """Validate identity/coverage mechanically; unresolved physics stays RETRY.

    IDs identify the actual selected KB entities. No free-text canonical-name echo,
    fuzzy alias matching or unselected-record anchor is an additional gate.
    """
    ledger = validate_ledger(ledger)
    audit = CheckerDimensionAudit.model_validate(model_dict(audit)).model_dump()
    digest = ledger_sha256(ledger)
    if audit["ledger_sha256"] != digest:
        raise FrontendContractError("Checker reviewed a stale or different ledger hash")
    names = ledger["input_variable_names"]
    rows = audit["variables"]
    if len({x["csv_name"] for x in rows}) != len(rows) or {x["csv_name"] for x in rows} != set(names):
        raise FrontendContractError("Checker must audit each current input exactly once")
    audits = {x["csv_name"]: x for x in rows}
    results, feedback, used_records, issues = [], [], [], []
    enabled = retrieval.get("status") == "success"
    matches = retrieval.get("matches_by_variable") or {}
    visible = retrieval.get("evidence_records_visible_by_variable")
    for variable in ledger["variables"]:
        name, before = variable["csv_name"], variable["si_exponents"]
        item = audits[name]
        if item["before_si_exponents"] != before:
            raise FrontendContractError(f"Checker before vector differs from actual ledger for {name}")
        ids = item["evidence_entry_ids"]
        if len(ids) != len(set(ids)):
            raise FrontendContractError("Duplicate evidence entry ID")
        available = {str(x.get("entry_id")): x for x in matches.get(name, [])} if enabled else {}
        selected = []
        for entry_id in ids:
            if not enabled or entry_id not in available or (visible is not None and entry_id not in visible.get(name, [])):
                raise FrontendContractError(f"Unseen, wrong-variable or unavailable evidence ID for {name}: {entry_id}")
            selected.append(available[entry_id])
        if item["support_mode"] == "KB_RECORD" and not selected:
            raise FrontendContractError(f"KB_RECORD support requires a real cited entry for {name}")
        if item["support_mode"] != "KB_RECORD" and selected:
            raise FrontendContractError(f"Evidence IDs require KB_RECORD support mode for {name}")
        suggested = item["suggested_si_exponents"]
        if item["decision"] == "PASSED" and suggested is not None and suggested != before:
            raise FrontendContractError(f"PASSED cannot ignore a changed suggested vector for {name}")
        if item["decision"] == "PASSED" and item["support_mode"] == "UNCERTAIN":
            raise FrontendContractError(f"PASSED cannot use UNCERTAIN support for {name}")
        target = before if item["decision"] == "PASSED" else suggested
        local_issues = []
        before_unknown = any(value is None for value in before.values())
        if before_unknown:
            local_issues.append("Current SI vector contains genuinely unknown coordinates; Analyzer must resolve them explicitly, never substitute0")
        if item["decision"] == "CONFLICT":
            if target is None or any(value is None for value in target.values()):
                local_issues.append("Checker correction is incomplete; preserve its unknowns and request Analyzer revision, no automatic KB-to-candidate filling")
            elif target == before:
                local_issues.append("CONFLICT has no exponent change; Analyzer must explicitly address the reported physical meaning or reasoning")
        cited_vectors = [record_vector(record) for record in selected]
        if cited_vectors and not all(vector == cited_vectors[0] for vector in cited_vectors):
            raise FrontendContractError(f"Cited KB records disagree for {name}")
        reference = cited_vectors[0] if cited_vectors else None
        known_reference_conflicts = {base: {"candidate_or_suggestion": target[base], "cited_record": reference[base]}
                                     for base in BASE_DIMENSIONS
                                     if target is not None and reference is not None and target[base] is not None and target[base] != reference[base]}
        if known_reference_conflicts:
            if item["decision"] == "PASSED":
                local_issues.append("Checker PASSED contradicts the known coordinates of its actual cited KB vector; program forces RETRY")
                # A complete candidate can receive the cited full vector as
                # correction feedback; no ledger is mutated. Unknown coordinates
                # remain unknown until Analyzer explicitly supplies them.
                if not before_unknown:
                    target = reference
            else:
                raise FrontendContractError(f"Cited KB vector differs from known Checker correction coordinates for {name}")
        if item["decision"] != "PASSED":
            local_issues.append("Checker decision " + item["decision"])
        diff = {base: {"before": before[base], "suggested": target[base]} for base in BASE_DIMENSIONS
                if target is not None and before[base] != target[base]}
        entities = [{"entry_id": str(record.get("entry_id")),
                     "canonical_quantity": record.get("canonical_quantity", ""),
                     "source_identifier": record.get("source_identifier", "")}
                    for record in selected]
        feedback.append(f"Variable {name}: decision={item['decision']}; reason={item['reason']}\nActual named SI={json.dumps(before)}\nSuggested named SI={json.dumps(target)}\nExact coordinate diff={json.dumps(diff)}\nCited entity IDs/canonical names (program-read, not another model echo)={json.dumps(entities, ensure_ascii=False)}\nKnown coordinate conflicts with cited record={json.dumps(known_reference_conflicts)}\nHard gate issues={json.dumps(local_issues)}")
        for record in selected:
            used_records.append({"csv_name": name, "record": record})
            feedback.append("Referenced KB original record for " + name + ":\n" + json.dumps(record, ensure_ascii=False, sort_keys=True))
        results.append({"csv_name": name, "decision": item["decision"], "before": before, "suggested": target,
                        "coordinate_diff": diff, "issues": local_issues,
                        "referenced_entry_ids": ids, "selected_entities": entities,
                        "cited_reference_si_exponents": reference,
                        "known_coordinate_conflicts_with_reference": known_reference_conflicts})
        issues.extend(f"{name}: {issue}" for issue in local_issues)
    status = "RETRY" if issues else "PASSED"
    return {"gate_status": status, "ledger_sha256": digest, "checker_audit": audit, "variable_results": results,
            "issues": issues, "referenced_records": used_records,
            "feedback": "Typed ledger " + digest + "; program gate=" + status + "\n" + "\n\n".join(feedback),
            "entity_policy": "Only Checker-selected visible record IDs identify cited canonical entities; no unselected-record or Analyzer-name anchor. Equal vectors do not establish equal physical entities.",
            "physical_correctness_scope": "A passed contract is not a theorem of physical correctness. Record ID and coordinate checks establish citation consistency; semantic relevance/applicability in the reason and internal-reasoning physics remain model judgements."}


class ParateraEmbeddingClient:
    """GLM-Embedding-3 adapter for the Paratera OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str = KNOWLEDGE_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.getenv(PARATERA_API_KEY_ENV)
        base_url = base_url or PARATERA_BASE_URL
        if not api_key:
            raise EnvironmentError(f"{PARATERA_API_KEY_ENV} is not set.")
        self.model = str(model)
        self.base_url = str(base_url)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        texts = [str(text) for text in texts]
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            data = sorted(response.data, key=lambda item: item.index)
            batch_embeddings = [list(item.embedding) for item in data]
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding response count does not match the number of input texts."
                )
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=str(text))
        return list(response.data[0].embedding)


class DimensionalKnowledgeRetriever:
    """Hybrid lexical/semantic retriever over the physical-quantity table."""

    def __init__(
        self,
        *,
        knowledge_base_path: str = PHYSICAL_KNOWLEDGE_BASE_FILEPATH,
        embedding_cache_path: str = KNOWLEDGE_INDEX_FILEPATH,
        embedding_model: str = KNOWLEDGE_EMBEDDING_MODEL,
        lexical_weight: float = KNOWLEDGE_LEXICAL_WEIGHT,
        semantic_weight: float = KNOWLEDGE_SEMANTIC_WEIGHT,
        min_hybrid_score: float = KNOWLEDGE_MIN_HYBRID_SCORE,
        enable_embeddings: bool = True,
        allow_lexical_fallback: bool = KNOWLEDGE_ALLOW_LEXICAL_FALLBACK,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.embedding_cache_path = Path(embedding_cache_path)
        self.embedding_model = str(embedding_model)
        total_weight = max(0.0, float(lexical_weight)) + max(0.0, float(semantic_weight))
        if total_weight <= 0.0:
            raise ValueError("At least one retrieval weight must be positive.")
        self.lexical_weight = max(0.0, float(lexical_weight)) / total_weight
        self.semantic_weight = max(0.0, float(semantic_weight)) / total_weight
        self.min_hybrid_score = max(0.0, float(min_hybrid_score))
        self.enable_embeddings = bool(enable_embeddings)
        self.allow_lexical_fallback = bool(allow_lexical_fallback)
        self._kb_signature: Optional[Tuple[int, int]] = None
        self._kb_hash: Optional[str] = None
        self._frame: Optional[pd.DataFrame] = None
        self._records: List[Dict[str, str]] = []
        self._semantic_texts: List[str] = []
        self._normalized_embeddings: Optional[np.ndarray] = None
        self._embedding_client: Optional[ParateraEmbeddingClient] = None
        self._embedding_status: str = "not_initialized"

    def _current_signature(self) -> Tuple[int, int]:
        stat = self.knowledge_base_path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _ensure_records_loaded(self) -> None:
        signature = self._current_signature()
        if self._frame is not None and signature == self._kb_signature:
            return
        frame = load_dimensional_knowledge_base(str(self.knowledge_base_path))
        self._frame = frame
        self._records = frame.to_dict(orient="records")
        self._semantic_texts = [_record_semantic_text(record) for record in self._records]
        self._kb_signature = signature
        self._kb_hash = _sha256_file(self.knowledge_base_path)
        self._normalized_embeddings = None
        self._embedding_status = "records_reloaded"

    def _get_embedding_client(self) -> ParateraEmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = ParateraEmbeddingClient(model=self.embedding_model)
        return self._embedding_client

    def _ensure_embeddings(self) -> Optional[np.ndarray]:
        self._ensure_records_loaded()
        if not self.enable_embeddings:
            self._embedding_status = "disabled"
            return None
        if self._normalized_embeddings is not None:
            return self._normalized_embeddings
        cached = _load_embedding_cache(
            self.embedding_cache_path,
            kb_hash=str(self._kb_hash),
            embedding_model=self.embedding_model,
            record_count=len(self._records),
        )
        if cached is not None:
            self._normalized_embeddings = cached
            self._embedding_status = "cache"
            return cached
        _private_rebuild_cache_path(self)
        cached = _load_embedding_cache(
            self.embedding_cache_path, kb_hash=str(self._kb_hash),
            embedding_model=self.embedding_model, record_count=len(self._records),
        )
        if cached is not None:
            self._normalized_embeddings = cached
            self._embedding_status = "cache"
            return cached
        try:
            embeddings = np.asarray(
                self._get_embedding_client().embed_documents(self._semantic_texts),
                dtype=float,
            )
            embeddings = _l2_normalize_rows(embeddings)
            _save_embedding_cache(
                self.embedding_cache_path,
                kb_hash=str(self._kb_hash),
                embedding_model=self.embedding_model,
                embeddings=embeddings,
            )
            self._normalized_embeddings = embeddings
            self._embedding_status = "rebuilt"
            return embeddings
        except Exception as exc:
            self._embedding_status = f"embedding_unavailable: {type(exc).__name__}"
            if not self.allow_lexical_fallback:
                raise
            warnings.warn(
                "Embedding retrieval is unavailable; falling back to lexical "
                f"knowledge retrieval only. Reason: {exc}",
                RuntimeWarning,
            )
            return None

    def _semantic_scores(self, query: str) -> Optional[np.ndarray]:
        embeddings = self._ensure_embeddings()
        if embeddings is None:
            return None
        try:
            query_vector = np.asarray(
                self._get_embedding_client().embed_query(query), dtype=float
            )
            norm = max(float(np.linalg.norm(query_vector)), 1.0e-15)
            query_vector = query_vector / norm
            if embeddings.shape[1] != query_vector.size:
                embeddings = _rebuild_dimension_mismatched_cache(self, query_vector.size)
            return np.maximum(embeddings @ query_vector, 0.0)
        except Exception as exc:
            self._embedding_status = f"query_embedding_unavailable: {type(exc).__name__}"
            if not self.allow_lexical_fallback:
                raise
            warnings.warn(
                "Query embedding is unavailable; using lexical retrieval only. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            return None

    def retrieve(
        self,
        *,
        variable_name: str,
        problem_context: str,
        analyzer_context: str,
        top_k: int = KNOWLEDGE_TOP_K_PER_VARIABLE,
    ) -> List[Dict[str, Any]]:
        self._ensure_records_loaded()
        local_problem_context = _extract_problem_variable_context(
            problem_context, variable_name
        )
        query = (
            f"Target CSV variable: {variable_name}\n"
            f"Local problem description: {local_problem_context}\n"
            f"Problem domain/context: {_problem_preamble(problem_context)}\n"
            f"Analyzer interpretation: {analyzer_context}"
        )
        self.last_query = query
        self.query_history = getattr(self, "query_history", []) + [{"variable_name": variable_name, "query": query}]
        semantic_scores = self._semantic_scores(query)
        ranked: List[Dict[str, Any]] = []
        for index, record in enumerate(self._records):
            lexical = _lexical_score(variable_name, query, record)
            semantic = float(semantic_scores[index]) if semantic_scores is not None else 0.0
            if semantic_scores is None:
                hybrid = lexical
                retrieval_mode = "lexical_only"
            else:
                hybrid = self.lexical_weight * lexical + self.semantic_weight * semantic
                retrieval_mode = "hybrid"
            effective_min_score = (
                max(self.min_hybrid_score, 0.45)
                if semantic_scores is None else self.min_hybrid_score
            )
            if hybrid < effective_min_score:
                continue
            ranked.append({
                "variable_name": str(variable_name),
                "entry_id": record["entry_id"],
                "canonical_quantity": record["canonical_quantity"],
                "symbols": record["symbols"],
                "aliases": record["aliases"],
                "definition": record["definition"],
                "si_dimension": record["si_dimension"],
                "dimension_vector": record["dimension_vector"],
                "si_unit": record["si_unit"],
                "domain": record["domain"],
                "ambiguity_notes": record["ambiguity_notes"],
                "source": record["source"],
                "source_identifier": record["source_identifier"],
                "confidence": record["confidence"],
                "lexical_score": float(lexical),
                "semantic_score": float(semantic),
                "hybrid_score": float(hybrid),
                "retrieval_mode": retrieval_mode,
            })
        ranked.sort(
            key=lambda item: (
                item["hybrid_score"], item["lexical_score"], item["semantic_score"]
            ),
            reverse=True,
        )
        return ranked[:max(1, int(top_k))]

    @property
    def kb_hash(self) -> Optional[str]:
        self._ensure_records_loaded()
        return self._kb_hash

    @property
    def embedding_status(self) -> str:
        return self._embedding_status


def _assert_private_cache_write(path):
    target = Path(path).resolve()
    if target.is_relative_to(DATA_DIRECTORY.resolve()):
        raise PermissionError("Refusing to rebuild/write caches in input DATA_DIRECTORY; configure a private writable DIMAGENT_KNOWLEDGE_INDEX_FILEPATH or DIMAGENT_DIMENSIONLESS_INDEX_FILEPATH")


def _private_rebuild_cache_path(retriever):
    """Read supplied NPZ in place; rebuild missing/stale caches only privately."""
    original = Path(retriever.embedding_cache_path).resolve()
    if not original.is_relative_to(DATA_DIRECTORY.resolve()):
        _assert_private_cache_write(original)
        return original
    cache_root = Path(os.getenv("DIMAGENT_CACHE_DIR", str(CACHE_DIR))).resolve()
    key = hashlib.sha256((str(retriever._kb_hash) + "|" + retriever.embedding_model + "|" + PARATERA_BASE_URL).encode("utf-8")).hexdigest()[:20]
    target = cache_root / (retriever.knowledge_base_path.stem + "-" + key + ".npz")
    _assert_private_cache_write(target)
    retriever.embedding_cache_source = str(original)
    retriever.embedding_cache_path = target
    return target


def _rebuild_dimension_mismatched_cache(retriever, expected_dimension):
    """Re-embed records privately after a real query exposes a stale dimension."""
    _private_rebuild_cache_path(retriever)
    embeddings = np.asarray(retriever._get_embedding_client().embed_documents(retriever._semantic_texts), dtype=float)
    if (embeddings.ndim != 2 or embeddings.shape != (len(retriever._records), int(expected_dimension))
            or not np.isfinite(embeddings).all()):
        raise ValueError("Rebuilt record embeddings do not match current query dimension/count")
    embeddings = _l2_normalize_rows(embeddings)
    _save_embedding_cache(retriever.embedding_cache_path, kb_hash=str(retriever._kb_hash),
                          embedding_model=retriever.embedding_model, embeddings=embeddings)
    retriever._normalized_embeddings = embeddings
    retriever._embedding_status = "rebuilt_after_query_dimension_mismatch"
    return embeddings


def _split_markdown_row(line: str) -> List[str]:
    """Split one Markdown-table row while respecting escaped pipes."""
    text = str(line).strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]

    cells: List[str] = []
    buffer: List[str] = []
    escaped = False
    for char in text:
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    if escaped:
        buffer.append("\\")
    cells.append("".join(buffer).strip())
    return cells


def _is_markdown_separator_row(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(
        bool(re.fullmatch(r":?-{3,}:?", cell.strip())) for cell in cells
    )


def _find_markdown_table(
    markdown_text: str,
    required_columns: Sequence[str],
) -> Tuple[List[str], List[List[str]]]:
    """Find a Markdown table by its required schema."""
    lines = markdown_text.splitlines()
    required = set(required_columns)
    for index in range(len(lines) - 1):
        header_line = lines[index].strip()
        separator_line = lines[index + 1].strip()
        if "|" not in header_line or not _is_markdown_separator_row(separator_line):
            continue
        header = [cell.strip() for cell in _split_markdown_row(header_line)]
        if not required.issubset(set(header)):
            continue

        rows: List[List[str]] = []
        cursor = index + 2
        while cursor < len(lines):
            row_line = lines[cursor].strip()
            if not row_line or "|" not in row_line:
                break
            row = _split_markdown_row(row_line)
            if len(row) != len(header):
                raise ValueError(
                    "Malformed knowledge-base row at Markdown line "
                    f"{cursor + 1}: expected {len(header)} cells, got {len(row)}."
                )
            rows.append(row)
            cursor += 1
        return header, rows

    raise ValueError(
        "No Markdown table matching the required schema was found. "
        f"Required columns: {tuple(required_columns)}"
    )


def _validate_dimension_vector(raw_value: str, entry_id: str) -> str:
    try:
        vector = json.loads(str(raw_value))
    except Exception as exc:
        raise ValueError(
            f"Knowledge-base entry {entry_id!r} has an invalid dimension_vector."
        ) from exc

    if not isinstance(vector, list) or len(vector) != len(CANONICAL_DIMENSION_ORDER):
        raise ValueError(
            f"Knowledge-base entry {entry_id!r} must have exactly "
            f"{len(CANONICAL_DIMENSION_ORDER)} components in order "
            f"{CANONICAL_DIMENSION_ORDER}."
        )

    normalized: List[float | int] = []
    for value in vector:
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ValueError(
                f"Knowledge-base entry {entry_id!r} contains a non-numeric "
                f"dimension-vector component: {value!r}."
            )
        number = float(value)
        normalized.append(int(number) if number.is_integer() else number)
    return json.dumps(normalized, ensure_ascii=False)


def _load_markdown_table(filepath: str, required_columns: Sequence[str]) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dimensional knowledge base not found: {path}")
    markdown_text = path.read_text(encoding="utf-8")
    header, rows = _find_markdown_table(markdown_text, required_columns)
    frame = pd.DataFrame(rows, columns=header)
    frame = frame.loc[:, list(required_columns)].copy()
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    if frame.empty:
        raise ValueError("The requested knowledge-base table is empty.")
    if (frame["entry_id"] == "").any():
        raise ValueError("Every knowledge-base record must have a non-empty entry_id.")
    duplicated = frame.loc[frame["entry_id"].duplicated(keep=False), "entry_id"].tolist()
    if duplicated:
        raise ValueError(
            "Duplicate knowledge-base entry_id values: "
            + ", ".join(sorted(set(duplicated)))
        )
    return frame.reset_index(drop=True)


def load_dimensional_knowledge_base(filepath: str) -> pd.DataFrame:
    frame = _load_markdown_table(filepath, PHYSICAL_QUANTITY_REQUIRED_COLUMNS)
    if (frame["canonical_quantity"] == "").any():
        raise ValueError("Every physical-quantity record must have a canonical_quantity.")
    frame["dimension_vector"] = [
        _validate_dimension_vector(value, entry_id)
        for value, entry_id in zip(frame["dimension_vector"], frame["entry_id"])
    ]
    allowed_confidence = {"curated", "high", "medium", "low"}
    invalid = sorted(set(frame["confidence"]) - allowed_confidence)
    if invalid:
        raise ValueError(
            f"Invalid confidence labels in physical-quantity KB: {invalid}."
        )
    return frame


def _replace_greek(text: str) -> str:
    result = str(text)
    for symbol, word in _GREEK_NORMALIZATION.items():
        result = result.replace(symbol, word)
    return result


def _normalize_text(text: str) -> str:
    text = _replace_greek(str(text)).lower()
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9.+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set[str]:
    return {
        token for token in _normalize_text(text).split()
        if token and token not in _STOPWORDS
    }


def _split_multi_value_cell(value: str) -> List[str]:
    parts = re.split(r"\s*[;,]\s*", str(value))
    return [part.strip() for part in parts if part.strip()]


def _record_semantic_text(record: Dict[str, str]) -> str:
    return (
        f"Canonical physical quantity: {record['canonical_quantity']}. "
        f"Symbols: {record.get('symbols', '')}. "
        f"Aliases: {record.get('aliases', '')}. "
        f"Definition: {record.get('definition', '')}. "
        f"SI dimensions: {record.get('si_dimension', '')}. "
        f"SI unit: {record.get('si_unit', '')}. "
        f"Domain: {record.get('domain', '')}. "
        f"Ambiguity notes: {record.get('ambiguity_notes', '')}."
    )


def _lexical_score(
    variable_name: str,
    query_context: str,
    record: Dict[str, str],
) -> float:
    variable_raw = str(variable_name).strip()
    variable_norm = _normalize_text(variable_raw)
    canonical = str(record.get("canonical_quantity", "")).strip()
    symbols = _split_multi_value_cell(record.get("symbols", ""))
    aliases = _split_multi_value_cell(record.get("aliases", ""))

    exact_component = 0.0
    if variable_raw in symbols:
        exact_component = max(exact_component, 1.00)
    elif any(variable_raw.lower() == symbol.lower() for symbol in symbols):
        exact_component = max(exact_component, 0.90)
    if variable_norm == _normalize_text(canonical):
        exact_component = max(exact_component, 0.98)
    for alias in aliases:
        alias_norm = _normalize_text(alias)
        if variable_norm == alias_norm:
            exact_component = max(exact_component, 0.97)
        elif (
            len(variable_norm) >= 3
            and alias_norm
            and (variable_norm in alias_norm or alias_norm in variable_norm)
        ):
            exact_component = max(exact_component, 0.74)

    query_tokens = _tokenize(query_context)
    record_tokens = _tokenize(_record_semantic_text(record))
    overlap = (
        len(query_tokens & record_tokens) / max(1, len(query_tokens | record_tokens))
        if query_tokens and record_tokens else 0.0
    )
    context_component = min(0.60, 2.5 * overlap)
    return float(
        min(1.0, max(exact_component, 0.45 * exact_component + context_component))
    )


def _extract_problem_variable_context(
    problem_description: str,
    variable_name: str,
    *,
    max_characters: int = 500,
) -> str:
    """Extract the prompt fragment that explicitly describes one input variable."""
    text = str(problem_description or "")
    if not text.strip():
        return ""
    raw_name = str(variable_name).strip()
    normalized_name = _normalize_text(raw_name)
    matches: List[str] = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        normalized_line = _normalize_text(line)
        tokens = normalized_line.split()
        if raw_name and re.search(r"\(" + re.escape(raw_name) + r"\)", line, flags=re.IGNORECASE):
            matches.append(line)
        elif normalized_name and normalized_name in tokens:
            matches.append(line)
    return " ".join(matches)[:max_characters]


def _problem_preamble(problem_description: str, max_characters: int = 600) -> str:
    text = str(problem_description or "").strip()
    if not text:
        return ""
    split = re.split(r"\bThis analysis uses\b", text, maxsplit=1, flags=re.IGNORECASE)
    return split[0].strip()[:max_characters]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0e-15)
    return matrix / norms


def _load_embedding_cache(
    cache_path: Path,
    *,
    kb_hash: str,
    embedding_model: str,
    record_count: int,
) -> Optional[np.ndarray]:
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        cached_hash = str(payload["kb_sha256"].item())
        cached_model = str(payload["embedding_model"].item())
        cached_count = int(payload["record_count"].item())
        cached_dimension = int(payload["embedding_dimension"].item())
        cached_base_url = str(payload["embedding_base_url"].item())
        embeddings = np.asarray(payload["embeddings"], dtype=float)
    except Exception:
        return None
    if (
        cached_hash != kb_hash
        or cached_model != embedding_model
        or cached_count != int(record_count)
        or cached_base_url != PARATERA_BASE_URL
        or embeddings.ndim != 2
        or embeddings.shape[0] != int(record_count)
        or embeddings.shape[1] != cached_dimension
    ):
        return None
    return _l2_normalize_rows(embeddings)


def _save_embedding_cache(
    cache_path: Path,
    *,
    kb_hash: str,
    embedding_model: str,
    embeddings: np.ndarray,
) -> None:
    _assert_private_cache_write(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        temp_path,
        kb_sha256=np.array(kb_hash),
        embedding_model=np.array(embedding_model),
        embedding_provider=np.array("Paratera"),
        embedding_base_url=np.array(PARATERA_BASE_URL),
        record_count=np.array(int(embeddings.shape[0]), dtype=int),
        embedding_dimension=np.array(int(embeddings.shape[1]), dtype=int),
        embeddings=np.asarray(embeddings, dtype=float),
    )
    temp_path.replace(cache_path)


def get_dimensional_kb_retriever() -> DimensionalKnowledgeRetriever:
    global _DIMENSIONAL_KB_RETRIEVER
    if _DIMENSIONAL_KB_RETRIEVER is None:
        _DIMENSIONAL_KB_RETRIEVER = DimensionalKnowledgeRetriever()
    return _DIMENSIONAL_KB_RETRIEVER


def retrieve_dimension_evidence(*, problem_description, input_variable_names, dimension_detect,
                                dimension_ledger, top_k_per_variable=KNOWLEDGE_TOP_K_PER_VARIABLE,
                                retriever=None):
    """Original hybrid algorithm; only meaning-only typed query context is new."""
    ledger = validate_ledger(dimension_ledger)
    if list(input_variable_names) != ledger["input_variable_names"]:
        raise FrontendContractError("Retrieval variable order differs from typed ledger")
    contexts = render_retrieval_interpretation(ledger)
    if retriever is None:
        retriever = get_dimensional_kb_retriever()
    matches, queries = {}, {}
    observed = {}
    for name in input_variable_names:
        matches[name] = retriever.retrieve(variable_name=name, problem_context=str(problem_description or ""),
                                            analyzer_context=contexts[name], top_k=top_k_per_variable)
        expected_query = (
            f"Target CSV variable: {name}\n"
            f"Local problem description: {_extract_problem_variable_context(problem_description, name)}\n"
            f"Problem domain/context: {_problem_preamble(problem_description)}\n"
            f"Analyzer interpretation: {contexts[name]}"
        )
        actual_query = getattr(retriever, "last_query", None)
        if actual_query is not None and actual_query != expected_query:
            raise FrontendContractError("Observed retriever query differs from typed semantic query")
        queries[name] = actual_query if actual_query is not None else expected_query
        observed[name] = actual_query is not None
    evidence, visible = format_evidence_records(matches, max_rows=KNOWLEDGE_MAX_EVIDENCE_ROWS)
    modes = sorted({m.get("retrieval_mode", "unknown") for rows in matches.values() for m in rows})
    return {"status": "success", "kb_path": str(retriever.knowledge_base_path), "kb_sha256": retriever.kb_hash,
            "embedding_cache_path": str(retriever.embedding_cache_path),
            "embedding_cache_source": getattr(retriever, "embedding_cache_source", None), "embedding_model": retriever.embedding_model,
            "embedding_provider": "Paratera", "embedding_base_url": PARATERA_BASE_URL,
            "embedding_status": retriever.embedding_status, "retrieval_modes": modes,
            "retrieved_variables": list(input_variable_names), "n_retrieved_records": sum(map(len, matches.values())),
            "top_k_per_variable": top_k_per_variable, "max_evidence_rows": KNOWLEDGE_MAX_EVIDENCE_ROWS,
            "lexical_weight": retriever.lexical_weight, "semantic_weight": retriever.semantic_weight,
            "min_hybrid_score": retriever.min_hybrid_score, "allow_lexical_fallback": retriever.allow_lexical_fallback,
            "matches_by_variable": matches, "evidence_markdown": evidence,
            "evidence_records_visible_by_variable": visible, "queries_by_variable": queries,
            "query_sha256_by_variable": {name: hashlib.sha256(query.encode("utf-8")).hexdigest() for name, query in queries.items()},
            "query_observed_in_retriever_by_variable": observed,
            "query_scope": "Candidate SI powers excluded. Original problem and typed quantity/definition provide semantic context; they still require physical interpretation."}


def safe_retrieve_dimension_evidence(**kwargs):
    try:
        return retrieve_dimension_evidence(**kwargs)
    except Exception as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc),
                "retrieval_modes": [], "n_retrieved_records": 0, "matches_by_variable": {},
                "queries_by_variable": {}, "query_sha256_by_variable": {}, "evidence_records_visible_by_variable": {},
                "evidence_markdown": "External evidence unavailable. Do not claim any KB_RECORD support; unresolved meanings must remain UNCERTAIN."}


def checker_messages(prompt, ledger, digest, coordinates, evidence):
    """Reconstruct the archived Checker message pair without a model call."""
    system = (
        "Audit every named physical input in this exact typed ledger, including literal SI coordinate values. "
        "Do not substitute the correct definition or a KB vector for a wrong candidate and then claim PASSED. "
        "Return ledger_sha256 and each csv_name exactly once; before_si_exponents must copy the actual candidate. "
        "PASSED confirms the actual vector, CONFLICT requires a complete corrected suggested vector, and UNCERTAIN "
        "does not permit numerical execution. KB_RECORD requires actual entry_id citations from that variable's "
        "retrieved records. Canonical entity names are read by the program from the selected IDs; do not echo "
        "another quantity-name field. Your reason must explain why each cited record applies to this variable in "
        "the original problem, not merely why its vector matches. Cited vectors must support all known coordinates "
        "of the proposed/confirmed SI vector. INTERNAL_REASONING uses no IDs and must explain "
        "the physical meaning. Retrieval rank is relevance, not truth. Resolve symbol ambiguity from the original "
        "problem; do not choose an unrelated matching symbol. Known absent base dimensions must be explicit '0'; "
        "null means genuinely unknown, not unused. If any current coordinate is unknown, request revision or report "
        "UNCERTAIN; never call a partly known vector PASSED. Give a complete correction when physically justified. "
        "If a correction remains partly unknown, preserve null and explain what remains unresolved; the program "
        "will request revision rather than invent zeros. Lack of KB alone is not a physical contradiction. "
        "Matrix compilation and n/r/k are program tasks, not audit tasks."
    )

    provisional_context = ""
    if coordinates:
        system += (
            " The listed provisional zero coordinates were null in the Analyzer response. "
            "Copy their current zero values in before_si_exponents and verify each from the physical definition "
            "or applicable retrieved evidence. Use PASSED only if you can affirm those zeros. "
            "If an exponent is nonzero, report CONFLICT with the complete corrected vector. "
            "If any provisional coordinate cannot be verified, report UNCERTAIN."
        )
        provisional_context = f"Provisional zero coordinates:\n{json.dumps(coordinates, ensure_ascii=False)}\n\n"

    human = (
        f"Problem:\n{prompt}\n\n"
        f"Typed ledger SHA256: {digest}\nTyped candidate ledger:\n"
        f"{json.dumps(ledger, ensure_ascii=False, sort_keys=True)}\n\n"
        f"{provisional_context}"
        f"Retrieved dimensional evidence:\n{evidence}"
    )
    return [{"type": "system", "content": system}, {"type": "human", "content": human}]
