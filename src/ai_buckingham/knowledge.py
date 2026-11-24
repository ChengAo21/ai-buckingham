"""Knowledge extracted from the author implementation."""
from __future__ import annotations

import os
import re
import json
import hashlib
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple
from openai import OpenAI
from . import config as _config
from . import dimensions as _dimensions
from . import provenance as _provenance
from . import schemas as _schemas


_DIMENSIONAL_KB_RETRIEVER = None


_DIMENSIONLESS_KB_RETRIEVER = None


class ParateraEmbeddingClient:
    """GLM-Embedding-3 adapter for the Paratera OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str = _config.KNOWLEDGE_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.getenv(_config.PARATERA_API_KEY_ENV)
        base_url = base_url or _config.PARATERA_BASE_URL
        if not api_key:
            raise EnvironmentError(f"{_config.PARATERA_API_KEY_ENV} is not set.")
        self.model = str(model)
        self.base_url = str(base_url)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, http_client=_provenance.RecordedHttpClient())

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
        knowledge_base_path: str = _config.PHYSICAL_KNOWLEDGE_BASE_FILEPATH,
        embedding_cache_path: str = _config.KNOWLEDGE_INDEX_FILEPATH,
        embedding_model: str = _config.KNOWLEDGE_EMBEDDING_MODEL,
        lexical_weight: float = _config.KNOWLEDGE_LEXICAL_WEIGHT,
        semantic_weight: float = _config.KNOWLEDGE_SEMANTIC_WEIGHT,
        min_hybrid_score: float = _config.KNOWLEDGE_MIN_HYBRID_SCORE,
        enable_embeddings: bool = True,
        allow_lexical_fallback: bool = _config.KNOWLEDGE_ALLOW_LEXICAL_FALLBACK,
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
        top_k: int = _config.KNOWLEDGE_TOP_K_PER_VARIABLE,
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


class DimensionlessRelationRetriever:
    """Semantic retriever for documented dimensionless relations used in Physical reading only."""

    def __init__(
        self,
        *,
        knowledge_base_path: str = _config.DIMENSIONLESS_KNOWLEDGE_BASE_FILEPATH,
        embedding_cache_path: str = _config.KNOWLEDGE_DIMENSIONLESS_INDEX_FILEPATH,
        embedding_model: str = _config.KNOWLEDGE_EMBEDDING_MODEL,
        allow_lexical_fallback: bool = _config.KNOWLEDGE_ALLOW_LEXICAL_FALLBACK,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.embedding_cache_path = Path(embedding_cache_path)
        self.embedding_model = str(embedding_model)
        self.allow_lexical_fallback = bool(allow_lexical_fallback)
        self._kb_signature: Optional[Tuple[int, int]] = None
        self._kb_hash: Optional[str] = None
        self._records: List[Dict[str, str]] = []
        self._semantic_texts: List[str] = []
        self._normalized_embeddings: Optional[np.ndarray] = None
        self._embedding_client: Optional[ParateraEmbeddingClient] = None
        self._embedding_status: str = "not_initialized"

    def _ensure_records_loaded(self) -> None:
        stat = self.knowledge_base_path.stat()
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if self._records and signature == self._kb_signature:
            return
        frame = load_dimensionless_relation_knowledge_base(str(self.knowledge_base_path))
        self._records = frame.to_dict(orient="records")
        self._semantic_texts = [
            _dimensionless_record_semantic_text(record) for record in self._records
        ]
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
                "Dimensionless-relation embeddings are unavailable; using lexical "
                f"fallback for Physical reading. Reason: {exc}",
                RuntimeWarning,
            )
            return None

    def retrieve(self, query: str, top_k: int = _config.KNOWLEDGE_INTERPRETATION_TOP_K) -> List[Dict[str, Any]]:
        self._ensure_records_loaded()
        embeddings = self._ensure_embeddings()
        query_norm = _normalize_text(query)
        query_tokens = _tokenize(query)
        semantic_scores: Optional[np.ndarray] = None
        if embeddings is not None:
            query_vector = np.asarray(
                self._get_embedding_client().embed_query(query), dtype=float
            )
            query_vector /= max(float(np.linalg.norm(query_vector)), 1.0e-15)
            if embeddings.shape[1] != query_vector.size:
                embeddings = _rebuild_dimension_mismatched_cache(self, query_vector.size)
            semantic_scores = np.maximum(embeddings @ query_vector, 0.0)

        ranked: List[Dict[str, Any]] = []
        for index, record in enumerate(self._records):
            record_text = _dimensionless_record_semantic_text(record)
            record_tokens = _tokenize(record_text)
            lexical_overlap = (
                len(query_tokens & record_tokens) / max(1, len(query_tokens | record_tokens))
                if query_tokens and record_tokens else 0.0
            )
            exact_reference = 0.0
            candidates = [record.get("canonical_name", "")]
            candidates += _split_multi_value_cell(record.get("symbols", ""))
            candidates += _split_multi_value_cell(record.get("aliases", ""))
            for candidate in candidates:
                normalized_candidate = _normalize_text(candidate)
                if len(normalized_candidate) >= 2 and normalized_candidate in query_norm:
                    exact_reference = max(exact_reference, 1.0)
            lexical_score = max(exact_reference, min(1.0, 3.0 * lexical_overlap))
            if semantic_scores is None:
                score = lexical_score
                mode = "lexical_only"
            else:
                # Final interpretation may recover a known group even when its conventional
                # name/symbol is absent from the discovered monomial, so semantic evidence
                # is intentionally dominant here.
                semantic_score = float(semantic_scores[index])
                score = 0.20 * lexical_score + 0.80 * semantic_score
                mode = "semantic_with_lexical_support"
            ranked.append({
                **record,
                "retrieval_score": float(score),
                "semantic_score": float(semantic_scores[index]) if semantic_scores is not None else 0.0,
                "lexical_score": float(lexical_score),
                "retrieval_mode": mode,
            })
        ranked.sort(key=lambda item: item["retrieval_score"], reverse=True)
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
    if any(target.is_relative_to(root.resolve()) for root in (_config.DATA_DIRECTORY, _config.KNOWLEDGE_DIR)):
        raise PermissionError("Refusing to rebuild/write caches in input DATA_DIRECTORY; configure a private writable DIMAGENT_KNOWLEDGE_INDEX_FILEPATH or DIMAGENT_DIMENSIONLESS_INDEX_FILEPATH")


def _private_rebuild_cache_path(retriever):
    """Read supplied NPZ in place; rebuild missing/stale caches only privately."""
    original = Path(retriever.embedding_cache_path).resolve()
    if not any(original.is_relative_to(root.resolve()) for root in (_config.DATA_DIRECTORY, _config.KNOWLEDGE_DIR)):
        _assert_private_cache_write(original)
        return original
    cache_root = Path(os.getenv("DIMAGENT_CACHE_DIR", str(_config.CACHE_DIR))).resolve()
    key = hashlib.sha256((str(retriever._kb_hash) + "|" + retriever.embedding_model + "|" + _config.PARATERA_BASE_URL).encode("utf-8")).hexdigest()[:20]
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


def load_file_local(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ not found at {filepath}. Returning empty string.")
        return ""


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

    if not isinstance(vector, list) or len(vector) != len(_config.CANONICAL_DIMENSION_ORDER):
        raise ValueError(
            f"Knowledge-base entry {entry_id!r} must have exactly "
            f"{len(_config.CANONICAL_DIMENSION_ORDER)} components in order "
            f"{_config.CANONICAL_DIMENSION_ORDER}."
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
    frame = _load_markdown_table(filepath, _config.PHYSICAL_QUANTITY_REQUIRED_COLUMNS)
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


def load_dimensionless_relation_knowledge_base(filepath: str) -> pd.DataFrame:
    frame = _load_markdown_table(filepath, _config.DIMENSIONLESS_RELATION_REQUIRED_COLUMNS)
    if (frame["canonical_name"] == "").any():
        raise ValueError("Every dimensionless-relation record must have a canonical_name.")
    return frame


def _replace_greek(text: str) -> str:
    result = str(text)
    for symbol, word in _config._GREEK_NORMALIZATION.items():
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
        if token and token not in _config._STOPWORDS
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


def _dimensionless_record_semantic_text(record: Dict[str, str]) -> str:
    return (
        f"Dimensionless relation: {record['canonical_name']}. "
        f"Symbols: {record.get('symbols', '')}. "
        f"Aliases: {record.get('aliases', '')}. "
        f"Defining expression: {record.get('defining_expression', '')}. "
        f"Physical interpretation: {record.get('physical_interpretation', '')}. "
        f"Domain: {record.get('domain', '')}."
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
        or cached_base_url != _config.PARATERA_BASE_URL
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
        embedding_base_url=np.array(_config.PARATERA_BASE_URL),
        record_count=np.array(int(embeddings.shape[0]), dtype=int),
        embedding_dimension=np.array(int(embeddings.shape[1]), dtype=int),
        embeddings=np.asarray(embeddings, dtype=float),
    )
    temp_path.replace(cache_path)


def _extract_variable_context(
    dimension_detect: str,
    variable_name: str,
    *,
    max_characters: int = 700,
) -> str:
    text = str(dimension_detect or "")
    if not text.strip():
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_name = str(variable_name).strip()
    normalized_name = _normalize_text(raw_name)
    matches: List[str] = []
    for line in lines:
        if raw_name and raw_name in line:
            matches.append(line)
            continue
        normalized_line = _normalize_text(line)
        if normalized_name and normalized_name in normalized_line.split():
            matches.append(line)
    return (" ".join(matches) if matches else text)[:max_characters]


def _format_retrieved_evidence(
    matches_by_variable: Dict[str, List[Dict[str, Any]]],
    *,
    max_rows: int = _config.KNOWLEDGE_MAX_EVIDENCE_ROWS,
) -> str:
    sections: List[str] = []
    emitted = 0
    for variable_name, matches in matches_by_variable.items():
        sections.append(f"### Input variable `{variable_name}`")
        if not matches:
            sections.append(
                "No sufficiently relevant knowledge-base record was retrieved. "
                "Treat this as NOT_FOUND, not as evidence of an error."
            )
            continue
        for match in matches:
            if emitted >= int(max_rows):
                break
            sections.append(
                "- "
                f"canonical_quantity: {match['canonical_quantity']}; "
                f"symbols: {match['symbols'] or '—'}; "
                f"aliases: {match['aliases'] or '—'}; "
                f"definition: {match['definition']}; "
                f"SI_dimension: {match['si_dimension']}; "
                f"dimension_vector[M,L,T,I,Theta,N,J]: {match['dimension_vector']}; "
                f"SI_unit: {match['si_unit']}; "
                f"domain: {match['domain']}; "
                f"ambiguity_notes: {match['ambiguity_notes'] or '—'}; "
                f"source: {match['source']}; "
                f"source_identifier: {match['source_identifier'] or '—'}; "
                f"confidence: {match['confidence']}."
            )
            emitted += 1
        if emitted >= int(max_rows):
            break
    return "\n".join(sections) if sections else (
        "No external dimensional evidence was retrieved. "
        "Do not infer a contradiction from missing evidence."
    )


def _format_dimensionless_interpretation_evidence(matches: Sequence[Dict[str, Any]]) -> str:
    if not matches:
        return (
            "No documented dimensionless relation was retrieved. Do not invent a "
            "classical parameter name from this absence."
        )
    lines = []
    for match in matches:
        lines.append(
            "- "
            f"name: {match.get('canonical_name')}; "
            f"symbols: {match.get('symbols') or '—'}; "
            f"aliases: {match.get('aliases') or '—'}; "
            f"defining_expression: {match.get('defining_expression')}; "
            f"physical_interpretation: {match.get('physical_interpretation')}; "
            f"domain: {match.get('domain')}; "
            f"retrieval_score: {float(match.get('retrieval_score', 0.0)):.3f}; "
            f"lexical_score: {float(match.get('lexical_score', 0.0)):.3f}; "
            f"semantic_score: {float(match.get('semantic_score', 0.0)):.3f}; "
            f"retrieval_mode: {match.get('retrieval_mode', '—')}; "
            f"source: {match.get('source')}; "
            f"source_identifier: {match.get('source_identifier') or '—'}; "
            f"confidence: {match.get('confidence')}."
        )
    return "\n".join(lines)


def get_dimensional_kb_retriever() -> DimensionalKnowledgeRetriever:
    global _DIMENSIONAL_KB_RETRIEVER
    if _DIMENSIONAL_KB_RETRIEVER is None:
        _DIMENSIONAL_KB_RETRIEVER = DimensionalKnowledgeRetriever()
    return _DIMENSIONAL_KB_RETRIEVER


def get_dimensionless_kb_retriever() -> DimensionlessRelationRetriever:
    global _DIMENSIONLESS_KB_RETRIEVER
    if _DIMENSIONLESS_KB_RETRIEVER is None:
        _DIMENSIONLESS_KB_RETRIEVER = DimensionlessRelationRetriever()
    return _DIMENSIONLESS_KB_RETRIEVER


def retrieve_dimensionless_interpretation_evidence(
    *,
    problem_description: str,
    deterministic_report: str,
    top_k: int = _config.KNOWLEDGE_INTERPRETATION_TOP_K,
) -> Dict[str, Any]:
    retriever = get_dimensionless_kb_retriever()
    query = (
        f"Physical problem:\n{problem_description}\n\n"
        f"Discovered dimensional-analysis result:\n{deterministic_report}"
    )
    matches = retriever.retrieve(query, top_k=top_k)
    return {
        "status": "success",
        "kb_path": str(retriever.knowledge_base_path),
        "kb_sha256": retriever.kb_hash,
        "embedding_cache_path": str(retriever.embedding_cache_path),
        "embedding_model": _config.KNOWLEDGE_EMBEDDING_MODEL,
        "embedding_provider": "Paratera",
        "embedding_base_url": _config.PARATERA_BASE_URL,
        "embedding_status": retriever.embedding_status,
        "retrieval_mode": (
            matches[0].get("retrieval_mode") if matches else "none"
        ),
        "n_retrieved_records": len(matches),
        "top_k": int(top_k),
        "allow_lexical_fallback": bool(retriever.allow_lexical_fallback),
        "matches": matches,
        "evidence_markdown": _format_dimensionless_interpretation_evidence(matches),
    }


def safe_retrieve_dimensionless_interpretation_evidence(
    *,
    problem_description: str,
    deterministic_report: str,
) -> Dict[str, Any]:
    try:
        return retrieve_dimensionless_interpretation_evidence(
            problem_description=problem_description,
            deterministic_report=deterministic_report,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "kb_path": str(Path(_config.DIMENSIONLESS_KNOWLEDGE_BASE_FILEPATH)),
            "embedding_cache_path": str(Path(_config.KNOWLEDGE_DIMENSIONLESS_INDEX_FILEPATH)),
            "embedding_model": _config.KNOWLEDGE_EMBEDDING_MODEL,
            "embedding_provider": "Paratera",
            "embedding_base_url": _config.PARATERA_BASE_URL,
            "n_retrieved_records": 0,
            "top_k": int(_config.KNOWLEDGE_INTERPRETATION_TOP_K),
            "allow_lexical_fallback": bool(_config.KNOWLEDGE_ALLOW_LEXICAL_FALLBACK),
            "matches": [],
            "evidence_markdown": (
                "No external dimensionless-relation evidence is available. "
                "Do not invent or force a classical dimensionless-number identity."
            ),
        }


def retrieve_dimension_evidence(*, problem_description, input_variable_names, dimension_detect,
                                dimension_ledger, top_k_per_variable=_config.KNOWLEDGE_TOP_K_PER_VARIABLE):
    """Original hybrid algorithm; only meaning-only typed query context is new."""
    ledger = _schemas.validate_ledger(dimension_ledger)
    if list(input_variable_names) != ledger["input_variable_names"]:
        raise _schemas.FrontendContractError("Retrieval variable order differs from typed ledger")
    contexts = _dimensions.render_retrieval_interpretation(ledger)
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
            raise _schemas.FrontendContractError("Observed retriever query differs from typed semantic query")
        queries[name] = actual_query if actual_query is not None else expected_query
        observed[name] = actual_query is not None
    evidence, visible = _dimensions.format_evidence_records(matches, max_rows=_config.KNOWLEDGE_MAX_EVIDENCE_ROWS)
    modes = sorted({m.get("retrieval_mode", "unknown") for rows in matches.values() for m in rows})
    return {"status": "success", "kb_path": str(retriever.knowledge_base_path), "kb_sha256": retriever.kb_hash,
            "embedding_cache_path": str(retriever.embedding_cache_path),
            "embedding_cache_source": getattr(retriever, "embedding_cache_source", None), "embedding_model": retriever.embedding_model,
            "embedding_provider": "Paratera", "embedding_base_url": _config.PARATERA_BASE_URL,
            "embedding_status": retriever.embedding_status, "retrieval_modes": modes,
            "retrieved_variables": list(input_variable_names), "n_retrieved_records": sum(map(len, matches.values())),
            "top_k_per_variable": top_k_per_variable, "max_evidence_rows": _config.KNOWLEDGE_MAX_EVIDENCE_ROWS,
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
