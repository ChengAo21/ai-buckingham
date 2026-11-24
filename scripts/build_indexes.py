"""Generate both GLM-Embedding-3 indexes using explicitly configured API access."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "knowledge" / "indexes")
    args = parser.parse_args()
    import numpy as np
    from ai_buckingham.knowledge import DimensionalKnowledgeRetriever, DimensionlessRelationRetriever
    from ai_buckingham.artifacts import sha256_file, write_json
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, cls in [("physical_quantity", DimensionalKnowledgeRetriever), ("dimensionless_relation", DimensionlessRelationRetriever)]:
        target = args.output_dir / f"{name}_index.npz"
        retriever = cls(embedding_cache_path=str(target), allow_lexical_fallback=False)
        retriever._ensure_records_loaded()
        vectors = retriever._ensure_embeddings()
        if vectors is None or vectors.shape[0] != len(retriever._records) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding matrix")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("Non-unit embedding rows")
        # Runtime retrieval rebuilds privately. This explicit authoring command
        # promotes that validated cache into the distributable read-only index.
        if Path(retriever.embedding_cache_path).resolve() != target.resolve():
            shutil.copy2(retriever.embedding_cache_path, target)
        record = {"knowledge_base": str(retriever.knowledge_base_path.relative_to(ROOT)).replace("\\", "/"),
                  "kb_sha256": retriever.kb_hash, "embedding_model": retriever.embedding_model,
                  "file": target.name, "sha256": sha256_file(target), "shape": list(vectors.shape),
                  "normalization": "unit L2 norm per record", "status": retriever.embedding_status}
        records.append(record)
        print(json.dumps({"index": name, "records": vectors.shape[0], "dimensions": vectors.shape[1], "status": retriever.embedding_status}), flush=True)
    write_json(args.output_dir / "manifest.json", {"built_at": datetime.now(timezone.utc).isoformat(),
                "provider": "Paratera", "method": "Actual GLM-Embedding-3 API embeddings with the original record serialization.",
                "indexes": records})


if __name__ == "__main__":
    main()
