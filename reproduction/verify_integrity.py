"""Verify published inputs and historical evidence without scientific dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    records = []

    def check(path: Path, expected: str, archive: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing published file: {path.relative_to(ROOT)}")
        if digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {path.relative_to(ROOT)}")
        records.append({"path": path.relative_to(ROOT).as_posix(), "archive": archive})

    data = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    for item in data["datasets"] + data["knowledge_bases"]:
        check(ROOT / item["path"], item["sha256"], "case_inputs")

    rag = HERE / "rag_ablation"
    for line in (rag / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        check(rag / relative, expected, "rag_ablation")

    for name in ("regime_comparison", "velocity_comparison"):
        package = HERE / name
        manifest = json.loads((package / "source_manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["source_files"].items():
            check(package / relative, expected, name)

    return {
        "status": "passed",
        "files_verified": len(records),
        "counts_by_archive": {
            name: sum(item["archive"] == name for item in records)
            for name in sorted({item["archive"] for item in records})
        },
        "method": "SHA-256 of published files against author-supplied records",
        "scientific_calculations_repeated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated" / "verification" / "input_integrity.json")
    args = parser.parse_args()
    output = args.output.resolve()
    for protected in (ROOT / "data", ROOT / "knowledge", ROOT / "configs", HERE):
        if output.is_relative_to(protected.resolve()):
            parser.error("Write the verification report outside published inputs and reproduction archives")
    report = verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['files_verified']} input/evidence files; report: {output}")


if __name__ == "__main__":
    main()
