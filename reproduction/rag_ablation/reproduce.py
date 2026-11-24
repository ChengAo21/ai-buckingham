"""Replay and summarize the archived trials in a temporary copy, preserving snapshots."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from shutil import copyfile, copytree, ignore_patterns
import subprocess
import sys
from tempfile import TemporaryDirectory

import pandas as pd


HERE = Path(__file__).resolve().parent


def equivalent_json(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(equivalent_json(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(equivalent_json(a, b) for a, b in zip(actual, expected, strict=True)))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
    return actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == HERE or output.is_relative_to(HERE / "results") or output.is_relative_to(HERE / "trials"):
        parser.error("Choose a separate generated-output directory")
    manifest = HERE / "SHA256SUMS"
    entries = [line.split("  ", 1) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for digest, relative in entries:
        if sha256((HERE / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archive checksum mismatch: {relative}")

    output.mkdir(parents=True, exist_ok=True)
    compared = []
    with TemporaryDirectory(prefix="buckingham-rag-") as temporary:
        work = Path(temporary)
        for folder in ("reference", "trials", "results"):
            copytree(HERE / folder, work / folder, ignore=ignore_patterns("__pycache__", "*.pyc"))
        for script in ("rag.py", "raw_evidence.py", "validate_rag.py", "summarize.py"):
            copyfile(HERE / script, work / script)
        for script in ("validate_rag.py", "summarize.py"):
            subprocess.run([sys.executable, "-B", str(work / script)], cwd=work, check=True)

        for expected_path in sorted((HERE / "results").iterdir()):
            generated_path = work / "results" / expected_path.name
            if expected_path.suffix == ".csv":
                actual = pd.read_csv(generated_path, float_precision="round_trip")
                expected = pd.read_csv(expected_path, float_precision="round_trip")
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-9, atol=1e-10)
            elif expected_path.suffix == ".json":
                actual = json.loads(generated_path.read_text(encoding="utf-8-sig"))
                expected = json.loads(expected_path.read_text(encoding="utf-8-sig"))
                if not equivalent_json(actual, expected):
                    raise ValueError(f"Recomputed JSON differs from the snapshot: {expected_path.name}")
            else:
                continue
            copyfile(generated_path, output / generated_path.name)
            compared.append(expected_path.name)

    for digest, relative in entries:
        if sha256((HERE / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Recorded archive changed: {relative}")
    report = {"snapshot_files_compared": compared, "archive_files_verified": len(entries),
              "snapshots_unchanged": True, "relative_tolerance": 1e-9, "absolute_tolerance": 1e-10}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(compared)} regenerated results; all {len(entries)} archive checksums match")
    print(output)


if __name__ == "__main__":
    main()
