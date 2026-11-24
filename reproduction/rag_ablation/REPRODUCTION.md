# Preserve the recorded archive during reproduction

The supplied README.md, scripts, data, results, and SHA256SUMS are preserved. Run the added wrapper to verify the checksums and reproduce the results in a temporary copy.

    python -m pip install -r requirements.txt
    python -B reproduce.py

The command requires no API credentials. It writes the regenerated CSV and JSON files to generated/ and compares them with the recorded results/. It also checks all archive hashes before and after the run. Verification results are written to generated/verification.json. Use --output-dir PATH to choose another generated-output directory.

The wrapper was verified with Python 3.12.12 and the installed scientific environment used for the other reproduction folders. The supplied requirements retain the versions recorded with the original archive.

The original validate_rag.py and summarize.py entry points remain available and write to results/ as described in the supplied README. The wrapper preserves those recorded result snapshots during verification.
