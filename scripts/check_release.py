"""Write or verify the distributable file checksums, without API access."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "generated", ".git", ".venv", "venv", "build", "dist"}
FORBIDDEN = {"API_BaseURL.dat", "MAIN-26.pdf", "SI-26.pdf", "DimAgent_26.zip", ".env", "secrets.toml"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def included_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if not path.is_file() or relative.as_posix() == "SHA256SUMS":
            continue
        if path.is_symlink() or path.name in FORBIDDEN or relative.parts[0].lower() == "reference":
            raise ValueError(f"Excluded material in release: {relative.as_posix()}")
        if path.stat().st_size >= 100 * 1024**2:
            raise ValueError(f"File exceeds regular GitHub file limit: {relative.as_posix()}")
        yield path, relative.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Explicitly create/update the release file manifest")
    args = parser.parse_args()
    checksum = ROOT / "SHA256SUMS"
    files = list(included_files())
    if args.write:
        checksum.write_text("".join(f"{digest(path)}  {relative}\n" for path, relative in files), encoding="utf-8", newline="\n")
    expected = {}
    for line in checksum.read_text(encoding="utf-8").splitlines():
        sha, relative = line.split("  ", 1)
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError("Checksum path is outside the repository")
        expected[relative] = sha
    actual_names = {relative for _, relative in files}
    if actual_names != set(expected):
        raise ValueError("Release file set differs from SHA256SUMS")
    for path, relative in files:
        if digest(path) != expected[relative]:
            raise ValueError(f"Changed release file: {relative}")
    print(json.dumps({"status":"passed", "files":len(files), "bytes":sum(p.stat().st_size for p,_ in files),
                      "largest_file_bytes":max(p.stat().st_size for p,_ in files), "credentials_and_manuscripts_excluded_by_filename":True}))


if __name__ == "__main__":
    main()
