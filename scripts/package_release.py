"""Build an allowlisted public archive and, optionally, a clean upload folder."""
import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from public_release import public_files, credential_paths, file_record

ROOT = Path(__file__).resolve().parents[1]


def create_release(root, output, directory=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    directory = Path(directory).resolve() if directory is not None else None
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing archive: ' + str(output))
    if directory is not None:
        if directory == root or directory.is_relative_to(root):
            raise ValueError('The clean upload directory must be outside the working source tree.')
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError('The upload directory must be new or empty: ' + str(directory))
    files = public_files(root)
    suspicious = credential_paths(root, files)
    if suspicious:
        raise ValueError('Potential credentials in: ' + ', '.join(suspicious))
    manifest = {
        'release': 'AI Buckingham public source release',
        'policy': 'Explicit allowlist in scripts/public_release.py',
        'excluded': ['full manuscript and supplementary documents', 'editable slides',
                     'authoring notebooks', 'credentials', 'environments', 'internal delivery files'],
        'files': [file_record(root, p) for p in files],
    }
    manifest_raw = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        for p in files:
            dest = directory / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        (directory / 'PUBLIC_RELEASE.json').write_bytes(manifest_raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            archive.write(p, Path('ai-buckingham') / p.relative_to(root))
        archive.writestr('ai-buckingham/PUBLIC_RELEASE.json', manifest_raw)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'ai-buckingham-public-v0.5.0.zip')
    parser.add_argument('--directory', type=Path, help='Also export a fresh directory for Git Bash upload.')
    args = parser.parse_args()
    manifest = create_release(ROOT, args.output, args.directory)
    print(f'{args.output.resolve()}\n{len(manifest["files"]) + 1} files; {args.output.stat().st_size / 1024**2:.1f} MiB')
    if args.directory:
        print('Clean upload directory: ' + str(args.directory.resolve()))


if __name__ == '__main__':
    main()
