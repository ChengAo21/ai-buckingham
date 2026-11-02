"""Validate public links, configured scientific settings and release contents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from public_release import SKIP_DIRS, is_public_path, public_files, credential_paths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ai_buckingham.catalog import CASES
from ai_buckingham.settings import paper_settings


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('src', 'href') and value)


def missing_links(root):
    errors = []
    documents = [root / 'README.md', root / 'docs/index.html']
    documents.extend(p for p in (root / 'docs').glob('*.md'))
    for document in documents:
        content = document.read_text(encoding='utf-8-sig')
        parser = Links()
        parser.feed(content)
        links = parser.links + re.findall(r'\]\(([^\s)]+)', content)
        for link in links:
            parsed = urlsplit(link.strip('<>'))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (document.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f'Missing local link in {document.relative_to(root)}: {link}')
    return errors


def validate_release(root):
    errors = []
    manifest_path = root / 'PUBLIC_RELEASE.json'
    if not manifest_path.exists():
        return ['PUBLIC_RELEASE.json is required for --public validation.']
    record = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = {r['path']: r for r in record['files']}
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith('.egg-info')]
        for name in names:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if relative == 'PUBLIC_RELEASE.json':
                continue
            if not is_public_path(relative) or relative not in entries:
                errors.append('Unexpected public release file: ' + relative)
    for relative, item in entries.items():
        path = root / relative
        if not is_public_path(relative) or not path.is_file():
            errors.append('Missing or disallowed inventory path: ' + relative)
            continue
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            errors.append('Release checksum mismatch: ' + relative)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', action='store_true', help='Also verify the exported release inventory.')
    args = parser.parse_args()
    errors = missing_links(ROOT)
    for key in CASES:
        hp = json.loads((ROOT / f'configs/paper/{key}.json').read_text(encoding='utf-8'))['manual_hyperparameters']
        if hp != paper_settings(key):
            errors.append('Stale config: ' + key)
    errors.extend('Potential credential in ' + p for p in credential_paths(ROOT, public_files(ROOT)))
    provenance = json.loads((ROOT / 'docs/assets/provenance.json').read_text(encoding='utf-8'))
    for item in provenance['public_assets']:
        path = ROOT / item['asset']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Public artwork checksum mismatch: ' + item['asset'])
    if args.public:
        errors.extend(validate_release(ROOT))
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Validated: nine configurations, README/document/page links, selected artwork hashes and credential scan.')
    if args.public:
        print('Public inventory verified; no full manuscripts or private authoring files.')


if __name__ == '__main__':
    main()
