"""Prevent private authoring files from entering public upload packages."""
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from public_release import is_public_path
from package_release import create_release


@pytest.mark.parametrize('path', [
    'papers/main.pdf', 'docs/papers/si.pdf', 'docs/assets/unpublished.pdf',
    'data/notes.docx', 'docs/assets/ai-buckingham-visuals.pptx',
    'src/ai_buckingham/notes.ipynb', '.env', '.streamlit/secrets.toml',
    'results/local/session.json', 'docs/DESIGN_NOTES.md',
    'docs/assets/previews/internal.png', 'scripts/prepare_visual_sources.py',
    'unexpected.txt', '../README.md', 'C:/private/README.md',
])
def test_private_paths_are_never_public(path):
    assert not is_public_path(path)


def test_package_contains_selected_assets_but_no_private_documents(tmp_path):
    root = tmp_path / 'source'
    entries = {
        'README.md': '# Public project',
        'app.py': '# Public application',
        'docs/assets/results/mhd.png': 'original artwork placeholder',
        'docs/assets/streamlit-explorer.png': 'app preview placeholder',
        'papers/main.pdf': 'unpublished manuscript',
        'docs/assets/private.pdf': 'unpublished supplement',
        'docs/assets/source.pptx': 'editable source',
        '.env': 'local credentials',
        'results/local/run.json': 'private local run',
    }
    for name, text in entries.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    archive = tmp_path / 'public.zip'
    directory = tmp_path / 'upload'
    record = create_release(root, archive, directory)
    expected = {name for name in entries if is_public_path(name)}
    assert {x['path'] for x in record['files']} == expected
    with ZipFile(archive) as z:
        assert set(z.namelist()) == {'ai-buckingham/' + x for x in expected | {'PUBLIC_RELEASE.json'}}
        for item in record['files']:
            raw = z.read('ai-buckingham/' + item['path'])
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            assert raw == (directory / item['path']).read_bytes()
    assert json.loads((directory / 'PUBLIC_RELEASE.json').read_text()) == record
    with pytest.raises(FileExistsError):
        create_release(root, tmp_path / 'second.zip', directory)


def test_credential_scan_prevents_archive_creation_without_echoing_key(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    secret = 'sk-' + 'a' * 30
    (root / 'app.py').write_text(secret)
    archive = tmp_path / 'public.zip'
    with pytest.raises(ValueError) as error:
        create_release(root, archive)
    assert secret not in str(error.value)
    assert not archive.exists()
