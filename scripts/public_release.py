"""Explicit public-file policy shared by packaging and validation."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import re

ROOT_FILES = {
    'app.py', 'README.md', 'LICENSE', 'CITATION.cff', 'CHANGELOG.md',
    'CONTRIBUTING.md', 'SECURITY.md', 'pyproject.toml', 'requirements.txt',
    'requirements-lock.txt', 'Start.ps1', '.gitignore', '.env.example',
}
DOCS = {'index.html', '.nojekyll', 'ARCHITECTURE.md', 'DATA_SOURCES.md',
        'DEPLOYMENT.md', 'PARAMETERS.md', 'REPRODUCIBILITY.md', 'VISUAL_STYLE.md'}
ASSETS = {'site.css', 'site.js', 'research-data.js', 'plotly.min.js', 'favicon.svg',
          'discovery-map.svg', 'agent-robot.png', 'framework-original.png',
          'streamlit-explorer.png', 'provenance.json', 'visual-provenance.json'}
RESULT_IMAGES = {'mhd.png', 'complete-similarity.png',
                 'incomplete-similarity.png', 'multiple-regimes.png'}
SCRIPTS = {'build_discovery_map.py', 'build_regime_learning.py',
           'build_research_assets.py', 'build_trajectory_previews.py',
           'export_parameter_inventory.py', 'recompute_regimes.py',
           'package_release.py', 'public_release.py', 'validate_repository.py'}
SKIP_DIRS = {'.venv', '.git', '__pycache__', '.pytest_cache', '_Workspace_',
             'outcmaes', 'local', 'build', 'dist'}
PRIVATE_SUFFIXES = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.ipynb', '.tex',
                    '.bib', '.zip', '.pyc', '.log'}
KEY_PATTERNS = [re.compile(r'\bsk-[A-Za-z0-9_-]{18,}'),
                re.compile(r'\blsv2_[A-Za-z0-9_-]{18,}')]
TEXT_SUFFIXES = {'.py', '.md', '.json', '.js', '.toml', '.txt', '.yml', '.cff', '.example'}


def is_public_path(path):
    """An unknown path is private until explicitly admitted here."""
    p = PurePosixPath(str(path).replace('\\', '/'))
    parts = p.parts
    if p.is_absolute() or not parts or '..' in parts or ':' in parts[0]:
        return False
    if any(x in SKIP_DIRS or x.startswith('_private') or x.endswith('.egg-info') for x in parts):
        return False
    if p.suffix.lower() in PRIVATE_SUFFIXES:
        return False
    if len(parts) == 1:
        return p.name in ROOT_FILES
    if parts[0] == 'src':
        return parts[:2] == ('src', 'ai_buckingham') and p.suffix == '.py'
    if parts[0] == 'tests':
        return len(parts) == 2 and p.suffix == '.py'
    if parts[0] == 'scripts':
        return len(parts) == 2 and p.name in SCRIPTS
    if parts[0] == 'configs':
        return (len(parts) == 2 and p.suffix in {'.json', '.csv'}) or (
            len(parts) == 3 and parts[1] == 'paper' and p.suffix == '.json')
    if parts[0] == 'data':
        return len(parts) == 2 and p.suffix in {'.csv', '.md', '.json'}
    if parts[0] == 'results':
        return len(parts) == 3 and parts[1] in {'reference', 'computed', 'previews'} and p.suffix == '.json'
    if parts[0] == '.github':
        return len(parts) == 3 and parts[1] == 'workflows' and p.name in {'ci.yml', 'pages.yml'}
    if parts[0] == '.streamlit':
        return len(parts) == 2 and p.name in {'config.toml', 'secrets.toml.example'}
    if parts[0] != 'docs':
        return False
    if len(parts) == 2:
        return p.name in DOCS
    if parts[1] == 'data':
        return len(parts) == 3 and p.suffix == '.csv'
    if parts[1] != 'assets':
        return False
    if len(parts) == 3:
        return p.name in ASSETS
    if len(parts) != 4:
        return False
    if parts[2] == 'results':
        return p.name in RESULT_IMAGES
    if parts[2] == 'schematics':
        return p.suffix == '.png'
    if parts[2] == 'formulas':
        return p.suffix == '.svg'
    if parts[2] == 'workflow':
        return p.name.endswith('-original.png') or p.name == 'original-artwork-provenance.json'
    return False


def public_files(root):
    root = Path(root).resolve()
    found = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('_private')
                   and not d.endswith('.egg-info') and not (Path(current) / d).is_symlink()]
        for name in names:
            p = Path(current) / name
            if is_public_path(p.relative_to(root).as_posix()):
                if p.is_symlink() or not p.resolve().is_relative_to(root):
                    raise ValueError('Public files must be ordinary files inside the source tree.')
                found.append(p)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def credential_paths(root, files):
    """Return paths only; never include matched values in diagnostics."""
    return [p.relative_to(root).as_posix() for p in files
            if p.suffix.lower() in TEXT_SUFFIXES
            and any(pattern.search(p.read_text(encoding='utf-8', errors='ignore')) for pattern in KEY_PATTERNS)]


def file_record(root, path):
    raw = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}
