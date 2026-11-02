$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    & $taskPython -m pip install -e '.[app]'
}
& $taskPython -m streamlit run app.py --server.address 127.0.0.1
