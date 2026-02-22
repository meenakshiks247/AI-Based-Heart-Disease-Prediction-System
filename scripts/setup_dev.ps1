# scripts/setup_dev.ps1
Write-Host "Installing python requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt
if (Test-Path "backend/requirements-dev.txt") {
    pip install -r backend/requirements-dev.txt
}
Write-Host "Done. Run: python -m pytest -q"
