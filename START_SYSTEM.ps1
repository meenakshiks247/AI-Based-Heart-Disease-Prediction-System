# START_SYSTEM.ps1
# Starts the backend (uvicorn) and optionally the frontend (Vite dev server).

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Heart Disease Prediction System - Startup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── Backend ──────────────────────────────────────────────────────

$backendEntry = ".\backend\app\main.py"

if (Test-Path $backendEntry) {
    Write-Host "[INFO] Starting backend (uvicorn on port 8000)..." -ForegroundColor Green
    try {
        Push-Location backend
        Start-Process -NoNewWindow -FilePath "python" `
            -ArgumentList "-m uvicorn app.main:app --reload --port 8000"
        Pop-Location
        Write-Host "[INFO] Backend started in background." -ForegroundColor Green
    }
    catch {
        Pop-Location
        Write-Host "[ERROR] Failed to start backend: $_" -ForegroundColor Red
    }
}
else {
    Write-Host "[WARN] backend/app/main.py not found - skipping backend startup." -ForegroundColor Yellow
}

Write-Host ""

# ── Frontend ─────────────────────────────────────────────────────

if (Test-Path ".\frontend\package.json") {
    Write-Host "[INFO] frontend found - starting dev server..." -ForegroundColor Green
    try {
        Push-Location frontend
        if (-not (Test-Path ".\node_modules")) {
            Write-Host "[INFO] Installing frontend dependencies (npm install)..."
            npm install
        }
        npm run dev
        Pop-Location
    }
    catch {
        Pop-Location
        Write-Host "[ERROR] Failed to start frontend: $_" -ForegroundColor Red
    }
}
else {
    Write-Host "[WARN] frontend/package.json not found. Skipping frontend start." -ForegroundColor Yellow
}
