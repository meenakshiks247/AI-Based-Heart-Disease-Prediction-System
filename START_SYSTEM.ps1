# START_SYSTEM.ps1
# Starts the backend (uvicorn) and the frontend (Vite dev server) in
# separate windows so neither blocks the other.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ProjectRoot "logs"

# Create log directory if missing
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Heart Disease Prediction System - Startup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── Backend ──────────────────────────────────────────────────────────

$backendDir = Join-Path $ProjectRoot "backend"
$backendEntry = Join-Path $backendDir  "app\main.py"

if (Test-Path $backendEntry) {
    Write-Host "[INFO] Starting backend (uvicorn on port 8000)..." -ForegroundColor Green

    # -ArgumentList as an ARRAY so Python receives separate args.
    # Each element becomes one argv entry:
    #   python  -m  uvicorn  app.main:app  --reload  --port  8000
    $backendProc = Start-Process -PassThru `
        -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" `
        -WorkingDirectory $backendDir `
        -RedirectStandardOutput (Join-Path $LogDir "backend_stdout.log") `
        -RedirectStandardError  (Join-Path $LogDir "backend_stderr.log")

    Write-Host "[INFO] Backend started (PID $($backendProc.Id)). Logs: logs\backend_*.log" -ForegroundColor Green
}
else {
    Write-Host "[WARN] backend\app\main.py not found — skipping backend." -ForegroundColor Yellow
}

Write-Host ""

# ── Frontend ─────────────────────────────────────────────────────────

$frontendDir = Join-Path $ProjectRoot "frontend"
$frontendPkgJson = Join-Path $frontendDir "package.json"

if (Test-Path $frontendPkgJson) {
    Write-Host "[INFO] Frontend found — starting Vite dev server..." -ForegroundColor Green

    # Install deps if needed (runs synchronously so node_modules is ready)
    if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
        Write-Host "[INFO] Installing frontend dependencies (npm install)..."
        Start-Process -Wait -NoNewWindow `
            -FilePath "npm" `
            -ArgumentList "install" `
            -WorkingDirectory $frontendDir
    }

    # Start Vite in a separate process (non-blocking)
    $frontendProc = Start-Process -PassThru `
        -FilePath "npm" `
        -ArgumentList "run", "dev" `
        -WorkingDirectory $frontendDir `
        -RedirectStandardOutput (Join-Path $LogDir "frontend_stdout.log") `
        -RedirectStandardError  (Join-Path $LogDir "frontend_stderr.log")

    Write-Host "[INFO] Frontend started (PID $($frontendProc.Id)). Logs: logs\frontend_*.log" -ForegroundColor Green
}
else {
    Write-Host "[WARN] frontend\package.json not found — skipping frontend." -ForegroundColor Yellow
}

# ── Summary ──────────────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Both servers launched!" -ForegroundColor Green
Write-Host "  Backend  → http://127.0.0.1:8000"         -ForegroundColor White
Write-Host "  Frontend → http://127.0.0.1:5173"         -ForegroundColor White
Write-Host "  Logs     → $LogDir"                        -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C here then close the server windows to stop." -ForegroundColor DarkGray
Write-Host ""

# Keep script alive so user can see output; killing this does NOT kill
# the child processes (they are independent).
try {
    if ($backendProc) { $backendProc.WaitForExit() }
    if ($frontendProc) { $frontendProc.WaitForExit() }
}
catch {
    # Ctrl+C pressed — script exits, servers keep running
}
