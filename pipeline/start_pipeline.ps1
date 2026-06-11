<#
  start_pipeline.ps1 - bring the whole self-hosted shorts pipeline up.

  Run in a NORMAL (non-administrator) PowerShell - the render service shells out
  to MiKTeX, which refuses to run elevated.

  Brings up: Docker Desktop -> n8n container -> host render service -> Tailscale
  funnel (8443), then prints the access URLs. Adjust paths/host for your machine.

  Stop later with:  .\stop_pipeline.ps1   (or: docker compose down)
#>
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Daemon {
  docker version --format '{{.Server.Version}}' 2>$null | Out-Null
  return ($LASTEXITCODE -eq 0)
}

# --- 1. Docker daemon ---------------------------------------------------------
if (Test-Daemon) {
  Write-Host "[1/4] Docker daemon: already up" -ForegroundColor Green
} else {
  Write-Host "[1/4] Starting Docker Desktop..." -ForegroundColor Yellow
  Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  $up = $false
  for ($i = 0; $i -lt 40; $i++) {
    if (Test-Daemon) { $up = $true; break }
    Start-Sleep 6
  }
  if (-not $up) {
    Write-Host "      Docker not responding - open Docker Desktop manually, then re-run." -ForegroundColor Red
    return
  }
}

# --- 2. n8n container ---------------------------------------------------------
Write-Host "[2/4] Starting n8n container..." -ForegroundColor Yellow
Push-Location $here
docker compose up -d | Out-Null
Pop-Location
$ok = $false
for ($i = 0; $i -lt 20; $i++) {
  try {
    $r = Invoke-WebRequest "http://localhost:5678/healthz" -UseBasicParsing -TimeoutSec 4
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch {}
  Start-Sleep 3
}
if ($ok) { Write-Host "      n8n: healthy" -ForegroundColor Green }
else { Write-Host "      n8n: not responding yet (docker compose logs -f)" -ForegroundColor Red }

# --- 3. render service --------------------------------------------------------
$svc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'app\.py' }
if ($svc) {
  $rpid = $svc.ProcessId | Select-Object -First 1
  Write-Host "[3/4] Render service: already running (pid $rpid)" -ForegroundColor Green
} else {
  Write-Host "[3/4] Starting render service..." -ForegroundColor Yellow
  Start-Process python -ArgumentList "app.py" -WorkingDirectory (Join-Path $here "render_service") -WindowStyle Hidden
  Write-Host "      Render service: started (http://localhost:8765)" -ForegroundColor Green
}

# --- 4. Tailscale funnel for n8n (8443, for OAuth) ---------------------------
$ts = Get-Command tailscale -ErrorAction SilentlyContinue
if ($ts) {
  $fstatus = tailscale funnel status 2>$null | Out-String
  if ($fstatus -match ':8443') {
    Write-Host "[4/4] Funnel 8443 -> n8n: already on" -ForegroundColor Green
  } else {
    Write-Host "[4/4] Enabling funnel 8443 -> n8n..." -ForegroundColor Yellow
    tailscale funnel --bg --https=8443 5678 | Out-Null
  }
} else {
  Write-Host "[4/4] Tailscale not found - skipping funnel (only needed for remote/OAuth)" -ForegroundColor DarkGray
}

# Read the public base URL from .env if set, else show a placeholder.
$remote = "https://YOUR-TUNNEL-HOST:8443/"
$envFile = Join-Path $here ".env"
if (Test-Path $envFile) {
  $m = Select-String -Path $envFile -Pattern "N8N_EDITOR_BASE_URL=(.*)"
  if ($m) { $remote = $m.Matches.Groups[1].Value.Trim() }
}

Write-Host ""
Write-Host "Pipeline is up." -ForegroundColor Cyan
Write-Host "  n8n editor (local) : http://localhost:5678"
Write-Host "  n8n editor (remote): $remote"
