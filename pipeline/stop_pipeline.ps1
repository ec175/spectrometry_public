<#
  stop_pipeline.ps1 — stop the pipeline pieces (leaves Docker Desktop running).
  The n8n_data volume persists, so workflows/credentials survive.
#>
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Stopping n8n container..." -ForegroundColor Yellow
Push-Location $here; docker compose down | Out-Null; Pop-Location

Write-Host "Stopping render service..." -ForegroundColor Yellow
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'app\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "  stopped pid $($_.ProcessId)" }

Write-Host "Done. (Tailscale funnel left on; turn off with: tailscale funnel --https=8443 off)" -ForegroundColor Cyan
