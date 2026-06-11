<#
  run.ps1 - start the host-side manim render bridge for n8n.

  Run this in a NORMAL (non-administrator) PowerShell, because MiKTeX refuses to
  compile LaTeX under elevation and the manim scenes use MathTex.

  Uses the global Python (stdlib only - no installs), and shells out to the
  manim venv for the actual render.
#>
param(
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8765,
  [string]$Quality = "h"          # manim l/m/h/p/k
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:RENDER_HOST = $BindHost
$env:RENDER_PORT = "$Port"
$env:RENDER_QUALITY = $Quality

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { throw "python not found on PATH" }

Write-Host "Starting render_service on ${BindHost}:${Port} (quality -q$Quality)..."
& $py (Join-Path $here "app.py")
