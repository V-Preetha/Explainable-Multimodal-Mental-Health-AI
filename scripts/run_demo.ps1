$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "Terminal 1: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8100"
Write-Host "Terminal 2: cd frontend; pnpm dev"
Write-Host "The frontend defaults to clearly labeled demo mode."
