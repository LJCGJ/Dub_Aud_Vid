# Cria o venv das engines e instala o pacote básico. Use -Cuda para torch com CUDA.
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\py"
python -m venv .venv
$pip = ".venv\Scripts\pip.exe"
& $pip install -U pip
if ($Cuda) { & $pip install torch --index-url https://download.pytorch.org/whl/cu124 }
& $pip install -e ".[basic,kokoro,youtube]"
Write-Host "ok — rode: cargo run -- doctor"
