# Cria o venv das engines e instala o pacote basico.
#   -Cuda  : instala torch com CUDA (GPU NVIDIA). Sem a flag, torch de CPU.
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\py"
python -m venv .venv
$pip = ".venv\Scripts\pip.exe"
& $pip install -U pip
if ($Cuda) {
    & $pip install torch --index-url https://download.pytorch.org/whl/cu124
} else {
    & $pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $pip install -e ".[basic,kokoro,youtube]"
Write-Host "ok - rode: .\target\release\dub.exe doctor"
