#!/usr/bin/env bash
# Cria o venv das engines e instala o pacote básico. Use --cuda para torch com CUDA.
set -euo pipefail
cd "$(dirname "$0")/../py"
python3 -m venv .venv
PIP=.venv/bin/pip
$PIP install -U pip
if [[ "${1:-}" == "--cuda" ]]; then
  $PIP install torch --index-url https://download.pytorch.org/whl/cu124
fi
$PIP install -e ".[basic,kokoro,youtube]"
echo "ok — rode: cargo run -- doctor"
