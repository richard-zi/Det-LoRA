#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found. Installing uv into $HOME/.local/bin ..."
  mkdir -p "$HOME/.local/bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available. Install uv manually and rerun this script." >&2
    exit 1
  fi
}

install_uv

if ! command -v uv >/dev/null 2>&1; then
  echo "uv install finished, but uv is still not on PATH." >&2
  echo "Add this to your shell first: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

echo "Using uv: $(uv --version)"
echo "Syncing project environment ..."
uv sync --frozen --group dev

echo
echo "Bootstrap complete."
echo "Run healthcheck next:"
echo "  ./scripts/healthcheck_cluster.sh"
