#!/usr/bin/env bash
#
# setup.sh — one-shot install + test + benchmark for the mosna fork.
#
# Usage:
#   ./setup.sh            # core install, tests, benchmark
#   ./setup.sh --all      # also install heavy optional backends (spatial,
#                         # integration, deep, gpu) — slow, may fail per platform
#
# Requires `uv` (https://docs.astral.sh/uv/). Install it once with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_ALL=0
[[ "${1:-}" == "--all" ]] && INSTALL_ALL=1

echo "==> Checking for uv"
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
uv --version

echo "==> Creating virtual environment (.venv, Python ${PYTHON_VERSION})"
uv venv --clear --python "${PYTHON_VERSION}" .venv

# Activate so the rest of the script (pytest/python) uses the venv.
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing mosna (editable) + dev + benchmark extras"
uv pip install -e '.[dev,benchmark]'

if [[ "${INSTALL_ALL}" -eq 1 ]]; then
    echo "==> Installing heavy optional backends ([spatial,integration,deep,gpu])"
    echo "    (some may fail depending on your platform/CUDA — that's OK)"
    uv pip install -e '.[spatial,integration,deep]' || \
        echo "    WARN: some optional backends failed to install (continuing)"
    uv pip install -e '.[gpu]' || \
        echo "    WARN: GPU (RAPIDS) backends unavailable on this machine (continuing)"
fi

echo "==> Import smoke test"
python -c "import mosna; print('mosna', mosna.__version__, '| gpu_clustering:', mosna.gpu_clustering)"

echo "==> Running test suite (scientific equivalence + API)"
pytest -q

echo "==> Running I/O benchmark (pandas vs Polars/Parquet)"
python benchmarks/bench_io.py --rows 200000 --cols 40

echo
echo "==> Done. Activate the env in new shells with:  source .venv/bin/activate"
