#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.10}"
if [[ "${1:-}" != --install ]]; then
  echo 'Usage: CUDA_HOME=/path/to/cuda13 PYTHON_BIN=python3.10 scripts/setup.sh --install'
  echo 'Creates only package .venv and installs pinned user-space dependencies. No driver/toolkit changes.'
  exit 0
fi
: "${CUDA_HOME:?Set an existing CUDA 13 toolkit explicitly; do not silently select a toolkit}"
"$CUDA_HOME/bin/nvcc" --version
"$python_bin" -c 'import sys,platform; assert platform.system()=="Linux" and platform.machine()=="x86_64", "lock targets Linux x86_64"; assert sys.version_info[:2] == (3,10), "Lock targets Linux x86_64 Python3.10; validate a new lock for another platform"'
"$python_bin" -m venv "$package_dir/.venv"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$package_dir/.venv/bin/python" -r "$package_dir/configs/runtime.lock.txt" -r "$package_dir/configs/tracer_requirements.lock.txt"
else
  "$package_dir/.venv/bin/python" -m pip install -r "$package_dir/configs/runtime.lock.txt" -r "$package_dir/configs/tracer_requirements.lock.txt"
fi
"$package_dir/.venv/bin/python" -m pip freeze > "$package_dir/results/installed_runtime.lock.txt"
"$package_dir/.venv/bin/python" "$package_dir/scripts/validate_package.py"
