#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

command -v cargo >/dev/null || { echo "cargo is required" >&2; exit 1; }
command -v rustup >/dev/null || { echo "rustup is required" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
}

echo "Creating training environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
PYTHON="${VENV_DIR}/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/wasmrl-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/wasmrl-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

"${PYTHON}" -m pip install --upgrade pip wheel
"${PYTHON}" -m pip install "maturin>=1.4,<2.0"

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    echo "Installing PyTorch from the configured wheel index"
    "${PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.0"
else
    "${PYTHON}" -m pip install "torch>=2.0"
fi

"${PYTHON}" -m pip install \
    "numpy>=1.20" \
    "gymnasium>=0.29" \
    "stable-baselines3>=2.0" \
    "tensorboard>=2.10" \
    "tqdm>=4.65" \
    "rich>=13.0"

rustup target add wasm32-wasip2
cargo build \
    --manifest-path "${ROOT_DIR}/Cargo.toml" \
    -p counter-env \
    --target wasm32-wasip2 \
    --release

"${PYTHON}" -m maturin develop \
    --release \
    --manifest-path "${ROOT_DIR}/crates/wasmrl-py/Cargo.toml"

"${PYTHON}" -m unittest discover \
    -s "${ROOT_DIR}/tests/python" \
    -p 'test_*.py' \
    -v

"${PYTHON}" - <<'PY'
import json
import torch
import wasmrl_py

diagnostics = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "wasmrl_py": wasmrl_py.__version__,
}
if torch.cuda.is_available():
    diagnostics["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
print(json.dumps(diagnostics, indent=2))
PY

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    "${PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
        echo "CUDA validation failed. Set TORCH_INDEX_URL to your cluster's CUDA wheel index." >&2
        echo "For CPU-only development, rerun with REQUIRE_CUDA=0." >&2
        exit 1
    }
fi

echo
echo "GPU node setup complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Run training with: just mvp-train cuda"
