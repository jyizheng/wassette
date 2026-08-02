#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
NUM_ENVS="${NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/ppo-counter-e2e}"
COMPONENT="${ROOT_DIR}/target/wasm32-wasip2/release/counter_env.wasm"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/wasmrl-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/wasmrl-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found at ${PYTHON}. Run scripts/setup_gpu_node.sh first." >&2
    exit 1
fi

"${PYTHON}" -c 'import gymnasium, stable_baselines3, torch, wasmrl_py'
cargo build \
    --manifest-path "${ROOT_DIR}/Cargo.toml" \
    -p counter-env \
    --target wasm32-wasip2 \
    --release

"${PYTHON}" -m unittest discover \
    -s "${ROOT_DIR}/tests/python" \
    -p 'test_*.py' \
    -v

"${PYTHON}" "${ROOT_DIR}/examples/python/ppo_training.py" \
    --component "${COMPONENT}" \
    --output "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --num-envs "${NUM_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --seed 42 \
    --verbose 1

"${PYTHON}" "${ROOT_DIR}/examples/python/evaluate_ppo.py" \
    --model "${OUTPUT_DIR}/model.zip" \
    --component "${COMPONENT}" \
    --device "${DEVICE}" \
    --num-envs "${NUM_ENVS}" \
    --episodes "${EVAL_EPISODES}" \
    --min-success-rate 0.90 \
    --min-reward-improvement 0.20

echo "WasmRL PPO end-to-end test passed."
