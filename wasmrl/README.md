# WasmRL - WebAssembly-based Execution Layer for Reinforcement Learning

WasmRL is a high-performance, security-oriented runtime for executing reinforcement learning environments as WebAssembly components. It builds on top of [Wassette](https://github.com/microsoft/wassette) and provides:

- **In-process execution** with high throughput
- **Vectorized stepping API** for RL framework integration
- **Instance pooling** for reusable Wasm environments
- **Snapshot/restore** for reset-heavy workloads
- **Resource budgets** (fuel, memory, timeout) with telemetry
- **Python VecEnv API** for integration with RL algorithms like PPO

## Architecture

```
┌─────────────────────────────────────────┐
│  Python (VecEnv API) / PPO Training     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      wasmrl-py (Python Bindings)        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   wasmrl-runtime (In-process Executor)  │
│  ├─ Instance Pool                       │
│  ├─ Vector API Dispatcher               │
│  ├─ Resource Budgets                    │
│  └─ Metrics/Telemetry                   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Wasmtime + WebAssembly Components     │
└─────────────────────────────────────────┘
```

## Quick Start

### Building

```bash
cd wasmrl
cargo build --workspace
```

### Running Tests

```bash
cargo test --workspace
```

## PPO MVP on a GPU Node

The MVP runs Wasm environments on CPU through Wasmtime and places the PPO
policy network on a CUDA GPU through PyTorch. The current runtime dispatches
the vector environment handles serially, so increasing `num_envs` improves
rollout batching but does not yet parallelize Wasm execution.

### One-time setup

The node needs Rust, Python 3, a working NVIDIA driver, and a CUDA-enabled
PyTorch wheel. From this directory:

```bash
# Optional when the cluster uses a dedicated PyTorch wheel index:
export TORCH_INDEX_URL="CLUSTER_APPROVED_PYTORCH_INDEX"

scripts/setup_gpu_node.sh
source .venv/bin/activate
```

`setup_gpu_node.sh` creates `.venv`, installs the training dependencies,
builds the real `counter_env.wasm` component, installs `wasmrl_py`, and fails
if PyTorch cannot see CUDA. Set `REQUIRE_CUDA=0` for a CPU-only development
machine. Choose the CUDA wheel index that matches the driver policy on the
target cluster when setting `TORCH_INDEX_URL`.

### Train and evaluate

```bash
# Arguments: device, num_envs, total_timesteps, output directory
just mvp-train cuda 32 100000 artifacts/gpu-run

# Arguments: device, num_envs, evaluation episodes, output directory
just mvp-evaluate cuda 32 100 artifacts/gpu-run
```

Training writes `model.zip`, `training.json`, and TensorBoard events under the
output directory. Evaluation writes `evaluation.json`, compares the trained
policy with a seeded random baseline, and returns a non-zero exit code unless
the configured success and reward-improvement thresholds pass.

Run the complete thresholded flow with:

```bash
just mvp-e2e cuda 32 100000
```

Without `just`, use the scripts directly:

```bash
DEVICE=cuda NUM_ENVS=32 TOTAL_TIMESTEPS=100000 scripts/mvp_e2e.sh
```

To inspect GPU use while training:

```bash
nvidia-smi -l 1
tensorboard --logdir artifacts/gpu-run/tensorboard
```

The CounterEnv policy is intentionally tiny, so high GPU utilization is not an
MVP acceptance criterion. The acceptance criterion is a real Wasm-to-PPO
training loop whose trained success rate and reward beat the random baseline.

### Code Formatting & Linting

```bash
# Format code
cargo +nightly fmt --all

# Check formatting
cargo +nightly fmt --all -- --check

# Run linter
cargo clippy --workspace

# Security checks
cargo deny check all
```

### Development Commands

```bash
# Build + test + lint (CI)
just ci

# Run tests with output
just test-verbose

# Clean build
just clean
```

## Project Structure

```
wasmrl/
├── crates/
│   ├── wasmrl-wit/          # WIT interface definitions
│   ├── wasmrl-runtime/      # In-process execution runtime
│   └── wasmrl-sdk-rust/     # Rust SDK for env authors
├── envs/
│   └── counter_env/         # Example: simple counter environment
├── docs/                    # Design docs and guides
├── deny.toml               # Supply chain security policy
├── Justfile                # Development commands
└── Cargo.toml              # Workspace configuration
```

## Development Workflow

1. **Make changes**: Edit crate source files
2. **Format**: `cargo +nightly fmt --all`
3. **Test**: `cargo test --workspace`
4. **Lint**: `cargo clippy --workspace`
5. **Security**: `cargo deny check all`
6. **Submit**: Ensure `just ci` passes

## Milestones

- **M0**: ✓ Bootstrap (repo + CI + cargo deny)
- **M1**: Freeze WIT ABI v0
- **M2**: Minimal environment components
- **M3**: Data plane runtime v0
- **M4**: Snapshot/restore
- **M5**: Policies, budgets, telemetry
- **M6**: Python VecEnv + PPO
- **M7**: Benchmark harness
- **M8**: Wassette/MCP bridge
- **M9**: CAMEL-AI comparisons

See [../todo.md](../todo.md) for detailed task breakdown.

## License

MIT License. See LICENSE for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
