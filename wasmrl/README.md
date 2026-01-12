# WasmRL - WebAssembly-based Execution Layer for Reinforcement Learning

WasmRL is a high-performance, security-oriented runtime for executing reinforcement learning environments as WebAssembly components. It builds on top of [Wassette](https://github.com/microsoft/wassette) and provides:

- **In-process execution** with high throughput
- **Batched stepping** for vectorized environment execution
- **Instance pooling** and micro-batching for efficiency
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
│  ├─ Batch Scheduler                     │
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
