# WasmRL Design Documentation

## Overview

WasmRL is a high-performance runtime layer built on top of Wassette for executing reinforcement learning environments as WebAssembly components. This document outlines the design principles, architecture, and key components.

## Design Principles

1. **Security First**: Deny-by-default permissions, resource budgets (fuel, memory, time)
2. **Performance**: In-process execution, batching, instance pooling
3. **Determinism**: Fixed seeds produce identical trajectories
4. **Composability**: Generic Wasm components with no WasmRL-specific dependencies
5. **Observability**: Comprehensive metrics and telemetry

## Core Components

### WIT Interface (wasmrl-wit)

The WIT (WebAssembly Interface Types) defines the contract all environments must implement:

```wit
init(config: list<u8>) -> result<u64, string>
reset(seed: u64) -> result<list<u8>, string>
step(action: list<u8>) -> result<list<u8>, string>
reset_batch(seeds: list<u64>) -> result<list<list<u8>>, string>
step_batch(actions: list<list<u8>>) -> result<list<list<u8>>, string>
snapshot() -> result<list<u8>, string>
restore(snapshot: list<u8>) -> result<unit, string>
```

### Runtime (wasmrl-runtime)

Manages in-process execution of Wasm components:

- **InstanceHandle**: Lightweight reference to a running environment
- **InstancePool**: Manages a pool of pre-warmed instances
- **Scheduler**: Coordinates batch operations across instances
- **ResourceBudgets**: Enforces memory, fuel, and timeout limits
- **Metrics**: Tracks throughput, latency (p50/p99), determinism

### SDK (wasmrl-sdk-rust)

Utilities for Rust environment authors:

- **DeterministicRng**: Cross-platform PRNG for reproducibility
- **TensorMetadata**: Encoding/decoding for observations and actions
- **SnapshotHelpers**: Versioned serialization for state snapshots

## Execution Model

### Single-Step Mode
```
Action -> [Wasm] -> (Observation, Reward, Done)
```

### Batch Mode
```
Actions[] -> [Wasm Instance Pool] -> (Observations[], Rewards[], Dones[])
```

Batch mode exploits:
- Parallelism across instances
- Micro-batching (queue aggregation)
- Snapshot/restore reuse patterns

## Performance Targets

- **Throughput**: `step_batch` >= 1.2× scalar loop at N>=256
- **Reset latency**: `restore` >= 2× improvement for reset-heavy workloads
- **Overhead**: budgets ON adds <10% cost

## Milestone M0 (Complete)

- Repository scaffolding
- Workspace and crate structure
- Cargo.toml configuration
- Cargo deny security policy
- Basic CI commands (format, lint, test)
- Initial tests passing

## Milestone M1 (Complete)

- WIT interface frozen: `wasmrl:env@0.1.0`
- Core types: DType, Tensor, StepResult, EnvHandle, EnvConfig
- Batch types: BatchStepResult
- Snapshot types: SnapshotData
- Traits: WasmRLEnvironment, WasmRLBatch, WasmRLSnapshot
- SDK utilities: TensorEncoder/Decoder, DeterministicRng, SnapshotHelper
- 46 tests total
- WIT_SPEC.md documentation

## Milestone M2 (Complete)

- CounterEnv: Simple counter environment reaching target value
  - Observation: single f32 (counter value)
  - Action: discrete i32 (0=decrement, 1=increment, 2=noop)
  - Reward: +1 at target, -0.01 per step
  - 11 tests including determinism verification

- Security Suite:
  - MaliciousLoopEnv: Infinite loop trigger for fuel/timeout testing (6 tests)
  - MaliciousMemoryEnv: Memory exhaustion for memory limit testing (5 tests)

- Determinism Testing:
  - 20-run trajectory hash verification
  - Snapshot/restore determinism
  - Cross-instance reproducibility

- Total tests: 79 across all crates

## Milestone M3 (Complete)

Data Plane Runtime v0 - In-process executor with instance pooling and batch scheduling.

### Runtime Architecture

```
┌─────────────────────────────────────────────────────┐
│                  WasmEnvFactory                      │
│  - ComponentRef (bytes/file/OCI)                    │
│  - PolicyConfig                                      │
│  - spawn(n) -> Vec<InstanceHandle>                  │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                   EnvRuntime                         │
│  - init(handle, config)                             │
│  - reset(handle, seed) -> obs                       │
│  - step(handle, action) -> StepResult               │
│  - reset_many(handles, seeds) -> obs[]              │
│  - step_many(handles, actions) -> BatchStepResult   │
│  - snapshot(handle) / restore(handle, snapshot)    │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  InstancePool                        │
│  - allocate() / allocate_many(n)                    │
│  - release(handle)                                  │
│  - mark_error(handle, fatal)                        │
│  - recycle(handle)                                  │
│  - SharedPool for thread-safe access                │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                 RuntimeMetrics                       │
│  - step_latency: LatencyStats (p50/p99/p999)       │
│  - reset_latency: LatencyStats                      │
│  - batch_step_latency: LatencyStats                 │
│  - throughput_steps_per_sec()                       │
│  - traps_count, timeouts_count, instances_recycled │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **WasmEnvFactory**: Loads components and spawns instances
   - `new(component_ref, policy) -> factory`
   - `spawn(n) -> Vec<InstanceHandle>` with pre-warming support
   - `FactoryBuilder` for fluent configuration

2. **EnvRuntime**: Executes environment operations
   - Single-instance: `reset()`, `step()`
   - Batch operations: `reset_many()`, `step_many()`
   - State management: `snapshot()`, `restore()`

3. **InstancePool**: Manages instance lifecycle
   - LIFO ready queue for cache locality
   - Error marking and recycling
   - Thread-safe via `SharedPool` wrapper

4. **RuntimeMetrics**: Latency statistics
   - Per-operation timing (step, reset, batch)
   - Percentile tracking (p50, p99, p999)
   - Error and throughput counters

5. **EngineContext**: Wasmtime integration
   - Shared engine and linker
   - Fuel metering configuration
   - WASI support for components

### Module Structure

```
crates/wasmrl-runtime/src/
├── lib.rs          # Public API exports
├── config.rs       # RuntimeConfig, PolicyConfig
├── error.rs        # RuntimeError, RuntimeResult
├── engine.rs       # EngineContext, EnvState
├── factory.rs      # WasmEnvFactory, FactoryBuilder, ComponentRef
├── instance.rs     # InstanceHandle, InstanceStatus, InstanceInfo
├── pool.rs         # InstancePool, SharedPool, PoolStats
├── metrics.rs      # LatencyStats, RuntimeMetrics, Timer
└── executor.rs     # EnvRuntime, BatchExecutor
```

### Test Coverage

- 68 unit tests in runtime modules
- 18 M3-specific integration tests
- Total: 161 tests across project

## Milestone M4 (Complete)

Snapshot/Restore + Reset-heavy Wins - Fast reset optimization using cached snapshots.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 SharedSnapshotCache                  │
│  - LRU eviction with max_entries and max_bytes      │
│  - Thread-safe via Arc<Mutex>                       │
│  - Hit/miss rate tracking                           │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                FastResetManager                      │
│  - cache_initial_state(key, snapshot)               │
│  - get_cached_state(key) -> Option<snapshot>        │
│  - record_full_reset(duration)                      │
│  - record_fast_reset(duration)                      │
│  - metrics() -> FastResetMetrics                    │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  ReplayManager                       │
│  - create_recorder(instance_id)                     │
│  - record_action(action, reward, term, trunc)       │
│  - record_checkpoint(snapshot)                      │
│  - get_replay_data(id) -> ReplayData               │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **SnapshotCache** (`snapshot_cache.rs`):
   - `SnapshotKey`: component_hash + seed for cache lookup
   - `CachedSnapshot`: data with touch time for LRU tracking
   - `SnapshotCache`: LRU with max entries and byte limits
   - `SharedSnapshotCache`: Thread-safe Arc<Mutex> wrapper
   - Features: hit/miss stats, age tracking, byte-aware eviction

2. **FastResetManager** (`fast_reset.rs`):
   - `FastResetConfig`: enabled, auto_cache, limits
   - `FastResetMetrics`: full/fast counts, timings, speedup_ratio()
   - `ResetResult`: enum distinguishing fast vs full reset
   - Automatic caching of initial states

3. **ReplayRecorder** (`replay.rs`):
   - `ReplayConfig`: snapshot_interval, max_snapshots
   - `Checkpoint`: step + snapshot for replay points
   - `RecordedAction`: action + reward + termination
   - `ReplayData`: serializable JSON format for debugging
   - `ReplayManager`: multi-instance recorder coordination

4. **ResetHeavyEnv** (`envs/reset_heavy_env/`):
   - Configurable grid world (default 100x100)
   - Large state (~10KB) for reset benchmarking
   - Short episodes (default max 50 steps)
   - Deterministic seeded RNG
   - Full snapshot/restore support

### Module Structure

```
crates/wasmrl-runtime/src/
├── snapshot_cache.rs  # LRU snapshot caching (13 tests)
├── fast_reset.rs      # Fast reset management (11 tests)
└── replay.rs          # Replay recording (13 tests)

envs/reset_heavy_env/
└── src/lib.rs         # Reset-heavy benchmark env (11 tests)

tests/
└── m4_fast_reset_test.rs  # Integration tests (16 tests)
```

### Performance Design

Fast reset workflow:
```
First reset (seed=42):
  1. Full reset: initialize env from scratch (~50ms)
  2. Cache snapshot with key (component_hash, 42)
  3. Record timing

Subsequent reset (seed=42):
  1. Check cache: found!
  2. Restore from cached snapshot (~0.5ms)
  3. Record fast reset timing
  4. Speedup: 100x
```

Replay workflow:
```
During rollout:
  1. Record initial state + seed
  2. Every K steps: save checkpoint (snapshot)
  3. Record each action + reward

For debugging:
  1. Find nearest checkpoint before bug
  2. Restore snapshot
  3. Replay actions deterministically
```

### Test Coverage

- 37 unit tests in M4 modules
- 16 M4 integration tests
- 11 reset_heavy_env tests
- **Total M4: 64 tests**
- **Total project: 225 tests**

## Milestone M5 (Complete)

Policies, Budgets, and Telemetry - Comprehensive policy system for resource control and security.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Policy                          │
│  ├── MemoryLimit (max_mb, initial_mb, pages)        │
│  ├── FuelBudget (per_step, per_reset, per_batch)    │
│  ├── TimeoutConfig (step_ms, reset_ms, batch_ms)    │
│  ├── WasiConfig (filesystem, network, env)          │
│  └── CapabilityConfig (simd, threading, etc.)       │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                BudgetEnforcer                        │
│  - check_memory(bytes) -> EnforcementResult         │
│  - check_fuel(operation) -> EnforcementResult       │
│  - check_read/write(path) -> EnforcementResult      │
│  - check_network() -> EnforcementResult             │
│  - record_fuel(consumed)                             │
│  - timed_guard(operation) -> TimedGuard             │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│               TelemetryCollector                     │
│  - record_step(duration, fuel)                      │
│  - record_reset(duration, fuel)                     │
│  - record_budget_overrun(type, limit, actual)       │
│  - record_capability_denial(cap, reason)            │
│  - record_trap(code, message)                       │
│  - snapshot() -> PolicyTelemetry                    │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Policy Schema** (`schema.rs`):
   - `Policy`: Complete configuration container
   - `PolicyBuilder`: Fluent API for policy construction
   - `MemoryLimit`: Memory caps with Wasm page conversion
   - `FuelBudget`: Per-operation fuel budgets
   - `TimeoutConfig`: Per-operation timeouts
   - `WasiConfig`: Filesystem, network, env capabilities
   - `CapabilityConfig`: Wasm feature flags

2. **Budget Enforcement** (`enforcement.rs`):
   - `BudgetEnforcer`: Runtime enforcement checks
   - `EnforcementResult`: Allowed/Denied/Exceeded status
   - `EnforcementAction`: Details of enforcement decision
   - `TimedGuard`: RAII timeout tracking

3. **Telemetry** (`telemetry.rs`):
   - `TelemetryCollector`: Event recording
   - `PolicyTelemetry`: Aggregated statistics
   - `BudgetOverrun`: Budget violation details
   - `CapabilityDenial`: Access denial details
   - `TrapInfo`: Crash/trap information

4. **Error Handling** (`error.rs`):
   - `PolicyError`: Typed error variants
   - Parse/validation/enforcement errors
   - Budget/capability/timeout exceeded

### Policy Schema Format

```toml
[memory]
max_mb = 64
initial_mb = 16
enabled = true

[fuel]
per_step = 1000000
per_reset = 5000000
per_batch = 10000000
enabled = true

[timeout]
step_ms = 100
reset_ms = 500
batch_ms = 1000
enabled = true

[wasi]
filesystem_read = ["/data"]
filesystem_write = ["/tmp"]
network = false  # deny by default
clock = true
random = true

[capabilities]
threading = false
simd = true
component_model = true
```

### Security Design

Deny-by-default principles:
1. **Network**: OFF unless explicitly enabled
2. **Filesystem**: No access unless paths whitelisted
3. **Memory**: Hard cap at configured maximum
4. **CPU**: Fuel budget per operation type
5. **Time**: Timeout per operation type

### Module Structure

```
crates/wasmrl-policy/src/
├── lib.rs          # Public API exports (5 tests)
├── schema.rs       # Policy structures (18 tests)
├── error.rs        # Error types (6 tests)
├── enforcement.rs  # Budget enforcement (17 tests)
└── telemetry.rs    # Event tracking (15 tests)

tests/
└── m5_policy_test.rs  # Integration tests (29 tests)
```

### Test Coverage

- 61 unit tests in policy modules
- 29 M5 integration tests
- **Total M5: 90 tests**
- **Total project: 315 tests**

## Milestone M6 (Complete)

Python VecEnv + PPO Integration - Python bindings for seamless RL training integration.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Python Layer                       │
│  import wasmrl_py as wasmrl                         │
│  env = wasmrl.WasmVecEnv("counter.wasm", config)    │
│  obs, info = env.reset()                            │
│  obs, rewards, dones, truncs, info = env.step(act)  │
└───────────────────┬─────────────────────────────────┘
                    │ PyO3 FFI
┌───────────────────▼─────────────────────────────────┐
│                  wasmrl-py Crate                     │
│  ├── PyWasmVecEnv: Vectorized environment          │
│  ├── PyWasmEnv: Single environment wrapper          │
│  ├── PyEnvConfig: Configuration class               │
│  ├── PyBox/PyDiscrete: Gymnasium-compatible spaces │
│  └── PyTensor: NumPy array conversions             │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│               wasmrl-runtime                         │
│  EnvFactory → EnvPool → WasmEnvInstance            │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **PyWasmVecEnv** (`vecenv.rs`):
   - Gymnasium VecEnv compatible interface
   - `reset(seed=None)` → `(obs, info)`
   - `step(actions)` → `(obs, rewards, terminated, truncated, info)`
   - `snapshot_all()` / `restore_all(snapshots)`
   - `sample_actions()` for random policy
   - Auto-reset support for continuous training
   - Thread-safe parallel execution

2. **PyWasmEnv** (`env.rs`):
   - Single environment wrapper
   - Standard Gymnasium interface
   - Snapshot/restore support
   - Episode tracking (rewards, lengths)

3. **PyEnvConfig** (`config.rs`):
   - `num_envs`: Parallel environment count
   - `max_memory_mb`: Per-environment memory limit
   - `fuel_per_step`: Compute budget per step
   - `timeout_step_ms`: Per-step timeout
   - `auto_reset`: Auto-reset on episode end
   - `seed`: Optional random seed

4. **Spaces** (`spaces.rs`):
   - `PyBox`: Continuous observation/action space
   - `PyDiscrete`: Discrete action space
   - `PyMultiDiscrete`: Multiple discrete spaces
   - Gymnasium-compatible `sample()` and `contains()`

5. **Tensor Conversion** (`tensor.rs`):
   - NumPy ↔ WasmRL Tensor conversion
   - Batch tensor stacking for VecEnv
   - Support for i32, i64, f32, f64 dtypes

### Module Structure

```
crates/wasmrl-py/
├── Cargo.toml         # PyO3/maturin configuration
├── pyproject.toml     # Python package metadata
├── README.md          # Python API documentation
└── src/
    ├── lib.rs         # Module entry + pymodule (1 test)
    ├── error.rs       # Python error types (3 tests)
    ├── config.rs      # EnvConfig class (3 tests)
    ├── spaces.rs      # Gymnasium spaces (12 tests)
    ├── tensor.rs      # NumPy conversions (6 tests)
    ├── env.rs         # Single env wrapper (2 tests)
    └── vecenv.rs      # VecEnv implementation (2 tests)

examples/python/
├── basic_usage.py     # Basic WasmVecEnv usage (152 lines)
└── ppo_training.py    # SB3 PPO training example (325 lines)

tests/
└── m6_python_test.rs  # Integration tests (16 tests)
```

### Python API Example

```python
import wasmrl_py as wasmrl
import numpy as np

# Configuration
config = wasmrl.EnvConfig(
    num_envs=8,
    max_memory_mb=64,
    fuel_per_step=1_000_000,
    auto_reset=True,
    seed=42,
)

# Create vectorized environment
env = wasmrl.WasmVecEnv("counter_env.wasm", config)

# Standard Gymnasium interface
obs, info = env.reset()
for _ in range(1000):
    actions = env.sample_actions()  # or np.random.randint(...)
    obs, rewards, terminated, truncated, info = env.step(actions)

# Snapshot/restore for MCTS or debugging
snapshots = env.snapshot_all()
# ... take actions ...
env.restore_all(snapshots)

env.close()
```

### Stable-Baselines3 Integration

```python
from wasmrl_py import WasmVecEnv, EnvConfig
from stable_baselines3 import PPO

# WasmVecEnv is SB3-compatible
config = EnvConfig(num_envs=8, auto_reset=True)
env = WasmVecEnv("counter_env.wasm", config)

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
```

### Build Instructions

```bash
# Install maturin
pip install maturin

# Development build
cd wasmrl/crates/wasmrl-py
maturin develop

# Release wheel
maturin build --release
pip install target/wheels/wasmrl_py-*.whl
```

### Test Coverage

- 29 unit tests in wasmrl-py modules
- 16 M6 integration tests
- **Total M6: 45 tests**
- **Total project: 360 tests**

## Milestone M7 (Complete)

Throughput Benchmarks + Optimization - Comprehensive benchmark suite for validating performance targets.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│               wasmrl-bench Crate                     │
│  ├── lib.rs: Timer, TimingResult, measure()         │
│  ├── stats.rs: RunningStats, Histogram, Comparison │
│  └── benches/: Criterion benchmark suites           │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│              Criterion Benchmarks                    │
│  ├── step_throughput: scalar vs batch               │
│  ├── reset_performance: full vs fast reset          │
│  ├── batch_scaling: N=1 to N=512                    │
│  └── memory_overhead: tensor/snapshot costs         │
└─────────────────────────────────────────────────────┘
```

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Batch Step | ≥ 1.2× scalar | `step_batch` faster than scalar loop at N≥256 |
| Fast Reset | ≥ 2× full reset | `restore` faster than full reset |
| Overhead | < 10% | Budgets/policies add minimal overhead |

### Key Components

1. **Benchmark Utilities** (`lib.rs`):
   - `Timer`: Manual timing with start/stop
   - `TimingResult`: Statistics (mean, p50, p99, min, max)
   - `measure()`: Closure timing helper
   - `measure_with_warmup()`: Warmup + measurement

2. **Statistics** (`stats.rs`):
   - `RunningStats`: Welford's online algorithm
   - `Histogram`: Distribution analysis with percentiles
   - `Comparison`: A/B comparison with target speedup
   - `BenchmarkResults`: Multi-metric collector

3. **Step Throughput** (`benches/step_throughput.rs`):
   - `scalar_loop`: Sequential N environment steps
   - `batch_step`: Optimized batch stepping
   - `comparison_n256`: Direct comparison at N=256

4. **Reset Performance** (`benches/reset_performance.rs`):
   - `full_reset`: Complete re-initialization
   - `fast_reset`: Snapshot restore
   - `rollout`: Reset-heavy workload patterns

5. **Batch Scaling** (`benches/batch_scaling.rs`):
   - `step_scaling`: N=1 to N=512
   - `reset_scaling`: Batch reset performance
   - `per_env_overhead`: Individual operation costs
   - `throughput_efficiency`: Fixed work scaling

6. **Memory Overhead** (`benches/memory_overhead.rs`):
   - `instance_creation`: Env creation cost
   - `snapshot_memory`: Snapshot size tracking
   - `tensor_allocation`: Small/medium/large tensors
   - `observation_stacking`: VecEnv batch assembly

### Module Structure

```
crates/wasmrl-bench/
├── Cargo.toml              # Criterion configuration
├── README.md               # Benchmark documentation
├── src/
│   ├── lib.rs              # Timer, measure utilities (5 tests)
│   └── stats.rs            # Statistics helpers (7 tests)
└── benches/
    ├── step_throughput.rs   # Step benchmarks (189 lines)
    ├── reset_performance.rs # Reset benchmarks (207 lines)
    ├── batch_scaling.rs     # Scaling benchmarks (236 lines)
    └── memory_overhead.rs   # Memory benchmarks (225 lines)

tests/
└── m7_benchmark_test.rs    # Integration tests (28 tests)
```

### Running Benchmarks

```bash
# Build environment components first
cd envs/counter_env && cargo component build --release

# Run all benchmarks
cargo bench -p wasmrl-bench

# Run specific benchmark
cargo bench --bench step_throughput
cargo bench --bench reset_performance

# Save baseline for regression detection
cargo bench --bench step_throughput -- --save-baseline main
cargo bench --bench step_throughput -- --baseline main
```

### Expected Results

```
step_throughput/comparison_n256
    scalar_loop         time:   [1.234 ms 1.256 ms 1.278 ms]
    batch               time:   [0.987 ms 1.012 ms 1.034 ms]
    speedup: 1.24x ✅

reset_performance/comparison
    full_reset          time:   [45.123 µs 46.234 µs 47.456 µs]
    fast_reset_restore  time:   [12.345 µs 12.567 µs 12.789 µs]
    speedup: 3.7x ✅
```

### Test Coverage

- 12 unit tests in wasmrl-bench modules
- 28 M7 integration tests
- **Total M7: 40 tests**
- **Total project: 400 tests**

## Next: Milestone M8

Documentation + CI Polish
- Complete API documentation
- CI benchmark regression checks
- Release preparation

---

For implementation details, see individual crate READMEs:
- [wasmrl-wit](../crates/wasmrl-wit/README.md)
- [wasmrl-runtime](../crates/wasmrl-runtime/README.md)
- [wasmrl-sdk-rust](../crates/wasmrl-sdk-rust/README.md)
- [wasmrl-policy](../crates/wasmrl-policy/README.md)
- [wasmrl-py](../crates/wasmrl-py/README.md)
- [wasmrl-bench](../crates/wasmrl-bench/README.md)
- [wasmrl-sdk-rust](../crates/wasmrl-sdk-rust/README.md)
- [wasmrl-policy](../crates/wasmrl-policy/README.md)
- [wasmrl-py](../crates/wasmrl-py/README.md)
