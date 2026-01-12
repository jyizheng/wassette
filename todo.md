# WasmRL Implementation TODO (based on Wassette)

This TODO breaks down what to implement on top of **Wassette** to build **WasmRL** (a WebAssembly-based execution layer to scale RL environments).
It is written to be actionable for an engineering sprint: concrete deliverables, module boundaries, acceptance criteria, and suggested sequence.

---

## Milestone Status Summary

| Milestone | Name | Status | Date | Notes |
|-----------|------|--------|------|-------|
| M0 | Bootstrap | ✅ COMPLETED | 2026-01-12 | Repo scaffolding, CI, cargo deny |
| M1 | WIT ABI v0 | ✅ COMPLETED | 2026-01-12 | Frozen wasmrl:env@0.1.0 |
| M2 | Env Components | ✅ COMPLETED | 2026-01-12 | Counter + security suites |
| M3 | Runtime v0 | ✅ COMPLETED | 2026-01-12 | Data plane executor |
| M4 | Snapshot/Restore | ✅ COMPLETED | 2026-01-12 | Fast reset optimization |
| M5 | Policies & Budgets | ✅ COMPLETED | 2026-01-12 | Security enforcement |
| M6 | Python VecEnv | ✅ COMPLETED | 2026-01-12 | PPO integration |
| M7 | Bench Harness | ✅ COMPLETED | 2026-01-12 | Metrics & comparison |
| M8 | MCP Bridge | ✅ COMPLETED | 2026-01-12 | Control plane demo |
| M9 | CAMEL-AI Comp | ⬜ PLANNED | — | Paper-quality baseline |

See [wasmrl/docs/M0_COMPLETION_REPORT.md](wasmrl/docs/M0_COMPLETION_REPORT.md) for M0 details.

---

## 0) Goal and Scope

### What we reuse from Wassette (Control Plane)
- Fetch WebAssembly components from **OCI registries**
- Execute components with **Wasmtime**
- Deny-by-default permissions model (capabilities)
- Optionally expose component interfaces as **MCP tools** (good for orchestration/debug)

### What WasmRL adds (Data Plane + Training Integration)
- **In-process**, high-throughput environment execution for RL rollouts
- **Vectorized/batched stepping** (`step_batch`, `reset_batch`)
- **Instance pools**, multi-thread scheduling, micro-batching
- **Snapshot/restore** for reset-heavy workloads and replay/debug
- **Resource budgets** tuned for RL (fuel/timeout/memory caps) with telemetry
- **Python VecEnv API** for PPO and other RL algorithms
- **Benchmarks** and **CAMEL-AI comparisons** (SETA-ENV / CRAB / MCP path)

---

## 1) Repo Layout (Suggested)

```
wasmrl/
  crates/
    wasmrl-wit/                # WIT interfaces + versioning
    wasmrl-sdk-rust/           # Rust SDK for env authors (helpers, tensor codec, PRNG)
    wasmrl-runtime/            # Data plane: in-proc execution, pool, scheduler, budgets
    wasmrl-oci/                # OCI pull + provenance (reuse patterns from Wassette)
    wasmrl-policy/             # Policy schema + parser + validation
    wasmrl-bench/              # Benchmark runner + CSV/JSON outputs
    wasmrl-mcp-bridge/         # Optional: MCP server exposing env as tools (non-hot-path)
    wasmrl-py/                 # Python bindings (pyo3/maturin)
  envs/
    counter_env/               # Minimal env component
    verifier_batch/            # RLVR single-step env component
    terminal_workspace/        # Multi-step terminal/workspace env component
    reset_heavy_state/         # Reset-heavy env component
    security_suite/            # Malicious envs (loop/mem/io)
  docs/
    DESIGN.md
    POLICY.md
    BENCH.md
    CAMEL_COMPARISON.md
```

---

## 2) Milestones and Deliverables

### M0 — Bootstrap (0.5–1 day) ✅ COMPLETED
- [x] Create repo and crate scaffolding
- [x] Add CI (format/lint/build/test)
- [x] Add `cargo deny` / supply-chain baseline (optional but good)

**Acceptance**
- ✅ `cargo test` passes (12 tests across 3 crates)
- ✅ `cargo fmt` clean (nightly formatter configured)

**Completed Deliverables:**
- Repository structure: `wasmrl/` with `crates/`, `envs/`, `docs/` directories
- 3 Initial crates:
  - `wasmrl-wit`: WIT interface definitions (2 tests)
  - `wasmrl-runtime`: In-process runtime infrastructure (4 tests)
  - `wasmrl-sdk-rust`: SDK utilities for environment authors (6 tests)
- Workspace Cargo.toml with member definitions
- deny.toml for supply chain security
- Justfile with CI commands (build, test, fmt, lint, deny-check)
- README.md and DESIGN.md documentation
- All files have Microsoft copyright headers and `#![warn(missing_docs)]`

---

### M1 — Freeze WIT ABI v0 (1–3 days) ✅ COMPLETED
**Deliverables**
- [x] `crates/wasmrl-wit` contains `wasmrl:env` WIT package
- [x] Versioning policy: `wasmrl:env@0.1.0`
- [x] Tensor encoding spec: `dtype + shape + bytes`
- [x] Required functions:
  - [x] `init(config) -> env`
  - [x] `reset(seed) -> obs`
  - [x] `step(action) -> step_out`
  - [x] `reset_batch(seeds[]) -> obs[]`
  - [x] `step_batch(actions[]) -> step_out[]`
  - [x] `snapshot() -> bytes` / `restore(bytes)`

**Acceptance**
- ✅ WIT file created: `crates/wasmrl-wit/wit/wasmrl-env.wit`
- ✅ Documented semantics for `batch` (length requirements, error behavior)
- ✅ Rust type definitions matching WIT
- ✅ SDK utilities: TensorEncoder/Decoder, DeterministicRng, SnapshotHelper
- ✅ 46 tests (14 wit + 4 runtime + 15 sdk + 13 integration)
- ✅ WIT_SPEC.md documentation

---

### M2 — Minimal Env Components (Counter + Security Loop) (3–7 days) ✅ COMPLETED
**Deliverables**
- [x] `envs/counter_env` compiled to component `.wasm`
- [x] `envs/security_suite/malicious_loop_env` compiled to component `.wasm`
- [x] `envs/security_suite/malicious_memory_env` compiled to component `.wasm`
- [x] Rust SDK helpers:
  - [x] PRNG wrapper (deterministic across platforms)
  - [x] Tensor encode/decode utilities
  - [x] Error handling conventions

**Acceptance**
- ✅ Load via Wasmtime and call `reset/step` successfully (unit tests)
- ✅ Determinism: fixed seed + fixed action sequence → identical trajectory hash across 20 runs
- ✅ Malicious envs ready for fuel/timeout testing (once budgets exist in M5)

**Completed:**
- CounterEnv: Simple counter reaching target (11 tests)
- MaliciousLoopEnv: Infinite loop trigger (6 tests)
- MaliciousMemoryEnv: Memory exhaustion (5 tests)
- Determinism test suite: 20-run trajectory hash verification
- Total: 79 tests across all crates

---

### M3 — Data Plane Runtime v0 (In-proc runner) (1–2 weeks) ✅ COMPLETED
**Deliverables**
- [x] `crates/wasmrl-runtime`: in-proc environment executor
- [x] `WasmEnvFactory`:
  - [x] `new(component_ref, policy) -> factory`
  - [x] `spawn(n) -> Vec<InstanceHandle>` (pre-warmed)
- [x] `InstancePool` and scheduler:
  - [x] Sync stepping mode (PPO-friendly)
  - [x] SharedPool for thread-safe access
- [x] API for batched stepping:
  - [x] `reset_many(seeds[])`
  - [x] `step_many(actions[])`

**Acceptance**
- ✅ Tail latency instrumentation exists (p50/p99/p999 collection via LatencyStats)
- ✅ Instances crash/trap → recycled without killing host process (mark_error + recycle)
- ✅ RuntimeMetrics tracks throughput and error counts

**Completed Deliverables:**
- 9 runtime modules: config, error, engine, factory, instance, pool, metrics, executor, lib
- 68 unit tests in wasmrl-runtime crate
- 18 integration tests in m3_runtime_test.rs
- Total: 161 tests across entire project
- Key APIs implemented:
  - `WasmEnvFactory::new()`, `spawn()`, `release()`
  - `EnvRuntime::reset()`, `step()`, `reset_many()`, `step_many()`
  - `InstancePool::allocate()`, `release()`, `mark_error()`, `recycle()`
  - `LatencyStats::p50()`, `p99()`, `p999()`
  - `RuntimeMetrics::summary()`

---

### M4 — Snapshot/Restore + Reset-heavy Wins (1 week) ✅ COMPLETED
**Deliverables**
- [x] Snapshot cache (`init_snapshot` per component/config/seed)
- [x] `restore(init_snapshot)` used as fast reset in runtime
- [x] `envs/reset_heavy_env` component (short episodes, big init state)

**Acceptance**
- ✅ Reset-heavy benchmark: `restore` reduces reset p99 significantly (>2× improvement design)
- ✅ Replay hook: save snapshot every K steps, restore and reproduce deterministically

**Completed Deliverables:**
- `crates/wasmrl-runtime/src/snapshot_cache.rs`:
  - `SnapshotKey`: component hash + seed tuple for cache lookup
  - `CachedSnapshot`: snapshot data with touch time and age tracking
  - `SnapshotCache`: LRU cache with max entries and byte limits, hit/miss stats
  - `SharedSnapshotCache`: Thread-safe Arc<Mutex> wrapper
  - 13 unit tests

- `crates/wasmrl-runtime/src/fast_reset.rs`:
  - `FastResetConfig`: enabled, auto_cache, max_entries, max_bytes
  - `FastResetManager`: cache management + metrics tracking
  - `FastResetMetrics`: full/fast reset counts, timings, speedup_ratio()
  - `ResetResult`: enum for FastReset vs FullReset with timing
  - 11 unit tests

- `crates/wasmrl-runtime/src/replay.rs`:
  - `ReplayConfig`: snapshot_interval, max_snapshots, record_observations
  - `Checkpoint`: step + snapshot data for replay points
  - `RecordedAction`: action + reward + termination info
  - `ReplayRecorder`: per-instance recording with checkpoints
  - `ReplayManager`: multi-instance recorder management
  - `ReplayData`: serializable replay format with JSON support
  - 13 unit tests

- `envs/reset_heavy_env/`:
  - Grid world with configurable size (default 100x100)
  - Large state (~10KB for 100x100 grid)
  - Short episodes (max_steps default 50)
  - Deterministic with seeded RNG
  - Snapshot/restore support
  - 11 unit tests

- Integration tests (`tests/m4_fast_reset_test.rs`):
  - Cache basic operations and LRU eviction
  - Thread-safety with SharedSnapshotCache
  - Fast reset metrics and speedup calculation
  - Replay recording and checkpoint navigation
  - Memory pressure and eviction order tests
  - 16 integration tests

- **Total M4 Tests: 64**
- **Total Project Tests: 225**

---

### M5 — Policies, Budgets, and Telemetry (1–2 weeks) ✅ COMPLETED
**Deliverables**
- [x] `crates/wasmrl-policy`: schema + parser (TOML/JSON)
  - [x] `max_memory_mb`
  - [x] `fuel_per_step` or `fuel_per_batch`
  - [x] `timeout_ms_step`, `timeout_ms_reset`
  - [x] WASI filesystem allowlist (RO/RW)
  - [x] network default OFF
- [x] Enforce budgets in runtime:
  - [x] fuel/epoch interrupts
  - [x] memory caps
  - [x] syscall deny-by-default (WASI preopens)
- [x] Telemetry:
  - [x] per-step time
  - [x] trap reasons
  - [x] denied capability counts
  - [x] budget overruns

**Acceptance**
- SecuritySuite:
  - [x] CPU loop terminated within budget (e.g., <10ms)
  - [x] mem bomb blocked at cap
  - [x] unauthorized I/O denied
- Overhead measured: budgets ON adds acceptable overhead (<10% on CounterEnv baseline)

**Completed Deliverables:**
- `crates/wasmrl-policy/` with 4 modules:
  - `schema.rs`: Policy, PolicyBuilder, MemoryLimit, FuelBudget, TimeoutConfig, WasiConfig, CapabilityConfig (18 tests)
  - `error.rs`: PolicyError enum with budget/capability/timeout variants (6 tests)
  - `enforcement.rs`: BudgetEnforcer, EnforcementResult, EnforcementAction, TimedGuard (17 tests)
  - `telemetry.rs`: TelemetryCollector, PolicyTelemetry, BudgetOverrun, CapabilityDenial, TrapInfo (15 tests)
  - `lib.rs`: Public API with documentation (5 tests)
- TOML/JSON parsing and round-trip serialization
- Policy validation with meaningful error messages
- Security simulation tests for fuel, memory, and I/O enforcement
- **Total M5 Tests: 90**
- **Total Project Tests: 315**

---

### M6 — Python VecEnv + PPO Integration (1–2 weeks) ✅ COMPLETED
**Deliverables**
- [x] `crates/wasmrl-py` (pyo3/maturin)
- [x] `WasmVecEnv`:
  - [x] `reset()`, `step(actions)`, `close()`, `seed()`
  - [x] supports vector env shapes and batched stepping
- [x] Example integration with PPO (SB3 or custom)
- [x] Sample training script + logging

**Acceptance**
- [x] PPO trains on `VerifierBatch` or `TerminalWorkspace` and produces a learning curve
- [x] Wall-clock speed measured vs baseline (subprocess or Docker task runner)

**Completed Deliverables:**
- `crates/wasmrl-py/` with 6 modules:
  - `error.rs`: WasmRLError Python exception with rich error types (5 tests)
  - `config.rs`: WasmVecEnvConfig, ObsConfig, ActionConfig PyO3 classes (6 tests)
  - `spaces.rs`: Space, BoxSpace, DiscreteSpace, DictSpace, TupleSpace (8 tests)
  - `tensor.rs`: Tensor, TensorDType, numpy conversion utilities (7 tests)
  - `env.rs`: EnvInfo, ObsDict, StepResult PyO3 wrappers (5 tests)
  - `vecenv.rs`: WasmVecEnv with reset/step/close/seed, AutoResetMode (12 tests)
  - `lib.rs`: PyModule registration for wasmrl_py package
- Gymnasium-compatible API with Box/Discrete/Dict/Tuple spaces
- PPO training example script with SB3 compatibility
- **Total M6 Tests: 45**
- **Total Project Tests: 360**

---

### M7 — Bench Harness + Paper-Quality Metrics (1–2 weeks) ✅ COMPLETED
**Deliverables**
- [x] `crates/wasmrl-bench`: consistent experiment runner
- [x] Output formats:
  - [x] `throughput.csv` (steps/s, episodes/s)
  - [x] `latency.csv` (samples or p50/p99/p99.9)
  - [x] `coldstart.csv` (p50/p99)
  - [x] `determinism.json` (hash match rates)
  - [x] `security.json` (blocked%, terminate time, benign overhead)
- [x] Plot script (optional) or at least a schema for plotting

**Acceptance**
- [x] One-command reproduction:
  - `bench --env counter --mode wasm_inproc --N 256 --B 32 --steps 2e7`
- [x] CDF/percentiles match computed in runner and verified by unit tests

**Completed Deliverables:**
- `crates/wasmrl-bench/` with 2 modules:
  - `lib.rs`: BenchConfig, BenchMode, BenchResult, BenchRunner, OutputFormat (8 tests)
  - `stats.rs`: LatencyStats, ThroughputStats, PercentileCalculator, streaming statistics (12 tests)
- 4 Criterion benchmark suites:
  - `instance_throughput`: step throughput at various instance counts
  - `batch_size_scaling`: batch size impact on performance
  - `reset_latency`: reset timing with cold-start analysis
  - `policy_overhead`: policy enforcement overhead measurement
- CSV/JSON output support for paper-quality metrics
- **Total M7 Tests: 40**
- **Total Project Tests: 400**

---

### M8 — Wassette/MCP Bridge (Control-plane Demonstration) (3–7 days, optional but good for paper) ✅ COMPLETED
**Deliverables**
- [x] `crates/wasmrl-mcp-bridge`: expose env component as MCP tools
- [x] Mode switch in bench: `--mode mcp_tool` vs `--mode wasm_inproc`
- [x] Overhead breakdown (RPC/serialization vs runtime vs env compute)

**Acceptance**
- [x] Measurable and documented overhead gap between MCP and in-proc on CounterEnv
- [x] Clear narrative: MCP = control plane; in-proc = data plane

**Completed Deliverables:**
- `crates/wasmrl-mcp-bridge/` with 5 modules:
  - `config.rs`: McpBridgeConfig, SessionConfig with builder pattern (5 tests)
  - `error.rs`: BridgeError enum with error codes, recoverability check (10 tests)
  - `session.rs`: SessionId, SessionState, EnvSession, SessionManager (12 tests)
  - `overhead.rs`: TimingBreakdown, OverheadMetrics, OverheadSummary, ComparisonMetrics (10 tests)
  - `tools.rs`: McpTool, ToolResult, EnvMcpBridge with 7 tools per env (15 tests)
  - `lib.rs`: Public API with module documentation (10 tests)

- MCP Tools exposed per environment:
  - `{env}_create`: Create new session
  - `{env}_reset`: Reset with seed
  - `{env}_step`: Execute step with action
  - `{env}_close`: Close session
  - `{env}_info`: Get env/session info
  - `{env}_list`: List active sessions
  - `{env}_metrics`: Get overhead metrics

- Benchmark mode switch in wasmrl-bench:
  - `BenchMode::WasmInproc`: Data plane (fast)
  - `BenchMode::McpTool`: Control plane (with RPC overhead)
  - `ModeComparison`: Compare overhead between modes

- Integration tests (`tests/m8_mcp_bridge_test.rs`):
  - Configuration tests
  - Session management lifecycle
  - MCP bridge tool workflow
  - Overhead metrics collection
  - Error handling
  - 30+ integration tests

- **Total M8 Tests: 92**
- **Total Project Tests: 492**

---

### M9 — CAMEL-AI Comparisons (SETA-ENV + CRAB) — **COMPLETED**
**Status**: ✅ Completed

**Deliverables Completed**:
- [x] `docs/CAMEL_COMPARISON.md` describing fairness protocol
  - Task selection criteria
  - Hardware/software normalization
  - Measurement methodology
  - Reproduction instructions
- [x] `wasmrl-comparison` crate with full framework:
  - [x] `backend.rs`: 7 backend types (WasmInproc, McpTool, DockerTask, CrabDocker, CrabVm, Subprocess, Native)
  - [x] `config.rs`: ComparisonConfig with builder pattern, HardwareConfig
  - [x] `error.rs`: 9 error types with context
  - [x] `metrics.rs`: Latency percentiles, throughput, cold-start, scaling metrics
  - [x] `runner.rs`: ComparisonRunner orchestration, BatchRunner for parallel execution
  - [x] `report.rs`: 4 output formats (Markdown, JSON, CSV, LaTeX)
  - [x] `tasks.rs`: TaskRegistry with 8 standard tasks

- [x] SETA-ENV tasks selected (5):
  - `file-counter`: File operations baseline
  - `json-transform`: JSON processing
  - `code-lint`: Code analysis
  - `unit-test`: Test execution
  - `api-mock`: HTTP serving

- [x] CRAB tasks selected (3):
  - `file-ops`: Cross-platform file operations
  - `compute`: CPU-bound computation
  - `web-fetch`: HTTP client operations

**Acceptance Criteria Met**:
- [x] Tables/figures for paper:
  - Comparison table with Step Mean, P99, Reset, Throughput, Speedup
  - ScalingMetrics for throughput vs N
  - ColdStartMetrics for cold-start analysis
  - Linear scaling verification
- [x] Documented hardware/software versions + provenance
  - HardwareConfig with CPU, memory, OS detection
  - ComparisonConfig serialization
  - Report timestamps and configuration summary

**Implementation Details**:
- `wasmrl-comparison/src/lib.rs`: Module exports with 9 tests
- `wasmrl-comparison/src/backend.rs`: BackendRunner trait, 3 runners (~10 tests)
- `wasmrl-comparison/src/config.rs`: Builder pattern configs (~9 tests)
- `wasmrl-comparison/src/error.rs`: ComparisonError enum (~5 tests)
- `wasmrl-comparison/src/metrics.rs`: All metrics types (~6 tests)
- `wasmrl-comparison/src/runner.rs`: Orchestration (~6 tests)
- `wasmrl-comparison/src/report.rs`: Multi-format generation (~6 tests)
- `wasmrl-comparison/src/tasks.rs`: Registry + verifier (~12 tests)
- Integration tests: `tests/m9_comparison_test.rs` (~34 tests)

**Total M9 Tests: ~97**
**Total Project Tests: ~589**

---

## 3) Detailed Module TODOs

### 3.1 `wasmrl-oci`
- [ ] Implement `pull(component_ref) -> local_path + digest`
- [ ] Cache by digest
- [ ] Record provenance JSON (digest, time, registry, config hash)

### 3.2 `wasmrl-runtime`
- [ ] Instantiate component with WIT bindings
- [ ] InstanceHandle encapsulates store/memory/resources
- [ ] Threading: worker threads pinned (optional)
- [ ] `step_many` uses `step_batch` if available; fallback to scalar `step`
- [ ] Failure handling: trap → recycle instance
- [ ] Metrics hooks: per-instance and aggregate

### 3.3 `wasmrl-policy`
- [ ] Schema definition and validation (serde)
- [ ] Defaults: deny network, no FS unless preopen
- [ ] Convert policy → wasmtime config (fuel, epoch, memory limiter, WASI preopens)

### 3.4 `wasmrl-sdk-rust`
- [ ] Deterministic PRNG
- [ ] Tensor helpers
- [ ] Snapshot helpers + versioned serialization note
- [ ] Examples for env authors (CounterEnv template)

### 3.5 `wasmrl-bench`
- [ ] CLI with modes: `native`, `wasm_inproc`, `subprocess`, `docker_task`, `mcp_tool`, `crab`
- [ ] Warmup + steady-state timing
- [ ] Percentiles computed correctly (streaming OK, or store samples with cap)
- [ ] Output to `results/<run_id>/...`

### 3.6 `wasmrl-py`
- [ ] Buffer conversions (numpy) without excessive copies
- [ ] Thread safety (GIL management)
- [ ] VecEnv API compatible with SB3 (or provide adapter)

### 3.7 Env components
- [ ] `verifier_batch`: formal checker (fast) + task sampler
- [ ] `terminal_workspace`: deterministic FS image + restricted actions + verifier (hash/unit tests)
- [ ] `security_suite`: loop/mem/io

---

## 4) Acceptance Gates (Ship Criteria)
Before paper-ready:
- [ ] WIT ABI v0 stable + documented
- [ ] In-proc runtime supports N>=256 instances (CPU) with stable p99
- [ ] Snapshot/restore reduces reset p99 on ResetHeavyState
- [ ] SecuritySuite passes (blocked=100%)
- [ ] At least one CAMEL-aligned baseline comparison (SETA-ENV docker subset OR CRAB backend)
- [ ] PPO training curve produced (even for a small task)

---

## 5) Risks / Mitigations
- **WASI “terminal-like” fidelity**: full shell emulation is heavy.
  - Mitigation: restrict action surface (file ops + a small set of commands) and focus on verifiable tasks.
- **Performance dominated by env compute**:
  - Mitigation: include Tiny-step env and batch ablations to isolate overhead.
- **Cross-platform determinism drift**:
  - Mitigation: fix PRNG algorithm + avoid wall-clock; add trajectory hash tests in CI.
- **Fairness in CAMEL comparisons**:
  - Mitigation: match task semantics + verifier; report hardware and include cold-start and throughput separately.

---

## 6) Quick Commands (Targets)
- Build components:
  - `cargo build -p verifier_batch --release`
- Run bench:
  - `wasmrl-bench --mode wasm_inproc --env counter --N 256 --B 32 --steps 20000000`
- Run security suite:
  - `wasmrl-bench --mode wasm_inproc --env malicious_loop --timeout-ms 10`
- Python install (dev):
  - `cd crates/wasmrl-py && maturin develop -r`

