# CAMEL-AI Comparison Protocol

This document describes the fairness protocol for comparing WasmRL against CAMEL-AI baselines (SETA-ENV and CRAB).

## 1. Overview

WasmRL provides a WebAssembly-based execution layer for RL environments with:
- **In-process execution**: No subprocess/container overhead
- **Snapshot/restore**: Fast resets via memory snapshots
- **Security sandbox**: Wasmtime-based isolation

We compare against two CAMEL-AI approaches:
1. **SETA-ENV**: Structured Environment for Task Automation with Docker-based execution
2. **CRAB**: Cross-platform Agent Benchmark with VM/container backends

## 2. Fairness Principles

### 2.1 Semantic Equivalence
- All compared implementations must produce **identical outputs** for identical inputs
- Verifiers must be functionally equivalent (same pass/fail criteria)
- Random seeds must be fixed and reproducible

### 2.2 Hardware Normalization
- All benchmarks run on the same hardware configuration
- CPU pinning for consistent performance
- Memory limits documented and equivalent

### 2.3 Measurement Methodology
- Warmup period before measurement (discard first N iterations)
- Sufficient sample size for statistical significance (≥1000 samples)
- Report mean, std, p50, p99, p99.9

### 2.4 Cold-Start vs Steady-State
- Report cold-start latency separately
- Steady-state measurements after cache warm-up
- Document any pre-loading or initialization

## 3. Benchmark Tasks

### 3.1 Selected SETA-ENV Tasks

We select 5 representative tasks from SETA-ENV covering different complexity levels:

| Task ID | Name | Description | Complexity |
|---------|------|-------------|------------|
| S1 | `file-counter` | Count lines in files | Simple I/O |
| S2 | `json-transform` | Transform JSON data | Data processing |
| S3 | `code-lint` | Lint Python code | Static analysis |
| S4 | `unit-test` | Run unit tests | Test execution |
| S5 | `api-mock` | Mock HTTP API calls | Network simulation |

### 3.2 Task Implementation Requirements

For each task:
1. **Docker baseline**: Original SETA-ENV Docker implementation
2. **WasmRL component**: Functionally equivalent Wasm component
3. **Verifier**: Same verification logic for both implementations

### 3.3 CRAB Backend Tasks

We select comparable tasks from CRAB that exercise similar capabilities:

| Task ID | Name | Backend | WasmRL Equivalent |
|---------|------|---------|-------------------|
| C1 | `file-ops` | Docker | `filesystem-env` |
| C2 | `compute` | VM | `counter-env` |
| C3 | `web-fetch` | Container | `fetch-env` |

## 4. Measurement Protocol

### 4.1 Metrics Collected

```
throughput:
  - steps_per_second
  - episodes_per_second
  - batch_throughput (for vectorized envs)

latency:
  - step_latency_p50
  - step_latency_p99
  - step_latency_p999
  - reset_latency_p50
  - reset_latency_p99

cold_start:
  - first_step_latency
  - instance_creation_time
  - component_load_time

resource:
  - memory_per_instance_mb
  - cpu_utilization_percent
  - max_instances_at_slo
```

### 4.2 Test Configuration

```toml
[benchmark]
warmup_iterations = 100
measurement_iterations = 10000
num_environments = [1, 4, 16, 64, 256]
batch_sizes = [1, 8, 32]

[hardware]
cpu_cores = 8
memory_gb = 32
cpu_pinning = true

[slo]
step_latency_p99_ms = 10
reset_latency_p99_ms = 100
```

### 4.3 Execution Modes

| Mode | Description | Expected Overhead |
|------|-------------|-------------------|
| `wasm_inproc` | WasmRL in-process | Baseline (lowest) |
| `mcp_tool` | WasmRL via MCP | +RPC overhead |
| `docker_task` | SETA-ENV Docker | +container overhead |
| `crab_docker` | CRAB Docker backend | +container overhead |
| `crab_vm` | CRAB VM backend | +VM overhead |
| `subprocess` | Native subprocess | +IPC overhead |

## 5. Expected Results

### 5.1 Throughput Comparison

Based on overhead analysis:

| Mode | Relative Throughput | Notes |
|------|---------------------|-------|
| `wasm_inproc` | 1.0x (baseline) | Direct function calls |
| `mcp_tool` | 0.1-0.3x | JSON-RPC overhead |
| `docker_task` | 0.01-0.05x | Container startup |
| `subprocess` | 0.1-0.5x | Process creation |

### 5.2 Scaling Characteristics

- **WasmRL**: Near-linear scaling up to CPU core count
- **Docker**: Limited by container orchestration overhead
- **Subprocess**: Limited by process table and IPC

### 5.3 Cold-Start Analysis

| Mode | Cold Start (p99) | Warm Step (p99) |
|------|------------------|-----------------|
| `wasm_inproc` | <10ms | <100µs |
| `docker_task` | 500ms-2s | 1-10ms |
| `subprocess` | 10-50ms | 100µs-1ms |

## 6. Reproduction Instructions

### 6.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/microsoft/wassette
cd wassette/wasmrl

# Install dependencies
cargo build --release

# Pull Docker images for baselines
docker pull camel-ai/seta-env:latest
docker pull crab-benchmark/runner:latest
```

### 6.2 Running Benchmarks

```bash
# WasmRL in-process benchmark
wasmrl-bench --mode wasm_inproc --env counter --N 256 --steps 1000000

# WasmRL MCP benchmark
wasmrl-bench --mode mcp_tool --env counter --N 16 --steps 10000

# Docker baseline (SETA-ENV)
wasmrl-bench --mode docker_task --task file-counter --N 16 --steps 1000

# CRAB baseline
wasmrl-bench --mode crab_docker --task file-ops --N 16 --steps 1000
```

### 6.3 Generating Reports

```bash
# Generate comparison report
wasmrl-bench report --output results/comparison.md

# Generate paper figures
wasmrl-bench plot --input results/ --output figures/
```

## 7. Provenance and Reproducibility

### 7.1 Component Digests

All Wasm components are content-addressed:

```
counter_env.wasm:     sha256:abc123...
filesystem_env.wasm:  sha256:def456...
fetch_env.wasm:       sha256:789ghi...
```

### 7.2 Software Versions

```
wasmrl:      0.1.0
wasmtime:    38.0.4
rust:        1.75.0
docker:      24.0.0
python:      3.11.0
```

### 7.3 Hardware Configuration

```
CPU:         AMD EPYC 7763 (8 cores allocated)
Memory:      32 GB
OS:          Ubuntu 22.04 LTS
Kernel:      5.15.0
```

## 8. Limitations and Caveats

### 8.1 Scope
- Comparison focuses on RL training throughput, not general compute
- Docker baseline may not be optimally configured
- Some SETA-ENV features (full shell) not replicated in Wasm

### 8.2 Fairness Considerations
- WasmRL benefits from in-process execution (inherent advantage)
- Docker provides stronger isolation guarantees
- Full shell emulation would require heavier Wasm runtime

### 8.3 Future Work
- Add WASI-threads for multi-threaded Wasm components
- Integrate with more CAMEL-AI tasks
- Measure power/energy consumption

## 9. References

1. CAMEL-AI: https://github.com/camel-ai/camel
2. SETA-ENV: https://github.com/camel-ai/seta-env
3. CRAB: https://github.com/camel-ai/crab
4. WasmRL: This project
5. Wasmtime: https://wasmtime.dev/
