# WasmRL Benchmark Suite

Comprehensive benchmark suite for measuring WasmRL runtime performance.

## Overview

This crate provides Criterion-based benchmarks to validate WasmRL performance targets:

| Metric | Target | Description |
|--------|--------|-------------|
| Batch Step | ≥ 1.2× scalar | `step_batch` should be 1.2× faster than scalar loop at N≥256 |
| Fast Reset | ≥ 2× full reset | `restore` should be 2× faster than full reset |
| Overhead | < 10% | Budgets/policies add less than 10% overhead |

## Benchmarks

### Step Throughput (`step_throughput`)

Measures step operation performance:
- **Scalar loop**: Sequential stepping of N environments
- **Batch step**: Parallel/optimized batch stepping
- **Comparison**: Direct comparison at N=256

```bash
cargo bench --bench step_throughput
```

### Reset Performance (`reset_performance`)

Measures reset operation performance:
- **Full reset**: Complete re-initialization
- **Fast reset**: Snapshot restore
- **Rollout comparison**: Reset-heavy workload patterns

```bash
cargo bench --bench reset_performance
```

### Batch Scaling (`batch_scaling`)

Measures how performance scales with environment count:
- Step scaling from N=1 to N=512
- Reset scaling
- Per-environment overhead
- Throughput efficiency

```bash
cargo bench --bench batch_scaling
```

### Memory Overhead (`memory_overhead`)

Measures memory characteristics:
- Instance creation overhead
- Snapshot memory usage
- Tensor allocation
- Observation stacking

```bash
cargo bench --bench memory_overhead
```

## Running Benchmarks

### Prerequisites

Build the environment components first:

```bash
# Build counter_env
cd envs/counter_env
cargo component build --release

# Build reset_heavy_env (optional, for reset benchmarks)
cd envs/reset_heavy_env
cargo component build --release
```

### Run All Benchmarks

```bash
cargo bench -p wasmrl-bench
```

### Run Specific Benchmark

```bash
cargo bench --bench step_throughput
cargo bench --bench reset_performance
cargo bench --bench batch_scaling
cargo bench --bench memory_overhead
```

### Run with HTML Reports

```bash
cargo bench --bench step_throughput -- --verbose
```

Reports are generated in `target/criterion/`.

## Utilities

The crate also provides benchmark utilities:

```rust
use wasmrl_bench::{measure, measure_with_warmup, Timer, TimingResult};
use wasmrl_bench::stats::{RunningStats, Histogram, Comparison};

// Measure a closure
let result = measure(100, || {
    // operation to measure
});
println!("Mean: {:?}", result.mean);

// Compare two implementations
let cmp = Comparison::new(
    "scalar", 100.0,  // baseline
    "batch", 50.0,    // candidate
    1.2,              // target speedup
);
if cmp.meets_target {
    println!("✅ Target met: {:.2}x speedup", cmp.speedup);
}
```

## Interpreting Results

### Step Throughput

Look at the `step_throughput/comparison_n256` group:
- Compare `scalar_loop` vs `batch` times
- Calculate: `speedup = scalar_time / batch_time`
- Target: `speedup >= 1.2`

### Reset Performance

Look at the `reset_performance/comparison` group:
- Compare `full_reset` vs `fast_reset_restore` times
- Calculate: `speedup = full_reset_time / fast_reset_time`
- Target: `speedup >= 2.0`

### Scaling Efficiency

Look at `batch_scaling/efficiency`:
- Throughput should increase with environment count
- Per-environment overhead should decrease

## Example Output

```
step_throughput/comparison_n256
                        time:   [1.234 ms 1.256 ms 1.278 ms]
                        thrpt:  [200.31 Kelem/s 203.82 Kelem/s 207.46 Kelem/s]

reset_performance/comparison
    full_reset          time:   [45.123 µs 46.234 µs 47.456 µs]
    fast_reset_restore  time:   [12.345 µs 12.567 µs 12.789 µs]
    
    ✅ Fast reset is 3.7x faster (target: 2.0x)
```

## Integration with CI

Add to your CI workflow:

```yaml
- name: Run benchmarks
  run: |
    cargo bench --bench step_throughput -- --noplot
    cargo bench --bench reset_performance -- --noplot
```

For regression detection, use criterion's baseline comparison:

```bash
# Save baseline
cargo bench --bench step_throughput -- --save-baseline main

# Compare against baseline
cargo bench --bench step_throughput -- --baseline main
```
