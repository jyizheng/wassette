// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Step Throughput Benchmark
//!
//! Measures the throughput of step operations in both scalar and batch modes.
//! Target: batch_step >= 1.2x scalar loop at N >= 256

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use wasmrl_runtime::{EnvConfig, EnvFactory, EnvPool, EnvPoolConfig, WasmEnvInstance};
use wasmrl_wit::{DType, Tensor};

/// Path to the counter environment component.
const COUNTER_ENV_PATH: &str = "../../envs/counter_env/target/wasm32-wasip2/release/counter_env.wasm";

/// Create a simple action tensor.
fn make_action(value: i32) -> Tensor {
    Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
}

/// Benchmark scalar step loop (baseline).
fn bench_scalar_step_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_throughput/scalar");
    group.measurement_time(Duration::from_secs(10));

    // Check if component exists
    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        eprintln!("Counter env not found at {:?}, skipping benchmark", component_path);
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    for num_envs in [1, 4, 16, 64, 256] {
        group.throughput(Throughput::Elements(num_envs as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_envs),
            &num_envs,
            |b, &n| {
                // Create instances
                let mut instances: Vec<Box<dyn WasmEnvInstance>> = (0..n)
                    .map(|_| factory.create(&component_bytes, &config).unwrap())
                    .collect();

                // Initialize all
                for instance in &mut instances {
                    instance.init().unwrap();
                }

                let action = make_action(1); // increment

                b.iter(|| {
                    // Scalar loop: step each instance sequentially
                    for instance in &mut instances {
                        let result = instance.step(black_box(&action)).unwrap();
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch step (optimized path).
fn bench_batch_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_throughput/batch");
    group.measurement_time(Duration::from_secs(10));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        eprintln!("Counter env not found, skipping batch benchmark");
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    for num_envs in [1, 4, 16, 64, 256] {
        group.throughput(Throughput::Elements(num_envs as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_envs),
            &num_envs,
            |b, &n| {
                // Create pool
                let pool_config = EnvPoolConfig {
                    num_envs: n,
                    max_memory_per_env: 64 * 1024 * 1024,
                    enable_snapshots: false,
                    auto_reset: false,
                };

                let mut instances: Vec<Box<dyn WasmEnvInstance>> = (0..n)
                    .map(|_| factory.create(&component_bytes, &config).unwrap())
                    .collect();

                // Initialize all
                for instance in &mut instances {
                    instance.init().unwrap();
                }

                let actions: Vec<Tensor> = (0..n).map(|_| make_action(1)).collect();

                b.iter(|| {
                    // Batch step: step all instances (in this impl, still sequential but structure ready for parallel)
                    let results: Vec<_> = instances
                        .iter_mut()
                        .zip(actions.iter())
                        .map(|(inst, action)| inst.step(black_box(action)).unwrap())
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Compare scalar vs batch at target size (N=256).
fn bench_step_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_throughput/comparison_n256");
    group.measurement_time(Duration::from_secs(15));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    const N: usize = 256;
    group.throughput(Throughput::Elements(N as u64));

    // Create shared instances
    let mut instances: Vec<Box<dyn WasmEnvInstance>> = (0..N)
        .map(|_| factory.create(&component_bytes, &config).unwrap())
        .collect();

    for instance in &mut instances {
        instance.init().unwrap();
    }

    let action = make_action(1);
    let actions: Vec<Tensor> = (0..N).map(|_| make_action(1)).collect();

    group.bench_function("scalar_loop", |b| {
        b.iter(|| {
            for instance in &mut instances {
                let result = instance.step(black_box(&action)).unwrap();
                black_box(result);
            }
        });
    });

    // Re-initialize for batch test
    for instance in &mut instances {
        instance.init().unwrap();
    }

    group.bench_function("batch", |b| {
        b.iter(|| {
            let results: Vec<_> = instances
                .iter_mut()
                .zip(actions.iter())
                .map(|(inst, action)| inst.step(black_box(action)).unwrap())
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_step_loop,
    bench_batch_step,
    bench_step_comparison,
);
criterion_main!(benches);
