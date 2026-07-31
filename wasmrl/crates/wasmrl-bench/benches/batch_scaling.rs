// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Batch Scaling Benchmark
//!
//! Measures how throughput scales with the number of parallel environments.
//! Tests the efficiency of batching at various scales.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use wasmrl_runtime::{EnvConfig, EnvFactory, WasmEnvInstance};
use wasmrl_wit::{DType, Tensor};

const COUNTER_ENV_PATH: &str =
    "../../envs/counter_env/target/wasm32-wasip2/release/counter_env.wasm";

/// Create a simple action tensor.
fn make_action(value: i32) -> Tensor {
    Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
}

/// Benchmark scaling of step operations.
fn bench_step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scaling/step");
    group.measurement_time(Duration::from_secs(10));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        eprintln!("Counter env not found, skipping scaling benchmark");
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    // Test various batch sizes
    for num_envs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        group.throughput(Throughput::Elements(num_envs as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_envs), &num_envs, |b, &n| {
            let mut instances: Vec<Box<dyn WasmEnvInstance>> = (0..n)
                .map(|_| factory.create(&component_bytes, &config).unwrap())
                .collect();

            for instance in &mut instances {
                instance.init().unwrap();
            }

            let actions: Vec<Tensor> = (0..n).map(|_| make_action(1)).collect();

            b.iter(|| {
                let results: Vec<_> = instances
                    .iter_mut()
                    .zip(actions.iter())
                    .map(|(inst, action)| inst.step(black_box(action)).unwrap())
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark scaling of reset operations.
fn bench_reset_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scaling/reset");
    group.measurement_time(Duration::from_secs(10));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    for num_envs in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        group.throughput(Throughput::Elements(num_envs as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_envs), &num_envs, |b, &n| {
            let mut instances: Vec<Box<dyn WasmEnvInstance>> = (0..n)
                .map(|_| factory.create(&component_bytes, &config).unwrap())
                .collect();

            // Get snapshots
            for instance in &mut instances {
                instance.init().unwrap();
            }
            let snapshots: Vec<_> = instances
                .iter()
                .map(|inst| inst.snapshot().unwrap())
                .collect();

            b.iter(|| {
                // Batch restore
                for (inst, snapshot) in instances.iter_mut().zip(snapshots.iter()) {
                    inst.restore(black_box(snapshot)).unwrap();
                }
                black_box(&instances);
            });
        });
    }

    group.finish();
}

/// Benchmark per-environment overhead.
fn bench_per_env_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scaling/overhead");
    group.measurement_time(Duration::from_secs(10));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    // Measure single env step time as baseline
    group.bench_function("single_step_baseline", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();
        let action = make_action(1);

        b.iter(|| {
            let result = instance.step(black_box(&action)).unwrap();
            black_box(result);
        });
    });

    // Measure init overhead
    group.bench_function("init_overhead", |b| {
        b.iter(|| {
            let mut instance = factory.create(&component_bytes, &config).unwrap();
            let result = instance.init().unwrap();
            black_box(result);
        });
    });

    // Measure snapshot overhead
    group.bench_function("snapshot_overhead", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();

        b.iter(|| {
            let snapshot = instance.snapshot().unwrap();
            black_box(snapshot);
        });
    });

    // Measure restore overhead
    group.bench_function("restore_overhead", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();
        let snapshot = instance.snapshot().unwrap();

        b.iter(|| {
            instance.restore(black_box(&snapshot)).unwrap();
            black_box(&instance);
        });
    });

    group.finish();
}

/// Benchmark throughput efficiency at different scales.
fn bench_throughput_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scaling/efficiency");
    group.measurement_time(Duration::from_secs(15));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    // Fixed number of total steps
    const TOTAL_STEPS: usize = 1024;

    for num_envs in [1, 4, 16, 64, 256, 1024] {
        let steps_per_env = TOTAL_STEPS / num_envs;
        if steps_per_env == 0 {
            continue;
        }

        group.throughput(Throughput::Elements(TOTAL_STEPS as u64));
        group.bench_with_input(
            BenchmarkId::new("envs", num_envs),
            &(num_envs, steps_per_env),
            |b, &(n, steps)| {
                let mut instances: Vec<Box<dyn WasmEnvInstance>> = (0..n)
                    .map(|_| factory.create(&component_bytes, &config).unwrap())
                    .collect();

                for instance in &mut instances {
                    instance.init().unwrap();
                }

                let action = make_action(1);

                b.iter(|| {
                    for _ in 0..steps {
                        for instance in &mut instances {
                            let result = instance.step(black_box(&action)).unwrap();
                            black_box(result);
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_step_scaling,
    bench_reset_scaling,
    bench_per_env_overhead,
    bench_throughput_efficiency,
);
criterion_main!(benches);
