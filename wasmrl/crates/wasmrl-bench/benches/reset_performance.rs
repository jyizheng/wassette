// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Reset Performance Benchmark
//!
//! Measures reset performance comparing full reset vs fast reset (snapshot restore).
//! Target: restore >= 2x improvement for reset-heavy workloads

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use wasmrl_runtime::{EnvConfig, EnvFactory, WasmEnvInstance};

/// Path to the reset-heavy environment component.
const RESET_HEAVY_ENV_PATH: &str =
    "../../envs/reset_heavy_env/target/wasm32-wasip2/release/reset_heavy_env.wasm";

/// Path to the counter environment component.
const COUNTER_ENV_PATH: &str =
    "../../envs/counter_env/target/wasm32-wasip2/release/counter_env.wasm";

/// Benchmark full reset (re-initialization).
fn bench_full_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("reset_performance/full_reset");
    group.measurement_time(Duration::from_secs(10));

    // Try reset-heavy env first, fall back to counter
    let component_path = if std::path::Path::new(RESET_HEAVY_ENV_PATH).exists() {
        RESET_HEAVY_ENV_PATH
    } else if std::path::Path::new(COUNTER_ENV_PATH).exists() {
        COUNTER_ENV_PATH
    } else {
        eprintln!("No environment found, skipping reset benchmark");
        return;
    };

    let component_bytes = std::fs::read(component_path).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    group.bench_function("single", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();

        b.iter(|| {
            // Full reset: create new instance (expensive)
            instance = factory.create(&component_bytes, &config).unwrap();
            let result = instance.init().unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark fast reset (snapshot restore).
fn bench_fast_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("reset_performance/fast_reset");
    group.measurement_time(Duration::from_secs(10));

    let component_path = if std::path::Path::new(RESET_HEAVY_ENV_PATH).exists() {
        RESET_HEAVY_ENV_PATH
    } else if std::path::Path::new(COUNTER_ENV_PATH).exists() {
        COUNTER_ENV_PATH
    } else {
        return;
    };

    let component_bytes = std::fs::read(component_path).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    group.bench_function("single", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();

        // Take initial snapshot
        let snapshot = instance.snapshot().expect("Failed to snapshot");

        b.iter(|| {
            // Fast reset: restore from snapshot
            instance.restore(black_box(&snapshot)).unwrap();
            black_box(&instance);
        });
    });

    group.finish();
}

/// Direct comparison of full vs fast reset.
fn bench_reset_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("reset_performance/comparison");
    group.measurement_time(Duration::from_secs(15));

    let component_path = if std::path::Path::new(RESET_HEAVY_ENV_PATH).exists() {
        RESET_HEAVY_ENV_PATH
    } else if std::path::Path::new(COUNTER_ENV_PATH).exists() {
        COUNTER_ENV_PATH
    } else {
        return;
    };

    let component_bytes = std::fs::read(component_path).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    // Full reset benchmark
    group.bench_function("full_reset", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();

        b.iter(|| {
            instance = factory.create(&component_bytes, &config).unwrap();
            let result = instance.init().unwrap();
            black_box(result);
        });
    });

    // Fast reset benchmark
    group.bench_function("fast_reset_restore", |b| {
        let mut instance = factory.create(&component_bytes, &config).unwrap();
        instance.init().unwrap();
        let snapshot = instance.snapshot().expect("Failed to snapshot");

        b.iter(|| {
            instance.restore(black_box(&snapshot)).unwrap();
            black_box(&instance);
        });
    });

    group.finish();
}

/// Benchmark reset-heavy rollout pattern.
fn bench_reset_heavy_rollout(c: &mut Criterion) {
    let mut group = c.benchmark_group("reset_performance/rollout");
    group.measurement_time(Duration::from_secs(20));

    let component_path = if std::path::Path::new(COUNTER_ENV_PATH).exists() {
        COUNTER_ENV_PATH
    } else {
        return;
    };

    let component_bytes = std::fs::read(component_path).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    const STEPS_PER_EPISODE: usize = 10;
    const EPISODES: usize = 10;

    let action = wasmrl_wit::Tensor::new(
        wasmrl_wit::DType::Int32,
        vec![1],
        1i32.to_le_bytes().to_vec(),
    );

    // Rollout with full reset
    group.bench_function("full_reset_rollout", |b| {
        b.iter(|| {
            let mut instance = factory.create(&component_bytes, &config).unwrap();

            for _ in 0..EPISODES {
                instance.init().unwrap();
                for _ in 0..STEPS_PER_EPISODE {
                    let result = instance.step(&action).unwrap();
                    if result.done {
                        break;
                    }
                }
                // Full reset: recreate instance
                instance = factory.create(&component_bytes, &config).unwrap();
            }
            black_box(&instance);
        });
    });

    // Rollout with fast reset
    group.bench_function("fast_reset_rollout", |b| {
        b.iter(|| {
            let mut instance = factory.create(&component_bytes, &config).unwrap();
            instance.init().unwrap();
            let snapshot = instance.snapshot().expect("Failed to snapshot");

            for _ in 0..EPISODES {
                for _ in 0..STEPS_PER_EPISODE {
                    let result = instance.step(&action).unwrap();
                    if result.done {
                        break;
                    }
                }
                // Fast reset: restore snapshot
                instance.restore(&snapshot).unwrap();
            }
            black_box(&instance);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_full_reset,
    bench_fast_reset,
    bench_reset_comparison,
    bench_reset_heavy_rollout,
);
criterion_main!(benches);
