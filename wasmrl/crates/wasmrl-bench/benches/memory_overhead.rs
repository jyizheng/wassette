// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Memory Overhead Benchmark
//!
//! Measures memory usage characteristics of the WasmRL runtime.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use wasmrl_runtime::{EnvConfig, EnvFactory, WasmEnvInstance};
use wasmrl_wit::{DType, Tensor};

const COUNTER_ENV_PATH: &str = "../../envs/counter_env/target/wasm32-wasip2/release/counter_env.wasm";

/// Create a simple action tensor.
fn make_action(value: i32) -> Tensor {
    Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
}

/// Benchmark instance creation memory overhead.
fn bench_instance_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/instance_creation");
    group.measurement_time(Duration::from_secs(10));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        eprintln!("Counter env not found, skipping memory benchmark");
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    for num_envs in [1, 4, 16, 64] {
        group.throughput(Throughput::Elements(num_envs as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_envs),
            &num_envs,
            |b, &n| {
                b.iter(|| {
                    let instances: Vec<Box<dyn WasmEnvInstance>> = (0..n)
                        .map(|_| factory.create(&component_bytes, &config).unwrap())
                        .collect();
                    black_box(instances);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark snapshot memory overhead.
fn bench_snapshot_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/snapshot");
    group.measurement_time(Duration::from_secs(10));

    let component_path = std::path::Path::new(COUNTER_ENV_PATH);
    if !component_path.exists() {
        return;
    }

    let component_bytes = std::fs::read(COUNTER_ENV_PATH).expect("Failed to read component");
    let config = EnvConfig::default();
    let factory = EnvFactory::new().expect("Failed to create factory");

    // Measure snapshot sizes at different states
    let mut instance = factory.create(&component_bytes, &config).unwrap();
    instance.init().unwrap();

    let action = make_action(1);

    // Initial state snapshot
    let initial_snapshot = instance.snapshot().unwrap();
    println!(
        "Initial snapshot size: {} bytes",
        initial_snapshot.len()
    );

    // After some steps
    for _ in 0..10 {
        instance.step(&action).unwrap();
    }
    let mid_snapshot = instance.snapshot().unwrap();
    println!("After 10 steps snapshot size: {} bytes", mid_snapshot.len());

    // After many steps
    for _ in 0..90 {
        instance.step(&action).unwrap();
    }
    let late_snapshot = instance.snapshot().unwrap();
    println!(
        "After 100 steps snapshot size: {} bytes",
        late_snapshot.len()
    );

    group.bench_function("take_snapshot", |b| {
        b.iter(|| {
            let snapshot = instance.snapshot().unwrap();
            black_box(snapshot);
        });
    });

    group.finish();
}

/// Benchmark tensor allocation overhead.
fn bench_tensor_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/tensor");
    group.measurement_time(Duration::from_secs(5));

    // Small tensor (single action)
    group.bench_function("small_tensor_1", |b| {
        b.iter(|| {
            let tensor = Tensor::new(DType::Int32, vec![1], 42i32.to_le_bytes().to_vec());
            black_box(tensor);
        });
    });

    // Medium tensor (typical observation)
    group.bench_function("medium_tensor_64", |b| {
        b.iter(|| {
            let data: Vec<u8> = (0..64).flat_map(|i| (i as f32).to_le_bytes()).collect();
            let tensor = Tensor::new(DType::Float32, vec![64], data);
            black_box(tensor);
        });
    });

    // Large tensor (image-like observation)
    group.bench_function("large_tensor_4096", |b| {
        b.iter(|| {
            let data: Vec<u8> = (0..4096).flat_map(|i| (i as f32).to_le_bytes()).collect();
            let tensor = Tensor::new(DType::Float32, vec![64, 64], data);
            black_box(tensor);
        });
    });

    group.finish();
}

/// Benchmark batch tensor operations.
fn bench_batch_tensor_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/batch_tensor");
    group.measurement_time(Duration::from_secs(10));

    for batch_size in [1, 8, 32, 128, 512] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &n| {
                b.iter(|| {
                    // Simulate batch action creation
                    let actions: Vec<Tensor> = (0..n)
                        .map(|i| {
                            Tensor::new(
                                DType::Int32,
                                vec![1],
                                (i as i32).to_le_bytes().to_vec(),
                            )
                        })
                        .collect();
                    black_box(actions);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark observation stacking (simulating VecEnv).
fn bench_observation_stacking(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/obs_stacking");
    group.measurement_time(Duration::from_secs(10));

    const OBS_DIM: usize = 64;

    for num_envs in [1, 8, 32, 128, 512] {
        group.throughput(Throughput::Elements((num_envs * OBS_DIM) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_envs),
            &num_envs,
            |b, &n| {
                // Pre-create observations
                let observations: Vec<Tensor> = (0..n)
                    .map(|_| {
                        let data: Vec<u8> = (0..OBS_DIM)
                            .flat_map(|i| (i as f32).to_le_bytes())
                            .collect();
                        Tensor::new(DType::Float32, vec![OBS_DIM as u32], data)
                    })
                    .collect();

                b.iter(|| {
                    // Stack observations into single batch
                    let total_size = n * OBS_DIM * 4; // f32 = 4 bytes
                    let mut stacked_data = Vec::with_capacity(total_size);
                    for obs in &observations {
                        stacked_data.extend_from_slice(&obs.data);
                    }
                    let stacked = Tensor::new(
                        DType::Float32,
                        vec![n as u32, OBS_DIM as u32],
                        stacked_data,
                    );
                    black_box(stacked);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_instance_creation,
    bench_snapshot_memory,
    bench_tensor_allocation,
    bench_batch_tensor_ops,
    bench_observation_stacking,
);
criterion_main!(benches);
