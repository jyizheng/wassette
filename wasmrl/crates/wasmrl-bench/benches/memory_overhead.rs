// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Runtime instance, snapshot, and tensor allocation benchmarks.

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use support::{component_path, counter_config, create_factory, RuntimeSet};
use wasmrl_wit::{DType, Tensor};

fn make_action(value: i32) -> Tensor {
    Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
}

fn bench_instance_creation(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("memory_overhead/instance_creation");
    group.measurement_time(Duration::from_secs(10));
    for count in [1, 4, 16, 64] {
        group.throughput(Throughput::Elements(count as u64));
        let factory = create_factory(&path, count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                black_box(RuntimeSet::new(factory.clone(), count, &counter_config()));
            });
        });
    }
    group.finish();
}

fn bench_snapshot_memory(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("memory_overhead/snapshot");
    group.measurement_time(Duration::from_secs(10));
    let factory = create_factory(&path, 1);
    let mut environment = RuntimeSet::new(factory, 1, &counter_config());
    let action = make_action(1);

    let initial = environment
        .runtime
        .snapshot(environment.handles[0])
        .unwrap();
    eprintln!("initial snapshot size: {} bytes", initial.data.len());
    for _ in 0..10 {
        environment
            .runtime
            .step(environment.handles[0], &action)
            .unwrap();
    }
    let stepped = environment
        .runtime
        .snapshot(environment.handles[0])
        .unwrap();
    eprintln!("snapshot size after 10 steps: {} bytes", stepped.data.len());

    group.bench_function("take_snapshot", |b| {
        b.iter(|| {
            black_box(
                environment
                    .runtime
                    .snapshot(environment.handles[0])
                    .unwrap(),
            );
        });
    });
    group.finish();
}

fn bench_tensor_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/tensor");
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("small_tensor_1", |b| {
        b.iter(|| {
            black_box(Tensor::new(
                DType::Int32,
                vec![1],
                42_i32.to_le_bytes().to_vec(),
            ));
        });
    });
    group.bench_function("medium_tensor_64", |b| {
        b.iter(|| {
            let data = (0..64).flat_map(|i| (i as f32).to_le_bytes()).collect();
            black_box(Tensor::new(DType::Float32, vec![64], data));
        });
    });
    group.bench_function("large_tensor_4096", |b| {
        b.iter(|| {
            let data = (0..4096).flat_map(|i| (i as f32).to_le_bytes()).collect();
            black_box(Tensor::new(DType::Float32, vec![64, 64], data));
        });
    });
    group.finish();
}

fn bench_batch_tensor_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead/batch_tensor");
    group.measurement_time(Duration::from_secs(10));
    for count in [1, 8, 32, 128, 512] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                black_box((0..count).map(make_action).collect::<Vec<_>>());
            });
        });
    }
    group.finish();
}

fn bench_observation_stacking(c: &mut Criterion) {
    const OBS_DIM: usize = 64;
    let mut group = c.benchmark_group("memory_overhead/obs_stacking");
    group.measurement_time(Duration::from_secs(10));
    for count in [1, 8, 32, 128, 512] {
        group.throughput(Throughput::Elements((count * OBS_DIM) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let observations = vec![Tensor::zeros(DType::Float32, vec![OBS_DIM as u32]); count];
            b.iter(|| {
                let mut data = Vec::with_capacity(count * OBS_DIM * 4);
                for observation in &observations {
                    data.extend_from_slice(&observation.data);
                }
                black_box(Tensor::new(
                    DType::Float32,
                    vec![count as u32, OBS_DIM as u32],
                    data,
                ));
            });
        });
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
