// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Scaling benchmarks for real batched step, reset, snapshot, and restore calls.

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use support::{component_path, counter_config, create_factory, RuntimeSet};
use wasmrl_wit::{DType, Tensor};

fn make_action(value: i32) -> Tensor {
    Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
}

fn bench_step_scaling(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("batch_scaling/step");
    group.measurement_time(Duration::from_secs(10));
    for count in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let factory = create_factory(&path, count);
            let mut environments = RuntimeSet::new(factory, count, &counter_config());
            let actions = vec![make_action(2); count];
            b.iter(|| {
                black_box(
                    environments
                        .runtime
                        .step_many(&environments.handles, black_box(&actions))
                        .unwrap(),
                );
            });
        });
    }
    group.finish();
}

fn bench_reset_scaling(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("batch_scaling/reset");
    group.measurement_time(Duration::from_secs(10));
    for count in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let factory = create_factory(&path, count);
            let mut environments = RuntimeSet::new(factory, count, &counter_config());
            let seeds: Vec<u64> = (0..count as u64).collect();
            b.iter(|| {
                black_box(
                    environments
                        .runtime
                        .reset_many(&environments.handles, black_box(&seeds))
                        .unwrap(),
                );
            });
        });
    }
    group.finish();
}

fn bench_per_env_overhead(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("batch_scaling/overhead");
    group.measurement_time(Duration::from_secs(10));

    let factory = create_factory(&path, 2);
    group.bench_function("single_step_baseline", |b| {
        let mut environment = RuntimeSet::new(factory.clone(), 1, &counter_config());
        let action = make_action(2);
        b.iter(|| {
            black_box(
                environment
                    .runtime
                    .step(environment.handles[0], black_box(&action))
                    .unwrap(),
            );
        });
    });

    group.bench_function("instantiate_init_reset", |b| {
        b.iter(|| {
            black_box(RuntimeSet::new(factory.clone(), 1, &counter_config()));
        });
    });

    group.bench_function("snapshot_overhead", |b| {
        let mut environment = RuntimeSet::new(factory.clone(), 1, &counter_config());
        b.iter(|| {
            black_box(
                environment
                    .runtime
                    .snapshot(environment.handles[0])
                    .unwrap(),
            );
        });
    });

    group.bench_function("restore_overhead", |b| {
        let mut environment = RuntimeSet::new(factory.clone(), 1, &counter_config());
        let snapshot = environment
            .runtime
            .snapshot(environment.handles[0])
            .unwrap();
        b.iter(|| {
            environment
                .runtime
                .restore(environment.handles[0], black_box(&snapshot))
                .unwrap();
        });
    });
    group.finish();
}

fn bench_throughput_efficiency(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    const TOTAL_STEPS: usize = 1024;
    let mut group = c.benchmark_group("batch_scaling/efficiency");
    group.measurement_time(Duration::from_secs(15));
    for count in [1, 4, 16, 64, 256, 1024] {
        let iterations = TOTAL_STEPS / count;
        group.throughput(Throughput::Elements(TOTAL_STEPS as u64));
        group.bench_with_input(BenchmarkId::new("envs", count), &count, |b, &count| {
            let factory = create_factory(&path, count);
            let mut environments = RuntimeSet::new(factory, count, &counter_config());
            let actions = vec![make_action(2); count];
            b.iter(|| {
                for _ in 0..iterations {
                    black_box(
                        environments
                            .runtime
                            .step_many(&environments.handles, &actions)
                            .unwrap(),
                    );
                }
            });
        });
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
