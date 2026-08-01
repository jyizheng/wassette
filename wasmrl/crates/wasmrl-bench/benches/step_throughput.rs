// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Scalar and batched step-throughput benchmarks using the real WasmRL runtime.

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use support::{component_path, counter_config, create_factory, RuntimeSet};
use wasmrl_wit::{DType, Tensor};

fn make_action(value: i32) -> Tensor {
    Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
}

fn bench_scalar_step_loop(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        eprintln!("build counter-env for wasm32-wasip2 before running benchmarks");
        return;
    };
    let mut group = c.benchmark_group("step_throughput/scalar");
    group.measurement_time(Duration::from_secs(10));

    for count in [1, 4, 16, 64, 256] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let factory = create_factory(&path, count);
            let mut environments = RuntimeSet::new(factory, count, &counter_config());
            let action = make_action(2);
            b.iter(|| {
                for handle in &environments.handles {
                    black_box(
                        environments
                            .runtime
                            .step(*handle, black_box(&action))
                            .unwrap(),
                    );
                }
            });
        });
    }
    group.finish();
}

fn bench_batch_step(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("step_throughput/batch");
    group.measurement_time(Duration::from_secs(10));

    for count in [1, 4, 16, 64, 256] {
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

fn bench_step_comparison(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    const COUNT: usize = 256;
    let mut group = c.benchmark_group("step_throughput/comparison_n256");
    group.measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Elements(COUNT as u64));

    let factory = create_factory(&path, COUNT * 2);
    let mut scalar = RuntimeSet::new(factory.clone(), COUNT, &counter_config());
    let action = make_action(2);
    group.bench_function("scalar_loop", |b| {
        b.iter(|| {
            for handle in &scalar.handles {
                black_box(scalar.runtime.step(*handle, black_box(&action)).unwrap());
            }
        });
    });

    let mut batch = RuntimeSet::new(factory, COUNT, &counter_config());
    let actions = vec![make_action(2); COUNT];
    group.bench_function("batch", |b| {
        b.iter(|| {
            black_box(
                batch
                    .runtime
                    .step_many(&batch.handles, black_box(&actions))
                    .unwrap(),
            );
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
