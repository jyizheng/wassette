// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Full reset and snapshot-restore benchmarks using the real WasmRL runtime.

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use support::{component_path, counter_config, create_factory, RuntimeSet};
use wasmrl_wit::{DType, Tensor};

fn bench_full_reset(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("reset_performance/full_reset");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("single", |b| {
        let factory = create_factory(&path, 1);
        let mut environment = RuntimeSet::new(factory, 1, &counter_config());
        let mut seed = 0_u64;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            black_box(
                environment
                    .runtime
                    .reset(environment.handles[0], seed)
                    .unwrap(),
            );
        });
    });
    group.finish();
}

fn bench_fast_reset(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("reset_performance/fast_reset");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("single", |b| {
        let factory = create_factory(&path, 1);
        let mut environment = RuntimeSet::new(factory, 1, &counter_config());
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

fn bench_reset_comparison(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    let mut group = c.benchmark_group("reset_performance/comparison");
    group.measurement_time(Duration::from_secs(15));
    let factory = create_factory(&path, 2);

    let mut full_reset = RuntimeSet::new(factory.clone(), 1, &counter_config());
    let mut seed = 0_u64;
    group.bench_function("full_reset", |b| {
        b.iter(|| {
            seed = seed.wrapping_add(1);
            black_box(
                full_reset
                    .runtime
                    .reset(full_reset.handles[0], seed)
                    .unwrap(),
            );
        });
    });

    let mut fast_reset = RuntimeSet::new(factory, 1, &counter_config());
    let snapshot = fast_reset.runtime.snapshot(fast_reset.handles[0]).unwrap();
    group.bench_function("fast_reset_restore", |b| {
        b.iter(|| {
            fast_reset
                .runtime
                .restore(fast_reset.handles[0], black_box(&snapshot))
                .unwrap();
        });
    });
    group.finish();
}

fn bench_reset_heavy_rollout(c: &mut Criterion) {
    let Some(path) = component_path("counter_env.wasm") else {
        return;
    };
    const STEPS_PER_EPISODE: usize = 10;
    const EPISODES: usize = 10;
    let action = Tensor::new(DType::Int32, vec![1], 2_i32.to_le_bytes().to_vec());
    let mut group = c.benchmark_group("reset_performance/rollout");
    group.measurement_time(Duration::from_secs(20));
    let factory = create_factory(&path, 2);

    let mut reset_rollout = RuntimeSet::new(factory.clone(), 1, &counter_config());
    group.bench_function("full_reset_rollout", |b| {
        b.iter(|| {
            for episode in 0..EPISODES {
                reset_rollout
                    .runtime
                    .reset(reset_rollout.handles[0], episode as u64)
                    .unwrap();
                for _ in 0..STEPS_PER_EPISODE {
                    black_box(
                        reset_rollout
                            .runtime
                            .step(reset_rollout.handles[0], &action)
                            .unwrap(),
                    );
                }
            }
        });
    });

    let mut restore_rollout = RuntimeSet::new(factory, 1, &counter_config());
    let snapshot = restore_rollout
        .runtime
        .snapshot(restore_rollout.handles[0])
        .unwrap();
    group.bench_function("fast_reset_rollout", |b| {
        b.iter(|| {
            for _ in 0..EPISODES {
                for _ in 0..STEPS_PER_EPISODE {
                    black_box(
                        restore_rollout
                            .runtime
                            .step(restore_rollout.handles[0], &action)
                            .unwrap(),
                    );
                }
                restore_rollout
                    .runtime
                    .restore(restore_rollout.handles[0], &snapshot)
                    .unwrap();
            }
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
