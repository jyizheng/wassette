// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! M7 Benchmark Tests
//!
//! Unit tests for benchmark utilities and statistics.

use std::time::Duration;

use wasmrl_bench::stats::{BenchmarkResults, Comparison, Histogram, RunningStats};
use wasmrl_bench::{measure, measure_with_warmup, BenchConfig, Timer, TimingResult};

// ============================================================================
// TimingResult Tests
// ============================================================================

#[test]
fn test_timing_result_basic() {
    let durations = vec![
        Duration::from_micros(100),
        Duration::from_micros(200),
        Duration::from_micros(150),
    ];

    let result = TimingResult::from_durations(durations);
    assert_eq!(result.samples, 3);
    assert_eq!(result.min, Duration::from_micros(100));
    assert_eq!(result.max, Duration::from_micros(200));
}

#[test]
fn test_timing_result_empty() {
    let result = TimingResult::from_durations(vec![]);
    assert_eq!(result.samples, 0);
    assert_eq!(result.mean, Duration::ZERO);
}

#[test]
fn test_timing_result_single() {
    let result = TimingResult::from_durations(vec![Duration::from_micros(100)]);
    assert_eq!(result.samples, 1);
    assert_eq!(result.mean, Duration::from_micros(100));
    assert_eq!(result.p50, Duration::from_micros(100));
}

#[test]
fn test_timing_result_throughput() {
    let result = TimingResult {
        mean: Duration::from_millis(10),
        std_dev: Duration::ZERO,
        min: Duration::ZERO,
        max: Duration::ZERO,
        p50: Duration::ZERO,
        p99: Duration::ZERO,
        samples: 1,
    };

    // 10ms per iter with 100 ops = 10,000 ops/sec
    let throughput = result.throughput(100);
    assert!((throughput - 10000.0).abs() < 100.0);
}

#[test]
fn test_timing_result_zero_mean_throughput() {
    let result = TimingResult {
        mean: Duration::ZERO,
        std_dev: Duration::ZERO,
        min: Duration::ZERO,
        max: Duration::ZERO,
        p50: Duration::ZERO,
        p99: Duration::ZERO,
        samples: 0,
    };

    assert_eq!(result.throughput(100), 0.0);
}

// ============================================================================
// Timer Tests
// ============================================================================

#[test]
fn test_timer_basic() {
    let mut timer = Timer::new();

    for _ in 0..3 {
        timer.start();
        std::thread::sleep(Duration::from_micros(100));
        timer.stop();
    }

    let result = timer.result();
    assert_eq!(result.samples, 3);
    assert!(result.min >= Duration::from_micros(100));
}

#[test]
fn test_timer_reset() {
    let mut timer = Timer::new();
    timer.start();
    timer.stop();

    assert_eq!(timer.result().samples, 1);

    timer.reset();
    assert_eq!(timer.result().samples, 0);
}

// ============================================================================
// Measure Functions Tests
// ============================================================================

#[test]
fn test_measure() {
    let result = measure(5, || {
        std::thread::sleep(Duration::from_micros(50));
    });

    assert_eq!(result.samples, 5);
    assert!(result.mean >= Duration::from_micros(50));
}

#[test]
fn test_measure_with_warmup() {
    let mut call_count = 0;
    let result = measure_with_warmup(3, 5, || {
        call_count += 1;
        std::thread::sleep(Duration::from_micros(10));
    });

    assert_eq!(result.samples, 5);
    assert_eq!(call_count, 8); // 3 warmup + 5 measured
}

// ============================================================================
// BenchConfig Tests
// ============================================================================

#[test]
fn test_bench_config_default() {
    let config = BenchConfig::default();
    assert_eq!(config.warmup_iters, 10);
    assert_eq!(config.measure_iters, 100);
    assert_eq!(config.num_envs, 256);
}

// ============================================================================
// RunningStats Tests
// ============================================================================

#[test]
fn test_running_stats_basic() {
    let mut stats = RunningStats::new();
    stats.push(10.0);
    stats.push(20.0);
    stats.push(30.0);

    assert_eq!(stats.count(), 3);
    assert!((stats.mean() - 20.0).abs() < 0.001);
    assert_eq!(stats.min(), 10.0);
    assert_eq!(stats.max(), 30.0);
}

#[test]
fn test_running_stats_variance() {
    let mut stats = RunningStats::new();
    for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        stats.push(v);
    }

    assert!((stats.mean() - 5.0).abs() < 0.001);
    assert!((stats.variance() - 4.571).abs() < 0.01);
}

#[test]
fn test_running_stats_single() {
    let mut stats = RunningStats::new();
    stats.push(42.0);

    assert_eq!(stats.count(), 1);
    assert_eq!(stats.mean(), 42.0);
    assert_eq!(stats.variance(), 0.0); // variance undefined for n=1
}

#[test]
fn test_running_stats_merge() {
    let mut stats1 = RunningStats::new();
    stats1.push(10.0);
    stats1.push(20.0);

    let mut stats2 = RunningStats::new();
    stats2.push(30.0);
    stats2.push(40.0);

    stats1.merge(&stats2);
    assert_eq!(stats1.count(), 4);
    assert!((stats1.mean() - 25.0).abs() < 0.001);
    assert_eq!(stats1.min(), 10.0);
    assert_eq!(stats1.max(), 40.0);
}

#[test]
fn test_running_stats_merge_empty() {
    let mut stats1 = RunningStats::new();
    stats1.push(10.0);

    let stats2 = RunningStats::new();
    stats1.merge(&stats2);

    assert_eq!(stats1.count(), 1);
    assert_eq!(stats1.mean(), 10.0);
}

#[test]
fn test_running_stats_merge_into_empty() {
    let mut stats1 = RunningStats::new();

    let mut stats2 = RunningStats::new();
    stats2.push(10.0);
    stats2.push(20.0);

    stats1.merge(&stats2);
    assert_eq!(stats1.count(), 2);
    assert!((stats1.mean() - 15.0).abs() < 0.001);
}

// ============================================================================
// Histogram Tests
// ============================================================================

#[test]
fn test_histogram_basic() {
    let mut hist = Histogram::new(0.0, 100.0, 10);

    for _ in 0..10 {
        hist.push(5.0); // All in first bucket
    }

    assert_eq!(hist.buckets()[0], 10);
    for i in 1..10 {
        assert_eq!(hist.buckets()[i], 0);
    }
}

#[test]
fn test_histogram_percentile() {
    let mut hist = Histogram::new(0.0, 100.0, 10);

    // Even distribution
    for i in 0..100 {
        hist.push(i as f64);
    }

    let p50 = hist.percentile(50.0);
    assert!(p50 >= 45.0 && p50 <= 55.0);

    let p99 = hist.percentile(99.0);
    assert!(p99 >= 95.0);
}

#[test]
fn test_histogram_out_of_range() {
    let mut hist = Histogram::new(0.0, 100.0, 10);

    hist.push(-10.0); // Below range
    hist.push(150.0); // Above range

    // Both should be ignored
    let total: u64 = hist.buckets().iter().sum();
    assert_eq!(total, 0);
}

// ============================================================================
// Comparison Tests
// ============================================================================

#[test]
fn test_comparison_speedup() {
    let cmp = Comparison::new("scalar", 100.0, "batch", 50.0, 1.5);

    assert!((cmp.speedup - 2.0).abs() < 0.001);
    assert!(cmp.meets_target);
}

#[test]
fn test_comparison_not_meeting_target() {
    let cmp = Comparison::new("scalar", 100.0, "batch", 90.0, 2.0);

    assert!(!cmp.meets_target);
    assert!((cmp.speedup - 1.111).abs() < 0.01);
}

#[test]
fn test_comparison_exact_target() {
    let cmp = Comparison::new("baseline", 200.0, "optimized", 100.0, 2.0);

    assert_eq!(cmp.speedup, 2.0);
    assert!(cmp.meets_target);
}

// ============================================================================
// BenchmarkResults Tests
// ============================================================================

#[test]
fn test_benchmark_results_record() {
    let mut results = BenchmarkResults::new();
    results.record("latency_us", 100.0);
    results.record("latency_us", 120.0);
    results.record("throughput_ops", 1000.0);

    let latency = results.get("latency_us").unwrap();
    assert_eq!(latency.count(), 2);
    assert!((latency.mean() - 110.0).abs() < 0.001);

    let throughput = results.get("throughput_ops").unwrap();
    assert_eq!(throughput.count(), 1);
}

#[test]
fn test_benchmark_results_missing_metric() {
    let results = BenchmarkResults::new();
    assert!(results.get("nonexistent").is_none());
}

// ============================================================================
// Performance Target Tests
// ============================================================================

#[test]
fn test_batch_speedup_target() {
    // Target: batch >= 1.2x scalar at N >= 256
    let scalar_time = 100.0; // microseconds
    let batch_time = 80.0; // Must be <= 83.3 for 1.2x

    let cmp = Comparison::new("scalar", scalar_time, "batch", batch_time, 1.2);
    assert!(
        cmp.meets_target,
        "Batch should be at least 1.2x faster than scalar"
    );
}

#[test]
fn test_fast_reset_speedup_target() {
    // Target: restore >= 2x improvement for reset-heavy
    let full_reset_time = 1000.0; // microseconds
    let fast_reset_time = 400.0; // Must be <= 500 for 2x

    let cmp = Comparison::new(
        "full_reset",
        full_reset_time,
        "fast_reset",
        fast_reset_time,
        2.0,
    );
    assert!(
        cmp.meets_target,
        "Fast reset should be at least 2x faster than full reset"
    );
}

// ============================================================================
// Integration-style Tests
// ============================================================================

#[test]
fn test_end_to_end_benchmark_workflow() {
    let mut results = BenchmarkResults::new();

    // Simulate benchmark runs
    for i in 0..10 {
        let latency = 100.0 + (i as f64 * 2.0);
        results.record("step_latency_us", latency);
    }

    let stats = results.get("step_latency_us").unwrap();
    assert_eq!(stats.count(), 10);
    assert!(stats.mean() > 100.0);
    assert!(stats.mean() < 120.0);
}

#[test]
fn test_timing_percentiles() {
    let mut durations = Vec::new();
    for i in 1..=100 {
        durations.push(Duration::from_micros(i as u64));
    }

    let result = TimingResult::from_durations(durations);

    // p50 should be around 50µs
    assert!(result.p50 >= Duration::from_micros(49));
    assert!(result.p50 <= Duration::from_micros(51));

    // p99 should be around 99µs
    assert!(result.p99 >= Duration::from_micros(98));
}
