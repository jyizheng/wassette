// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Latency metrics collection for runtime operations.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Latency statistics with percentile tracking.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Number of samples collected.
    pub count: u64,
    /// Sum of all latencies for mean calculation.
    pub sum_ns: u128,
    /// Minimum latency observed.
    pub min_ns: u64,
    /// Maximum latency observed.
    pub max_ns: u64,
    /// Recent samples for percentile calculation.
    samples: VecDeque<u64>,
    /// Maximum number of samples to retain.
    max_samples: usize,
}

impl LatencyStats {
    /// Create a new latency stats collector.
    pub fn new(max_samples: usize) -> Self {
        Self {
            count: 0,
            sum_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Record a latency sample.
    pub fn record(&mut self, duration: Duration) {
        let ns = duration.as_nanos() as u64;
        self.count += 1;
        self.sum_ns += ns as u128;
        self.min_ns = self.min_ns.min(ns);
        self.max_ns = self.max_ns.max(ns);

        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(ns);
    }

    /// Get mean latency in nanoseconds.
    pub fn mean_ns(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        (self.sum_ns / self.count as u128) as u64
    }

    /// Get mean latency as Duration.
    pub fn mean(&self) -> Duration {
        Duration::from_nanos(self.mean_ns())
    }

    /// Get minimum latency as Duration.
    pub fn min(&self) -> Duration {
        if self.min_ns == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.min_ns)
        }
    }

    /// Get maximum latency as Duration.
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.max_ns)
    }

    /// Calculate percentile from recent samples.
    pub fn percentile(&self, p: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted: Vec<u64> = self.samples.iter().copied().collect();
        sorted.sort_unstable();

        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        let idx = idx.min(sorted.len() - 1);

        Duration::from_nanos(sorted[idx])
    }

    /// Get p50 (median) latency.
    pub fn p50(&self) -> Duration {
        self.percentile(50.0)
    }

    /// Get p99 latency.
    pub fn p99(&self) -> Duration {
        self.percentile(99.0)
    }

    /// Get p999 latency.
    pub fn p999(&self) -> Duration {
        self.percentile(99.9)
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.sum_ns = 0;
        self.min_ns = u64::MAX;
        self.max_ns = 0;
        self.samples.clear();
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Metrics collector for the runtime.
#[derive(Debug)]
pub struct RuntimeMetrics {
    /// Latency statistics for step operations.
    pub step_latency: LatencyStats,

    /// Latency statistics for reset operations.
    pub reset_latency: LatencyStats,

    /// Latency statistics for batch step operations.
    pub batch_step_latency: LatencyStats,

    /// Latency statistics for instance creation.
    pub instantiation_latency: LatencyStats,

    /// Number of successful steps.
    pub steps_completed: u64,

    /// Number of successful resets.
    pub resets_completed: u64,

    /// Number of traps encountered.
    pub traps_count: u64,

    /// Number of timeouts.
    pub timeouts_count: u64,

    /// Number of fuel exhaustion events.
    pub fuel_exhausted_count: u64,

    /// Number of instances recycled due to errors.
    pub instances_recycled: u64,
}

impl RuntimeMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            step_latency: LatencyStats::new(1000),
            reset_latency: LatencyStats::new(1000),
            batch_step_latency: LatencyStats::new(1000),
            instantiation_latency: LatencyStats::new(100),
            steps_completed: 0,
            resets_completed: 0,
            traps_count: 0,
            timeouts_count: 0,
            fuel_exhausted_count: 0,
            instances_recycled: 0,
        }
    }

    /// Record a completed step with its duration.
    pub fn record_step(&mut self, duration: Duration) {
        self.step_latency.record(duration);
        self.steps_completed += 1;
    }

    /// Record a completed reset with its duration.
    pub fn record_reset(&mut self, duration: Duration) {
        self.reset_latency.record(duration);
        self.resets_completed += 1;
    }

    /// Record a completed batch step with its duration.
    pub fn record_batch_step(&mut self, duration: Duration, batch_size: usize) {
        self.batch_step_latency.record(duration);
        self.steps_completed += batch_size as u64;
    }

    /// Record an instance trap.
    pub fn record_trap(&mut self) {
        self.traps_count += 1;
    }

    /// Record a timeout.
    pub fn record_timeout(&mut self) {
        self.timeouts_count += 1;
    }

    /// Record fuel exhaustion.
    pub fn record_fuel_exhausted(&mut self) {
        self.fuel_exhausted_count += 1;
    }

    /// Record an instance being recycled.
    pub fn record_instance_recycled(&mut self) {
        self.instances_recycled += 1;
    }

    /// Get throughput (steps per second) based on recent step latency.
    pub fn throughput_steps_per_sec(&self) -> f64 {
        let mean_ns = self.step_latency.mean_ns();
        if mean_ns == 0 {
            return 0.0;
        }
        1_000_000_000.0 / mean_ns as f64
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.step_latency.reset();
        self.reset_latency.reset();
        self.batch_step_latency.reset();
        self.instantiation_latency.reset();
        self.steps_completed = 0;
        self.resets_completed = 0;
        self.traps_count = 0;
        self.timeouts_count = 0;
        self.fuel_exhausted_count = 0;
        self.instances_recycled = 0;
    }

    /// Generate a summary report.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            steps_completed: self.steps_completed,
            resets_completed: self.resets_completed,
            step_p50_us: self.step_latency.p50().as_micros() as u64,
            step_p99_us: self.step_latency.p99().as_micros() as u64,
            reset_p50_us: self.reset_latency.p50().as_micros() as u64,
            reset_p99_us: self.reset_latency.p99().as_micros() as u64,
            traps_count: self.traps_count,
            timeouts_count: self.timeouts_count,
            throughput_steps_per_sec: self.throughput_steps_per_sec(),
        }
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of runtime metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Total steps completed.
    pub steps_completed: u64,
    /// Total resets completed.
    pub resets_completed: u64,
    /// Step p50 latency in microseconds.
    pub step_p50_us: u64,
    /// Step p99 latency in microseconds.
    pub step_p99_us: u64,
    /// Reset p50 latency in microseconds.
    pub reset_p50_us: u64,
    /// Reset p99 latency in microseconds.
    pub reset_p99_us: u64,
    /// Number of traps.
    pub traps_count: u64,
    /// Number of timeouts.
    pub timeouts_count: u64,
    /// Throughput in steps per second.
    pub throughput_steps_per_sec: f64,
}

/// RAII timer for measuring operation duration.
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time since timer start.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop timer and return elapsed duration.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats_basic() {
        let mut stats = LatencyStats::new(100);

        stats.record(Duration::from_micros(100));
        stats.record(Duration::from_micros(200));
        stats.record(Duration::from_micros(150));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.min(), Duration::from_micros(100));
        assert_eq!(stats.max(), Duration::from_micros(200));
    }

    #[test]
    fn test_latency_stats_mean() {
        let mut stats = LatencyStats::new(100);

        for _ in 0..10 {
            stats.record(Duration::from_micros(100));
        }

        assert_eq!(stats.mean(), Duration::from_micros(100));
    }

    #[test]
    fn test_latency_stats_percentile() {
        let mut stats = LatencyStats::new(100);

        // Add 100 samples: 1, 2, 3, ..., 100 microseconds
        for i in 1..=100 {
            stats.record(Duration::from_micros(i));
        }

        // p50 should be around 50
        let p50 = stats.p50();
        assert!(p50.as_micros() >= 49 && p50.as_micros() <= 51);

        // p99 should be around 99
        let p99 = stats.p99();
        assert!(p99.as_micros() >= 98 && p99.as_micros() <= 100);
    }

    #[test]
    fn test_latency_stats_reset() {
        let mut stats = LatencyStats::new(100);
        stats.record(Duration::from_micros(100));
        assert_eq!(stats.count, 1);

        stats.reset();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean_ns(), 0);
    }

    #[test]
    fn test_latency_stats_max_samples() {
        let mut stats = LatencyStats::new(10);

        // Add more samples than max
        for i in 0..20 {
            stats.record(Duration::from_micros(i));
        }

        // Should only retain last 10 samples
        assert_eq!(stats.samples.len(), 10);
        assert_eq!(stats.count, 20); // But count tracks all
    }

    #[test]
    fn test_runtime_metrics_step() {
        let mut metrics = RuntimeMetrics::new();

        metrics.record_step(Duration::from_micros(100));
        metrics.record_step(Duration::from_micros(200));

        assert_eq!(metrics.steps_completed, 2);
        assert_eq!(metrics.step_latency.count, 2);
    }

    #[test]
    fn test_runtime_metrics_batch() {
        let mut metrics = RuntimeMetrics::new();

        metrics.record_batch_step(Duration::from_millis(10), 100);

        assert_eq!(metrics.steps_completed, 100);
        assert_eq!(metrics.batch_step_latency.count, 1);
    }

    #[test]
    fn test_runtime_metrics_errors() {
        let mut metrics = RuntimeMetrics::new();

        metrics.record_trap();
        metrics.record_timeout();
        metrics.record_fuel_exhausted();
        metrics.record_instance_recycled();

        assert_eq!(metrics.traps_count, 1);
        assert_eq!(metrics.timeouts_count, 1);
        assert_eq!(metrics.fuel_exhausted_count, 1);
        assert_eq!(metrics.instances_recycled, 1);
    }

    #[test]
    fn test_runtime_metrics_summary() {
        let mut metrics = RuntimeMetrics::new();

        for _ in 0..100 {
            metrics.record_step(Duration::from_micros(100));
        }
        metrics.record_trap();

        let summary = metrics.summary();
        assert_eq!(summary.steps_completed, 100);
        assert_eq!(summary.traps_count, 1);
        assert!(summary.throughput_steps_per_sec > 0.0);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(10));
    }
}
