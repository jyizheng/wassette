// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Metrics collection for comparison benchmarks.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::backend::Backend;

/// Aggregated comparison metrics across all backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// Metrics per backend.
    pub backends: HashMap<String, BackendMetrics>,

    /// Scaling metrics.
    pub scaling: Option<ScalingMetrics>,

    /// Cold-start metrics.
    pub cold_start: Option<ColdStartMetrics>,

    /// Hardware configuration used.
    pub hardware_info: Option<String>,
}

impl ComparisonMetrics {
    /// Create new comparison metrics.
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            scaling: None,
            cold_start: None,
            hardware_info: None,
        }
    }

    /// Add backend metrics.
    pub fn add_backend(&mut self, metrics: BackendMetrics) {
        self.backends.insert(metrics.backend.to_string(), metrics);
    }

    /// Get backend metrics.
    pub fn get_backend(&self, backend: &Backend) -> Option<&BackendMetrics> {
        self.backends.get(&backend.to_string())
    }

    /// Set scaling metrics.
    pub fn with_scaling(mut self, scaling: ScalingMetrics) -> Self {
        self.scaling = Some(scaling);
        self
    }

    /// Set cold-start metrics.
    pub fn with_cold_start(mut self, cold_start: ColdStartMetrics) -> Self {
        self.cold_start = Some(cold_start);
        self
    }

    /// Generate comparison table.
    pub fn comparison_table(&self) -> Vec<ComparisonRow> {
        let baseline = self.backends.get(&Backend::WasmInproc.to_string());

        self.backends
            .values()
            .map(|m| {
                let speedup = baseline
                    .map(|b| b.step_mean_us as f64 / m.step_mean_us.max(1) as f64)
                    .unwrap_or(1.0);

                ComparisonRow {
                    backend: m.backend.to_string(),
                    step_mean_us: m.step_mean_us,
                    step_p99_us: m.step_p99_us,
                    reset_mean_us: m.reset_mean_us,
                    throughput_sps: m.throughput_steps_per_sec,
                    speedup_vs_baseline: speedup,
                }
            })
            .collect()
    }
}

impl Default for ComparisonMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for a single backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetrics {
    /// Backend type.
    pub backend: Backend,

    /// Number of samples collected.
    pub samples: usize,

    /// Mean step latency in microseconds.
    pub step_mean_us: u64,

    /// P50 step latency in microseconds.
    pub step_p50_us: u64,

    /// P99 step latency in microseconds.
    pub step_p99_us: u64,

    /// P99.9 step latency in microseconds.
    pub step_p999_us: u64,

    /// Min step latency in microseconds.
    pub step_min_us: u64,

    /// Max step latency in microseconds.
    pub step_max_us: u64,

    /// Mean reset latency in microseconds.
    pub reset_mean_us: u64,

    /// P99 reset latency in microseconds.
    pub reset_p99_us: u64,

    /// Throughput in steps per second.
    pub throughput_steps_per_sec: f64,

    /// Step latencies (raw data for percentile calculation).
    #[serde(skip)]
    step_latencies: Vec<Duration>,

    /// Reset latencies (raw data).
    #[serde(skip)]
    reset_latencies: Vec<Duration>,
}

impl BackendMetrics {
    /// Create new backend metrics.
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            samples: 0,
            step_mean_us: 0,
            step_p50_us: 0,
            step_p99_us: 0,
            step_p999_us: 0,
            step_min_us: 0,
            step_max_us: 0,
            reset_mean_us: 0,
            reset_p99_us: 0,
            throughput_steps_per_sec: 0.0,
            step_latencies: Vec::new(),
            reset_latencies: Vec::new(),
        }
    }

    /// Record a step latency.
    pub fn record_step(&mut self, duration: Duration) {
        self.step_latencies.push(duration);
        self.samples += 1;
        self.recompute_stats();
    }

    /// Record a reset latency.
    pub fn record_reset(&mut self, duration: Duration) {
        self.reset_latencies.push(duration);
        self.recompute_reset_stats();
    }

    /// Recompute statistics from raw data.
    fn recompute_stats(&mut self) {
        if self.step_latencies.is_empty() {
            return;
        }

        let mut sorted: Vec<u64> = self.step_latencies.iter().map(|d| d.as_micros() as u64).collect();
        sorted.sort();

        let n = sorted.len();
        let sum: u64 = sorted.iter().sum();

        self.step_mean_us = sum / n as u64;
        self.step_min_us = sorted[0];
        self.step_max_us = sorted[n - 1];
        self.step_p50_us = sorted[n / 2];
        self.step_p99_us = sorted[(n * 99) / 100];
        self.step_p999_us = sorted[(n * 999) / 1000];

        // Throughput
        if self.step_mean_us > 0 {
            self.throughput_steps_per_sec = 1_000_000.0 / self.step_mean_us as f64;
        }
    }

    /// Recompute reset statistics.
    fn recompute_reset_stats(&mut self) {
        if self.reset_latencies.is_empty() {
            return;
        }

        let mut sorted: Vec<u64> = self.reset_latencies.iter().map(|d| d.as_micros() as u64).collect();
        sorted.sort();

        let n = sorted.len();
        let sum: u64 = sorted.iter().sum();

        self.reset_mean_us = sum / n as u64;
        self.reset_p99_us = sorted[(n * 99) / 100];
    }

    /// Get step latency as Duration.
    pub fn step_mean(&self) -> Duration {
        Duration::from_micros(self.step_mean_us)
    }

    /// Get step P99 as Duration.
    pub fn step_p99(&self) -> Duration {
        Duration::from_micros(self.step_p99_us)
    }
}

/// A row in the comparison table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    /// Backend name.
    pub backend: String,

    /// Mean step latency.
    pub step_mean_us: u64,

    /// P99 step latency.
    pub step_p99_us: u64,

    /// Mean reset latency.
    pub reset_mean_us: u64,

    /// Throughput.
    pub throughput_sps: f64,

    /// Speedup vs baseline.
    pub speedup_vs_baseline: f64,
}

/// Scaling metrics across different environment counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetrics {
    /// Backend tested.
    pub backend: Backend,

    /// Results at different environment counts.
    pub results: Vec<ScalingPoint>,
}

impl ScalingMetrics {
    /// Create new scaling metrics.
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            results: Vec::new(),
        }
    }

    /// Add a scaling point.
    pub fn add_point(&mut self, env_count: usize, throughput: f64, latency_p99_us: u64) {
        self.results.push(ScalingPoint {
            env_count,
            throughput_sps: throughput,
            latency_p99_us,
        });
    }

    /// Check if scaling is linear (within tolerance).
    pub fn is_linear(&self, tolerance: f64) -> bool {
        if self.results.len() < 2 {
            return true;
        }

        let first = &self.results[0];
        let base_efficiency = first.throughput_sps / first.env_count as f64;

        for point in &self.results[1..] {
            let efficiency = point.throughput_sps / point.env_count as f64;
            let ratio = efficiency / base_efficiency;
            if ratio < (1.0 - tolerance) {
                return false;
            }
        }
        true
    }
}

/// A point in scaling measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPoint {
    /// Number of environments.
    pub env_count: usize,

    /// Throughput at this count.
    pub throughput_sps: f64,

    /// P99 latency at this count.
    pub latency_p99_us: u64,
}

/// Cold-start metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMetrics {
    /// Backend tested.
    pub backend: Backend,

    /// First step latency (cold).
    pub first_step_us: u64,

    /// Warm step latency (after warmup).
    pub warm_step_us: u64,

    /// Instance creation time.
    pub instance_create_us: u64,

    /// Component load time.
    pub component_load_us: u64,
}

impl ColdStartMetrics {
    /// Create new cold-start metrics.
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            first_step_us: 0,
            warm_step_us: 0,
            instance_create_us: 0,
            component_load_us: 0,
        }
    }

    /// Calculate cold-start overhead ratio.
    pub fn overhead_ratio(&self) -> f64 {
        if self.warm_step_us == 0 {
            return 0.0;
        }
        self.first_step_us as f64 / self.warm_step_us as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_metrics_new() {
        let metrics = ComparisonMetrics::new();
        assert!(metrics.backends.is_empty());
    }

    #[test]
    fn test_backend_metrics_recording() {
        let mut metrics = BackendMetrics::new(Backend::WasmInproc);

        for i in 1..=100 {
            metrics.record_step(Duration::from_micros(i * 10));
        }

        assert_eq!(metrics.samples, 100);
        assert!(metrics.step_mean_us > 0);
        assert!(metrics.step_p99_us >= metrics.step_p50_us);
        assert!(metrics.throughput_steps_per_sec > 0.0);
    }

    #[test]
    fn test_backend_metrics_reset() {
        let mut metrics = BackendMetrics::new(Backend::DockerTask);

        for _ in 0..10 {
            metrics.record_reset(Duration::from_millis(100));
        }

        assert!(metrics.reset_mean_us >= 100_000);
    }

    #[test]
    fn test_comparison_table() {
        let mut comparison = ComparisonMetrics::new();

        let mut wasm = BackendMetrics::new(Backend::WasmInproc);
        wasm.step_mean_us = 100;
        wasm.throughput_steps_per_sec = 10000.0;
        comparison.add_backend(wasm);

        let mut docker = BackendMetrics::new(Backend::DockerTask);
        docker.step_mean_us = 5000;
        docker.throughput_steps_per_sec = 200.0;
        comparison.add_backend(docker);

        let table = comparison.comparison_table();
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_scaling_metrics() {
        let mut scaling = ScalingMetrics::new(Backend::WasmInproc);

        scaling.add_point(1, 10000.0, 100);
        scaling.add_point(4, 38000.0, 110);
        scaling.add_point(16, 140000.0, 120);

        assert!(scaling.is_linear(0.2)); // Within 20% of linear
    }

    #[test]
    fn test_cold_start_metrics() {
        let mut cold = ColdStartMetrics::new(Backend::WasmInproc);
        cold.first_step_us = 10000;
        cold.warm_step_us = 100;

        assert!((cold.overhead_ratio() - 100.0).abs() < 0.1);
    }
}
