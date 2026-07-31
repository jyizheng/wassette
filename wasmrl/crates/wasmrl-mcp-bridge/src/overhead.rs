// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Overhead metrics for comparing MCP vs in-process execution.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Breakdown of timing for a single tool call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimingBreakdown {
    /// RPC and serialization overhead in microseconds.
    pub rpc_serialization_us: u64,

    /// Runtime overhead (instance lookup, state management) in microseconds.
    pub runtime_overhead_us: u64,

    /// Actual environment compute time in microseconds.
    pub env_compute_us: u64,

    /// Total call time in microseconds.
    pub total_us: u64,
}

impl TimingBreakdown {
    /// Create a new timing breakdown.
    pub fn new(rpc_us: u64, runtime_us: u64, env_us: u64) -> Self {
        Self {
            rpc_serialization_us: rpc_us,
            runtime_overhead_us: runtime_us,
            env_compute_us: env_us,
            total_us: rpc_us + runtime_us + env_us,
        }
    }

    /// Create from duration measurements.
    pub fn from_durations(rpc: Duration, runtime: Duration, env: Duration) -> Self {
        Self::new(
            rpc.as_micros() as u64,
            runtime.as_micros() as u64,
            env.as_micros() as u64,
        )
    }

    /// Calculate the overhead ratio (non-env time / total time).
    pub fn overhead_ratio(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        let overhead = self.rpc_serialization_us + self.runtime_overhead_us;
        overhead as f64 / self.total_us as f64
    }

    /// Calculate the efficiency (env time / total time).
    pub fn efficiency(&self) -> f64 {
        1.0 - self.overhead_ratio()
    }
}

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self {
            rpc_serialization_us: 0,
            runtime_overhead_us: 0,
            env_compute_us: 0,
            total_us: 0,
        }
    }
}

/// Aggregated overhead metrics across multiple calls.
#[derive(Debug)]
pub struct OverheadMetrics {
    /// Total number of calls recorded.
    pub total_calls: u64,

    /// Total RPC/serialization time in microseconds.
    total_rpc_us: AtomicU64,

    /// Total runtime overhead in microseconds.
    total_runtime_us: AtomicU64,

    /// Total environment compute time in microseconds.
    total_env_us: AtomicU64,

    /// Total call time in microseconds.
    total_time_us: AtomicU64,

    /// Minimum overhead ratio observed.
    min_overhead: Mutex<f64>,

    /// Maximum overhead ratio observed.
    max_overhead: Mutex<f64>,

    /// Sum of overhead ratios (for average calculation).
    sum_overhead: Mutex<f64>,
}

impl OverheadMetrics {
    /// Create new overhead metrics.
    pub fn new() -> Self {
        Self {
            total_calls: 0,
            total_rpc_us: AtomicU64::new(0),
            total_runtime_us: AtomicU64::new(0),
            total_env_us: AtomicU64::new(0),
            total_time_us: AtomicU64::new(0),
            min_overhead: Mutex::new(f64::MAX),
            max_overhead: Mutex::new(f64::MIN),
            sum_overhead: Mutex::new(0.0),
        }
    }

    /// Record a single call's timing.
    pub fn record_call(&mut self, rpc: Duration, runtime: Duration, env: Duration) {
        let breakdown = TimingBreakdown::from_durations(rpc, runtime, env);
        self.record_breakdown(&breakdown);
    }

    /// Record a timing breakdown.
    pub fn record_breakdown(&mut self, breakdown: &TimingBreakdown) {
        self.total_calls += 1;
        self.total_rpc_us
            .fetch_add(breakdown.rpc_serialization_us, Ordering::Relaxed);
        self.total_runtime_us
            .fetch_add(breakdown.runtime_overhead_us, Ordering::Relaxed);
        self.total_env_us
            .fetch_add(breakdown.env_compute_us, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(breakdown.total_us, Ordering::Relaxed);

        let overhead = breakdown.overhead_ratio();
        {
            let mut min = self.min_overhead.lock().unwrap();
            if overhead < *min {
                *min = overhead;
            }
        }
        {
            let mut max = self.max_overhead.lock().unwrap();
            if overhead > *max {
                *max = overhead;
            }
        }
        {
            let mut sum = self.sum_overhead.lock().unwrap();
            *sum += overhead;
        }
    }

    /// Get total RPC/serialization time.
    pub fn total_rpc_time(&self) -> Duration {
        Duration::from_micros(self.total_rpc_us.load(Ordering::Relaxed))
    }

    /// Get total runtime overhead time.
    pub fn total_runtime_time(&self) -> Duration {
        Duration::from_micros(self.total_runtime_us.load(Ordering::Relaxed))
    }

    /// Get total environment compute time.
    pub fn total_env_time(&self) -> Duration {
        Duration::from_micros(self.total_env_us.load(Ordering::Relaxed))
    }

    /// Get average overhead ratio.
    pub fn avg_overhead_ratio(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        *self.sum_overhead.lock().unwrap() / self.total_calls as f64
    }

    /// Get minimum overhead ratio.
    pub fn min_overhead_ratio(&self) -> f64 {
        let min = *self.min_overhead.lock().unwrap();
        if min == f64::MAX {
            0.0
        } else {
            min
        }
    }

    /// Get maximum overhead ratio.
    pub fn max_overhead_ratio(&self) -> f64 {
        let max = *self.max_overhead.lock().unwrap();
        if max == f64::MIN {
            0.0
        } else {
            max
        }
    }

    /// Get average call duration.
    pub fn avg_call_duration(&self) -> Duration {
        if self.total_calls == 0 {
            return Duration::ZERO;
        }
        let total = self.total_time_us.load(Ordering::Relaxed);
        Duration::from_micros(total / self.total_calls)
    }

    /// Generate a summary report.
    pub fn summary(&self) -> OverheadSummary {
        OverheadSummary {
            total_calls: self.total_calls,
            avg_overhead_ratio: self.avg_overhead_ratio(),
            min_overhead_ratio: self.min_overhead_ratio(),
            max_overhead_ratio: self.max_overhead_ratio(),
            avg_call_us: self.avg_call_duration().as_micros() as u64,
            total_rpc_us: self.total_rpc_us.load(Ordering::Relaxed),
            total_runtime_us: self.total_runtime_us.load(Ordering::Relaxed),
            total_env_us: self.total_env_us.load(Ordering::Relaxed),
        }
    }
}

impl Default for OverheadMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of overhead metrics for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadSummary {
    /// Total number of calls.
    pub total_calls: u64,

    /// Average overhead ratio.
    pub avg_overhead_ratio: f64,

    /// Minimum overhead ratio.
    pub min_overhead_ratio: f64,

    /// Maximum overhead ratio.
    pub max_overhead_ratio: f64,

    /// Average call time in microseconds.
    pub avg_call_us: u64,

    /// Total RPC time in microseconds.
    pub total_rpc_us: u64,

    /// Total runtime overhead in microseconds.
    pub total_runtime_us: u64,

    /// Total environment compute time in microseconds.
    pub total_env_us: u64,
}

impl OverheadSummary {
    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        format!(
            "Overhead Report:\n\
             - Total Calls: {}\n\
             - Avg Call Time: {} µs\n\
             - Avg Overhead: {:.2}%\n\
             - Min Overhead: {:.2}%\n\
             - Max Overhead: {:.2}%\n\
             - Time Breakdown:\n\
             -   RPC/Serialization: {} µs ({:.1}%)\n\
             -   Runtime Overhead:  {} µs ({:.1}%)\n\
             -   Env Compute:       {} µs ({:.1}%)",
            self.total_calls,
            self.avg_call_us,
            self.avg_overhead_ratio * 100.0,
            self.min_overhead_ratio * 100.0,
            self.max_overhead_ratio * 100.0,
            self.total_rpc_us,
            self.rpc_percentage(),
            self.total_runtime_us,
            self.runtime_percentage(),
            self.total_env_us,
            self.env_percentage(),
        )
    }

    fn total_us(&self) -> u64 {
        self.total_rpc_us + self.total_runtime_us + self.total_env_us
    }

    fn rpc_percentage(&self) -> f64 {
        if self.total_us() == 0 {
            0.0
        } else {
            self.total_rpc_us as f64 / self.total_us() as f64 * 100.0
        }
    }

    fn runtime_percentage(&self) -> f64 {
        if self.total_us() == 0 {
            0.0
        } else {
            self.total_runtime_us as f64 / self.total_us() as f64 * 100.0
        }
    }

    fn env_percentage(&self) -> f64 {
        if self.total_us() == 0 {
            0.0
        } else {
            self.total_env_us as f64 / self.total_us() as f64 * 100.0
        }
    }
}

/// Comparison metrics between MCP and in-process execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// MCP overhead summary.
    pub mcp: OverheadSummary,

    /// In-process execution summary (baseline).
    pub inproc: OverheadSummary,
}

impl ComparisonMetrics {
    /// Create a new comparison.
    pub fn new(mcp: OverheadSummary, inproc: OverheadSummary) -> Self {
        Self { mcp, inproc }
    }

    /// Calculate speedup factor (inproc time / mcp time).
    pub fn speedup(&self) -> f64 {
        if self.inproc.avg_call_us == 0 {
            return 0.0;
        }
        self.mcp.avg_call_us as f64 / self.inproc.avg_call_us as f64
    }

    /// Generate a comparison report.
    pub fn report(&self) -> String {
        format!(
            "MCP vs In-Process Comparison:\n\
             \n\
             MCP Mode:\n\
             - Avg Call Time: {} µs\n\
             - Overhead: {:.2}%\n\
             \n\
             In-Process Mode:\n\
             - Avg Call Time: {} µs\n\
             - Overhead: {:.2}%\n\
             \n\
             Speedup: {:.2}x (in-process is {:.2}x faster)",
            self.mcp.avg_call_us,
            self.mcp.avg_overhead_ratio * 100.0,
            self.inproc.avg_call_us,
            self.inproc.avg_overhead_ratio * 100.0,
            self.speedup(),
            self.speedup(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_breakdown_new() {
        let breakdown = TimingBreakdown::new(10, 5, 85);
        assert_eq!(breakdown.rpc_serialization_us, 10);
        assert_eq!(breakdown.runtime_overhead_us, 5);
        assert_eq!(breakdown.env_compute_us, 85);
        assert_eq!(breakdown.total_us, 100);
    }

    #[test]
    fn test_timing_breakdown_overhead_ratio() {
        let breakdown = TimingBreakdown::new(10, 10, 80);
        assert!((breakdown.overhead_ratio() - 0.2).abs() < 0.001);
        assert!((breakdown.efficiency() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_timing_breakdown_from_durations() {
        let breakdown = TimingBreakdown::from_durations(
            Duration::from_micros(100),
            Duration::from_micros(50),
            Duration::from_micros(850),
        );
        assert_eq!(breakdown.total_us, 1000);
    }

    #[test]
    fn test_overhead_metrics_new() {
        let metrics = OverheadMetrics::new();
        assert_eq!(metrics.total_calls, 0);
    }

    #[test]
    fn test_overhead_metrics_record() {
        let mut metrics = OverheadMetrics::new();
        metrics.record_call(
            Duration::from_micros(10),
            Duration::from_micros(5),
            Duration::from_micros(85),
        );
        assert_eq!(metrics.total_calls, 1);
        assert_eq!(metrics.total_env_time().as_micros(), 85);
    }

    #[test]
    fn test_overhead_metrics_multiple() {
        let mut metrics = OverheadMetrics::new();
        for _ in 0..10 {
            metrics.record_call(
                Duration::from_micros(10),
                Duration::from_micros(10),
                Duration::from_micros(80),
            );
        }
        assert_eq!(metrics.total_calls, 10);
        assert!((metrics.avg_overhead_ratio() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_overhead_summary_report() {
        let mut metrics = OverheadMetrics::new();
        metrics.record_call(
            Duration::from_micros(100),
            Duration::from_micros(50),
            Duration::from_micros(850),
        );
        let summary = metrics.summary();
        let report = summary.report();

        assert!(report.contains("Total Calls: 1"));
        assert!(report.contains("RPC/Serialization"));
    }

    #[test]
    fn test_comparison_metrics() {
        let mcp = OverheadSummary {
            total_calls: 100,
            avg_overhead_ratio: 0.4,
            min_overhead_ratio: 0.3,
            max_overhead_ratio: 0.5,
            avg_call_us: 1000,
            total_rpc_us: 20000,
            total_runtime_us: 20000,
            total_env_us: 60000,
        };

        let inproc = OverheadSummary {
            total_calls: 100,
            avg_overhead_ratio: 0.05,
            min_overhead_ratio: 0.04,
            max_overhead_ratio: 0.06,
            avg_call_us: 100,
            total_rpc_us: 0,
            total_runtime_us: 500,
            total_env_us: 9500,
        };

        let comparison = ComparisonMetrics::new(mcp, inproc);
        assert!((comparison.speedup() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_overhead_metrics_empty() {
        let metrics = OverheadMetrics::new();
        assert_eq!(metrics.avg_overhead_ratio(), 0.0);
        assert_eq!(metrics.min_overhead_ratio(), 0.0);
        assert_eq!(metrics.max_overhead_ratio(), 0.0);
    }
}
