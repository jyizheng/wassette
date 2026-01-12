// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Statistical utilities for benchmarking.

use std::collections::HashMap;

/// Running statistics accumulator.
#[derive(Debug, Clone, Default)]
pub struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    /// Create a new accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add a sample using Welford's online algorithm.
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Get the count.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the minimum value.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get the maximum value.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Merge with another accumulator.
    pub fn merge(&mut self, other: &RunningStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean =
            (self.count as f64 * self.mean + other.count as f64 * other.mean) / combined_count as f64;
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count * other.count) as f64 / combined_count as f64;

        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

/// Histogram for distribution analysis.
#[derive(Debug, Clone)]
pub struct Histogram {
    buckets: Vec<u64>,
    bucket_width: f64,
    min_value: f64,
    max_value: f64,
    count: u64,
}

impl Histogram {
    /// Create a histogram with specified range and bucket count.
    pub fn new(min_value: f64, max_value: f64, num_buckets: usize) -> Self {
        Self {
            buckets: vec![0; num_buckets],
            bucket_width: (max_value - min_value) / num_buckets as f64,
            min_value,
            max_value,
            count: 0,
        }
    }

    /// Add a value to the histogram.
    pub fn push(&mut self, value: f64) {
        if value < self.min_value || value >= self.max_value {
            return;
        }

        let bucket_idx = ((value - self.min_value) / self.bucket_width) as usize;
        let bucket_idx = bucket_idx.min(self.buckets.len() - 1);
        self.buckets[bucket_idx] += 1;
        self.count += 1;
    }

    /// Get percentile value.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let target = (self.count as f64 * p / 100.0) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return self.min_value + (i as f64 + 0.5) * self.bucket_width;
            }
        }

        self.max_value
    }

    /// Get the bucket counts.
    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }
}

/// Benchmark comparison result.
#[derive(Debug, Clone)]
pub struct Comparison {
    /// Name of baseline.
    pub baseline_name: String,
    /// Name of candidate.
    pub candidate_name: String,
    /// Baseline mean (e.g., latency in µs).
    pub baseline_mean: f64,
    /// Candidate mean.
    pub candidate_mean: f64,
    /// Speedup ratio (baseline / candidate, >1 means faster).
    pub speedup: f64,
    /// Whether the speedup meets the target.
    pub meets_target: bool,
    /// Target speedup for comparison.
    pub target_speedup: f64,
}

impl Comparison {
    /// Create a comparison between two measurements.
    pub fn new(
        baseline_name: &str,
        baseline_mean: f64,
        candidate_name: &str,
        candidate_mean: f64,
        target_speedup: f64,
    ) -> Self {
        let speedup = baseline_mean / candidate_mean;
        Self {
            baseline_name: baseline_name.to_string(),
            candidate_name: candidate_name.to_string(),
            baseline_mean,
            candidate_mean,
            speedup,
            meets_target: speedup >= target_speedup,
            target_speedup,
        }
    }

    /// Display the comparison result.
    pub fn display(&self) {
        let status = if self.meets_target { "✅" } else { "❌" };
        println!(
            "{} {}/{}: {:.2}x speedup (target: {:.2}x) [{}: {:.2}µs, {}: {:.2}µs]",
            status,
            self.candidate_name,
            self.baseline_name,
            self.speedup,
            self.target_speedup,
            self.baseline_name,
            self.baseline_mean,
            self.candidate_name,
            self.candidate_mean,
        );
    }
}

/// Multi-metric benchmark result collector.
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    metrics: HashMap<String, RunningStats>,
}

impl BenchmarkResults {
    /// Create a new result collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a metric value.
    pub fn record(&mut self, name: &str, value: f64) {
        self.metrics
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get a metric's statistics.
    pub fn get(&self, name: &str) -> Option<&RunningStats> {
        self.metrics.get(name)
    }

    /// Print a summary of all metrics.
    pub fn summary(&self) {
        println!("\n=== Benchmark Results ===");
        let mut names: Vec<_> = self.metrics.keys().collect();
        names.sort();

        for name in names {
            let stats = &self.metrics[name];
            println!(
                "{}: mean={:.2}, std={:.2}, min={:.2}, max={:.2}, n={}",
                name,
                stats.mean(),
                stats.std_dev(),
                stats.min(),
                stats.max(),
                stats.count()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_stats() {
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

        // Known variance for this dataset
        assert!((stats.mean() - 5.0).abs() < 0.001);
        assert!((stats.variance() - 4.571).abs() < 0.01);
    }

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new(0.0, 100.0, 10);
        
        // All in first bucket
        for _ in 0..10 {
            hist.push(5.0);
        }
        
        assert_eq!(hist.buckets()[0], 10);
        assert!(hist.percentile(50.0) < 15.0);
    }

    #[test]
    fn test_comparison() {
        let cmp = Comparison::new("scalar", 100.0, "batch", 50.0, 1.5);
        
        assert!((cmp.speedup - 2.0).abs() < 0.001);
        assert!(cmp.meets_target);
    }

    #[test]
    fn test_comparison_not_meeting_target() {
        let cmp = Comparison::new("scalar", 100.0, "batch", 90.0, 2.0);
        
        assert!(!cmp.meets_target);
    }

    #[test]
    fn test_benchmark_results() {
        let mut results = BenchmarkResults::new();
        results.record("latency_us", 100.0);
        results.record("latency_us", 120.0);
        results.record("throughput_ops", 1000.0);

        let latency = results.get("latency_us").unwrap();
        assert_eq!(latency.count(), 2);
        assert!((latency.mean() - 110.0).abs() < 0.001);
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
    }
}
