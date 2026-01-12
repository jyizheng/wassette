// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL Benchmark Utilities
//!
//! Shared utilities and helpers for the benchmark suite.
//!
//! # Benchmark Modes
//!
//! The benchmark suite supports multiple execution modes:
//!
//! - `WasmInproc`: In-process Wasm execution (data plane, fastest)
//! - `McpTool`: MCP tool calls (control plane, includes RPC overhead)
//! - `Native`: Native code baseline (no Wasm overhead)
//! - `Subprocess`: Subprocess execution baseline
//! - `Docker`: Docker container baseline
//!
//! # Example
//!
//! ```ignore
//! use wasmrl_bench::{BenchConfig, BenchMode, BenchRunner};
//!
//! let config = BenchConfig::default()
//!     .with_mode(BenchMode::WasmInproc)
//!     .with_num_envs(256);
//!
//! let runner = BenchRunner::new(config);
//! let results = runner.run_step_benchmark()?;
//! ```

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub mod stats;

/// Benchmark execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchMode {
    /// In-process WebAssembly execution (data plane).
    /// This is the fastest mode, suitable for RL training.
    WasmInproc,

    /// MCP tool call execution (control plane).
    /// Includes RPC and serialization overhead.
    McpTool,

    /// Native code baseline (no WebAssembly).
    Native,

    /// Subprocess execution baseline.
    Subprocess,

    /// Docker container baseline.
    Docker,
}

impl BenchMode {
    /// Check if this is an in-process mode.
    pub fn is_inproc(&self) -> bool {
        matches!(self, Self::WasmInproc | Self::Native)
    }

    /// Check if this involves RPC overhead.
    pub fn has_rpc_overhead(&self) -> bool {
        matches!(self, Self::McpTool | Self::Subprocess | Self::Docker)
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::WasmInproc => "In-process WebAssembly (data plane)",
            Self::McpTool => "MCP tool calls (control plane)",
            Self::Native => "Native code baseline",
            Self::Subprocess => "Subprocess execution",
            Self::Docker => "Docker container",
        }
    }
}

impl Default for BenchMode {
    fn default() -> Self {
        Self::WasmInproc
    }
}

impl std::fmt::Display for BenchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::WasmInproc => "wasm_inproc",
            Self::McpTool => "mcp_tool",
            Self::Native => "native",
            Self::Subprocess => "subprocess",
            Self::Docker => "docker",
        };
        write!(f, "{}", name)
    }
}

impl std::str::FromStr for BenchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wasm_inproc" | "wasm-inproc" | "inproc" => Ok(Self::WasmInproc),
            "mcp_tool" | "mcp-tool" | "mcp" => Ok(Self::McpTool),
            "native" => Ok(Self::Native),
            "subprocess" | "subproc" => Ok(Self::Subprocess),
            "docker" => Ok(Self::Docker),
            _ => Err(format!("Unknown bench mode: {}", s)),
        }
    }
}

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Execution mode.
    pub mode: BenchMode,
    /// Number of warmup iterations.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Number of environments for batch tests.
    pub num_envs: usize,
    /// Steps per episode.
    pub steps_per_episode: usize,
    /// Whether to enable detailed reporting.
    pub verbose: bool,
    /// Output directory for results.
    pub output_dir: Option<String>,
}

impl BenchConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the execution mode.
    pub fn with_mode(mut self, mode: BenchMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the number of environments.
    pub fn with_num_envs(mut self, n: usize) -> Self {
        self.num_envs = n;
        self
    }

    /// Set the number of warmup iterations.
    pub fn with_warmup(mut self, n: usize) -> Self {
        self.warmup_iters = n;
        self
    }

    /// Set the number of measurement iterations.
    pub fn with_iters(mut self, n: usize) -> Self {
        self.measure_iters = n;
        self
    }

    /// Set output directory.
    pub fn with_output_dir(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = Some(dir.into());
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            mode: BenchMode::WasmInproc,
            warmup_iters: 10,
            measure_iters: 100,
            num_envs: 256,
            steps_per_episode: 100,
            verbose: false,
            output_dir: None,
        }
    }
}

/// Comparison result between two modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeComparison {
    /// Baseline mode.
    pub baseline_mode: String,
    /// Comparison mode.
    pub compare_mode: String,
    /// Baseline timing.
    pub baseline_mean_us: u64,
    /// Comparison timing.
    pub compare_mean_us: u64,
    /// Speedup factor (baseline / compare).
    pub speedup: f64,
    /// Overhead percentage ((compare - baseline) / baseline * 100).
    pub overhead_percent: f64,
}

impl ModeComparison {
    /// Create a new mode comparison.
    pub fn new(
        baseline_mode: BenchMode,
        compare_mode: BenchMode,
        baseline_result: &TimingResult,
        compare_result: &TimingResult,
    ) -> Self {
        let baseline_us = baseline_result.mean.as_micros() as u64;
        let compare_us = compare_result.mean.as_micros() as u64;

        let speedup = if compare_us > 0 {
            baseline_us as f64 / compare_us as f64
        } else {
            0.0
        };

        let overhead = if baseline_us > 0 {
            (compare_us as f64 - baseline_us as f64) / baseline_us as f64 * 100.0
        } else {
            0.0
        };

        Self {
            baseline_mode: baseline_mode.to_string(),
            compare_mode: compare_mode.to_string(),
            baseline_mean_us: baseline_us,
            compare_mean_us: compare_us,
            speedup,
            overhead_percent: overhead,
        }
    }

    /// Generate a comparison report.
    pub fn report(&self) -> String {
        format!(
            "Mode Comparison: {} vs {}\n\
             - {} mean: {} µs\n\
             - {} mean: {} µs\n\
             - Speedup: {:.2}x\n\
             - Overhead: {:.1}%",
            self.baseline_mode,
            self.compare_mode,
            self.baseline_mode,
            self.baseline_mean_us,
            self.compare_mode,
            self.compare_mean_us,
            self.speedup,
            self.overhead_percent,
        )
    }
}

/// Timing result with statistics.
#[derive(Debug, Clone)]
pub struct TimingResult {
    /// Mean duration.
    pub mean: Duration,
    /// Standard deviation.
    pub std_dev: Duration,
    /// Minimum duration.
    pub min: Duration,
    /// Maximum duration.
    pub max: Duration,
    /// P50 (median) duration.
    pub p50: Duration,
    /// P99 duration.
    pub p99: Duration,
    /// Number of samples.
    pub samples: usize,
}

impl TimingResult {
    /// Compute from a list of durations.
    pub fn from_durations(mut durations: Vec<Duration>) -> Self {
        if durations.is_empty() {
            return Self {
                mean: Duration::ZERO,
                std_dev: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                p50: Duration::ZERO,
                p99: Duration::ZERO,
                samples: 0,
            };
        }

        durations.sort();
        let n = durations.len();

        let sum: Duration = durations.iter().sum();
        let mean = sum / n as u32;

        let min = durations[0];
        let max = durations[n - 1];
        let p50 = durations[n / 2];
        let p99 = durations[(n * 99) / 100];

        // Compute standard deviation
        let mean_nanos = mean.as_nanos() as f64;
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        Self {
            mean,
            std_dev,
            min,
            max,
            p50,
            p99,
            samples: n,
        }
    }

    /// Throughput in operations per second.
    pub fn throughput(&self, ops_per_iter: usize) -> f64 {
        if self.mean.is_zero() {
            return 0.0;
        }
        ops_per_iter as f64 / self.mean.as_secs_f64()
    }

    /// Pretty print the result.
    pub fn display(&self, name: &str) {
        println!(
            "{}: mean={:?}, p50={:?}, p99={:?}, min={:?}, max={:?} (n={})",
            name, self.mean, self.p50, self.p99, self.min, self.max, self.samples
        );
    }
}

/// Simple timer for manual benchmarking.
pub struct Timer {
    start: Instant,
    durations: Vec<Duration>,
}

impl Timer {
    /// Create a new timer.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            durations: Vec::new(),
        }
    }

    /// Start timing.
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// Stop timing and record duration.
    pub fn stop(&mut self) {
        let elapsed = self.start.elapsed();
        self.durations.push(elapsed);
    }

    /// Get timing result.
    pub fn result(&self) -> TimingResult {
        TimingResult::from_durations(self.durations.clone())
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.durations.clear();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Measure a closure multiple times.
pub fn measure<F>(iterations: usize, mut f: F) -> TimingResult
where
    F: FnMut(),
{
    let mut durations = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }
    
    TimingResult::from_durations(durations)
}

/// Measure with warmup.
pub fn measure_with_warmup<F>(warmup: usize, iterations: usize, mut f: F) -> TimingResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..warmup {
        f();
    }
    
    // Measure
    measure(iterations, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_mode_default() {
        assert_eq!(BenchMode::default(), BenchMode::WasmInproc);
    }

    #[test]
    fn test_bench_mode_parse() {
        assert_eq!("wasm_inproc".parse::<BenchMode>().unwrap(), BenchMode::WasmInproc);
        assert_eq!("mcp_tool".parse::<BenchMode>().unwrap(), BenchMode::McpTool);
        assert_eq!("mcp".parse::<BenchMode>().unwrap(), BenchMode::McpTool);
        assert_eq!("native".parse::<BenchMode>().unwrap(), BenchMode::Native);
        assert_eq!("docker".parse::<BenchMode>().unwrap(), BenchMode::Docker);
    }

    #[test]
    fn test_bench_mode_display() {
        assert_eq!(BenchMode::WasmInproc.to_string(), "wasm_inproc");
        assert_eq!(BenchMode::McpTool.to_string(), "mcp_tool");
    }

    #[test]
    fn test_bench_mode_properties() {
        assert!(BenchMode::WasmInproc.is_inproc());
        assert!(BenchMode::Native.is_inproc());
        assert!(!BenchMode::McpTool.is_inproc());

        assert!(BenchMode::McpTool.has_rpc_overhead());
        assert!(!BenchMode::WasmInproc.has_rpc_overhead());
    }

    #[test]
    fn test_bench_config_builder() {
        let config = BenchConfig::new()
            .with_mode(BenchMode::McpTool)
            .with_num_envs(128)
            .with_warmup(5)
            .with_iters(50)
            .with_verbose(true);

        assert_eq!(config.mode, BenchMode::McpTool);
        assert_eq!(config.num_envs, 128);
        assert_eq!(config.warmup_iters, 5);
        assert_eq!(config.measure_iters, 50);
        assert!(config.verbose);
    }

    #[test]
    fn test_mode_comparison() {
        let inproc = TimingResult {
            mean: Duration::from_micros(100),
            std_dev: Duration::ZERO,
            min: Duration::ZERO,
            max: Duration::ZERO,
            p50: Duration::ZERO,
            p99: Duration::ZERO,
            samples: 100,
        };

        let mcp = TimingResult {
            mean: Duration::from_micros(1000),
            std_dev: Duration::ZERO,
            min: Duration::ZERO,
            max: Duration::ZERO,
            p50: Duration::ZERO,
            p99: Duration::ZERO,
            samples: 100,
        };

        let comparison = ModeComparison::new(
            BenchMode::WasmInproc,
            BenchMode::McpTool,
            &inproc,
            &mcp,
        );

        assert_eq!(comparison.baseline_mean_us, 100);
        assert_eq!(comparison.compare_mean_us, 1000);
        assert!((comparison.speedup - 0.1).abs() < 0.01);
        assert!((comparison.overhead_percent - 900.0).abs() < 1.0);
    }

    #[test]
    fn test_timing_result_from_durations() {
        let durations = vec![
            Duration::from_micros(100),
            Duration::from_micros(200),
            Duration::from_micros(150),
            Duration::from_micros(120),
            Duration::from_micros(180),
        ];

        let result = TimingResult::from_durations(durations);
        assert_eq!(result.samples, 5);
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
    fn test_timer() {
        let mut timer = Timer::new();
        
        for _ in 0..5 {
            timer.start();
            std::thread::sleep(Duration::from_micros(100));
            timer.stop();
        }
        
        let result = timer.result();
        assert_eq!(result.samples, 5);
        assert!(result.min >= Duration::from_micros(100));
    }

    #[test]
    fn test_measure() {
        let result = measure(10, || {
            std::thread::sleep(Duration::from_micros(50));
        });
        
        assert_eq!(result.samples, 10);
    }

    #[test]
    fn test_throughput() {
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
}
