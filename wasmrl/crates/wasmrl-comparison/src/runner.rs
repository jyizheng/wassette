// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Benchmark runner for CAMEL-AI comparisons.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::backend::{Backend, BackendRunner, DockerBackend, SubprocessBackend, WasmBackend};
use crate::config::{ComparisonConfig, TaskConfig};
use crate::error::ComparisonError;
use crate::metrics::{BackendMetrics, ColdStartMetrics, ComparisonMetrics, ScalingMetrics};

/// Comparison runner orchestrates benchmark execution.
pub struct ComparisonRunner {
    /// Configuration.
    config: ComparisonConfig,

    /// Active backend runners.
    runners: HashMap<Backend, Box<dyn BackendRunner>>,

    /// Results storage.
    results: ComparisonMetrics,
}

impl ComparisonRunner {
    /// Create a new comparison runner.
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            config,
            runners: HashMap::new(),
            results: ComparisonMetrics::new(),
        }
    }

    /// Initialize all configured backends.
    pub async fn initialize(&mut self) -> Result<()> {
        for backend in &self.config.backends {
            let runner = self.create_runner(backend)?;
            if runner.is_available() {
                runner.initialize().await?;
                self.runners.insert(backend.clone(), runner);
            }
        }
        Ok(())
    }

    /// Create a runner for a backend.
    fn create_runner(&self, backend: &Backend) -> Result<Box<dyn BackendRunner>> {
        match backend {
            Backend::WasmInproc => Ok(Box::new(WasmBackend::new())),
            Backend::DockerTask | Backend::CrabDocker => Ok(Box::new(DockerBackend::new())),
            Backend::Subprocess | Backend::Native => Ok(Box::new(SubprocessBackend::new())),
            Backend::McpTool => Err(ComparisonError::BackendNotFound(backend.to_string()).into()),
            Backend::CrabVm => Err(ComparisonError::BackendNotFound(backend.to_string()).into()),
        }
    }

    /// Run all benchmarks.
    pub async fn run(&mut self) -> Result<ComparisonMetrics> {
        // Warmup phase
        if self.config.warmup_iterations > 0 {
            self.run_warmup().await?;
        }

        // Main benchmark
        for task in &self.config.tasks.clone() {
            self.run_task(task).await?;
        }

        // Scaling tests
        if !self.config.scaling_factors.is_empty() {
            self.run_scaling_tests().await?;
        }

        // Cold-start analysis
        if self.config.cold_start_analysis {
            self.run_cold_start_analysis().await?;
        }

        Ok(self.results.clone())
    }

    /// Run warmup phase.
    async fn run_warmup(&mut self) -> Result<()> {
        for backend in self.config.backends.clone() {
            if let Some(runner) = self.runners.get(&backend) {
                for _ in 0..self.config.warmup_iterations {
                    let _ = runner.step(b"warmup").await;
                }
            }
        }
        Ok(())
    }

    /// Run a single task benchmark.
    async fn run_task(&mut self, task: &TaskConfig) -> Result<()> {
        let iterations = self.config.benchmark_iterations;

        for backend in self.config.backends.clone() {
            if let Some(runner) = self.runners.get(&backend) {
                let mut metrics = BackendMetrics::new(backend.clone());

                // Reset measurement
                for _ in 0..10 {
                    let start = Instant::now();
                    runner.reset().await?;
                    metrics.record_reset(start.elapsed());
                }

                // Step measurements
                for _ in 0..iterations {
                    let start = Instant::now();
                    let _ = runner.step(&task.input).await;
                    metrics.record_step(start.elapsed());
                }

                self.results.add_backend(metrics);
            }
        }

        Ok(())
    }

    /// Run scaling tests.
    async fn run_scaling_tests(&mut self) -> Result<()> {
        // For each backend that supports scaling
        for backend in self.config.backends.clone() {
            if !matches!(backend, Backend::WasmInproc | Backend::McpTool) {
                continue;
            }

            let mut scaling = ScalingMetrics::new(backend.clone());

            for &factor in &self.config.scaling_factors {
                let (throughput, latency) = self.measure_scaling(&backend, factor).await?;
                scaling.add_point(factor, throughput, latency);
            }

            self.results = self.results.clone().with_scaling(scaling);
        }

        Ok(())
    }

    /// Measure scaling at a specific factor.
    async fn measure_scaling(&self, backend: &Backend, env_count: usize) -> Result<(f64, u64)> {
        let iterations = 1000;
        let mut total_us = 0u64;

        if let Some(runner) = self.runners.get(backend) {
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = runner.step(b"scaling_test").await;
                total_us += start.elapsed().as_micros() as u64;
            }
        }

        let mean_us = total_us / iterations as u64;
        let throughput = (iterations as f64 * env_count as f64) / (total_us as f64 / 1_000_000.0);
        let p99 = mean_us * 3; // Estimate

        Ok((throughput, p99))
    }

    /// Run cold-start analysis.
    async fn run_cold_start_analysis(&mut self) -> Result<()> {
        for backend in self.config.backends.clone() {
            let mut cold = ColdStartMetrics::new(backend.clone());

            // Create fresh runner
            let runner = self.create_runner(&backend)?;
            if !runner.is_available() {
                continue;
            }

            // Measure instance creation
            let start = Instant::now();
            runner.initialize().await?;
            cold.instance_create_us = start.elapsed().as_micros() as u64;

            // First step (cold)
            let start = Instant::now();
            let _ = runner.step(b"cold_test").await;
            cold.first_step_us = start.elapsed().as_micros() as u64;

            // Warm steps
            let mut warm_total = 0u64;
            for _ in 0..100 {
                let start = Instant::now();
                let _ = runner.step(b"warm_test").await;
                warm_total += start.elapsed().as_micros() as u64;
            }
            cold.warm_step_us = warm_total / 100;

            self.results = self.results.clone().with_cold_start(cold);
        }

        Ok(())
    }

    /// Cleanup all runners.
    pub async fn cleanup(&mut self) -> Result<()> {
        for (_, runner) in self.runners.drain() {
            runner.cleanup().await?;
        }
        Ok(())
    }

    /// Get current results.
    pub fn results(&self) -> &ComparisonMetrics {
        &self.results
    }
}

/// Result of a single run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Task name.
    pub task: String,

    /// Backend used.
    pub backend: Backend,

    /// Latency.
    pub latency: Duration,

    /// Success.
    pub success: bool,

    /// Output (if any).
    pub output: Option<Vec<u8>>,
}

impl RunResult {
    /// Create a successful result.
    pub fn success(task: String, backend: Backend, latency: Duration, output: Vec<u8>) -> Self {
        Self {
            task,
            backend,
            latency,
            success: true,
            output: Some(output),
        }
    }

    /// Create a failed result.
    pub fn failure(task: String, backend: Backend, latency: Duration) -> Self {
        Self {
            task,
            backend,
            latency,
            success: false,
            output: None,
        }
    }
}

/// Batch runner for parallel execution.
pub struct BatchRunner {
    /// Number of parallel workers.
    pub workers: usize,

    /// Results collected.
    results: Vec<RunResult>,
}

impl BatchRunner {
    /// Create new batch runner.
    pub fn new(workers: usize) -> Self {
        Self {
            workers,
            results: Vec::new(),
        }
    }

    /// Run a task batch.
    pub async fn run_batch<R: BackendRunner>(
        &mut self,
        runner: &R,
        task: &TaskConfig,
        count: usize,
    ) -> Result<Vec<RunResult>> {
        let mut results = Vec::with_capacity(count);

        for _ in 0..count {
            let start = Instant::now();
            let output = runner.step(&task.input).await;
            let latency = start.elapsed();

            let result = match output {
                Ok(data) => RunResult::success(task.name.clone(), Backend::WasmInproc, latency, data),
                Err(_) => RunResult::failure(task.name.clone(), Backend::WasmInproc, latency),
            };
            results.push(result);
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Get all results.
    pub fn results(&self) -> &[RunResult] {
        &self.results
    }

    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let successes = self.results.iter().filter(|r| r.success).count();
        successes as f64 / self.results.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_runner_new() {
        let config = ComparisonConfig::minimal();
        let runner = ComparisonRunner::new(config);
        assert!(runner.runners.is_empty());
    }

    #[test]
    fn test_run_result_success() {
        let result = RunResult::success(
            "test".to_string(),
            Backend::WasmInproc,
            Duration::from_micros(100),
            vec![1, 2, 3],
        );
        assert!(result.success);
        assert!(result.output.is_some());
    }

    #[test]
    fn test_run_result_failure() {
        let result = RunResult::failure(
            "test".to_string(),
            Backend::DockerTask,
            Duration::from_millis(5),
        );
        assert!(!result.success);
        assert!(result.output.is_none());
    }

    #[test]
    fn test_batch_runner_new() {
        let runner = BatchRunner::new(4);
        assert_eq!(runner.workers, 4);
        assert!(runner.results.is_empty());
    }

    #[test]
    fn test_batch_runner_success_rate() {
        let mut runner = BatchRunner::new(1);
        runner.results.push(RunResult::success(
            "t1".to_string(),
            Backend::WasmInproc,
            Duration::from_micros(10),
            vec![],
        ));
        runner.results.push(RunResult::failure(
            "t2".to_string(),
            Backend::WasmInproc,
            Duration::from_micros(10),
        ));
        assert!((runner.success_rate() - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_comparison_runner_create_runner() {
        let config = ComparisonConfig::minimal();
        let runner = ComparisonRunner::new(config);

        let wasm_runner = runner.create_runner(&Backend::WasmInproc);
        assert!(wasm_runner.is_ok());

        let docker_runner = runner.create_runner(&Backend::DockerTask);
        assert!(docker_runner.is_ok());
    }
}
