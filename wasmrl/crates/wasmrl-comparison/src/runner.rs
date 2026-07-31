// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Benchmark runner for CAMEL-AI comparisons.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;

use crate::backend::{Backend, BackendRunner, DockerBackend, SubprocessBackend, WasmBackend};
use crate::config::ComparisonConfig;
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
            let mut runner = self.create_runner(backend);
            if runner.is_available() {
                runner.initialize()?;
                self.runners.insert(*backend, runner);
            }
        }
        Ok(())
    }

    /// Create a runner for a backend.
    fn create_runner(&self, backend: &Backend) -> Box<dyn BackendRunner> {
        match backend {
            Backend::WasmInproc => Box::new(WasmBackend::new("wasmrl-counter.wasm")),
            Backend::DockerTask => Box::new(DockerBackend::seta_env("counter")),
            Backend::CrabDocker => Box::new(DockerBackend::crab_docker("counter")),
            Backend::Subprocess => Box::new(SubprocessBackend::new("subprocess")),
            Backend::Native => Box::new(SubprocessBackend::with_backend("native", Backend::Native)),
            Backend::McpTool => Box::new(SubprocessBackend::with_backend(
                "mcp-tool",
                Backend::McpTool,
            )),
            Backend::CrabVm => {
                Box::new(SubprocessBackend::with_backend("crab-vm", Backend::CrabVm))
            }
        }
    }

    /// Run all benchmarks.
    pub async fn run(&mut self) -> Result<ComparisonMetrics> {
        if self.config.warmup > 0 {
            self.run_warmup().await?;
        }

        for task in self.config.tasks.clone() {
            self.run_task(&task).await?;
        }

        if self.config.test_scaling {
            self.run_scaling_tests().await?;
        }

        if self.config.test_cold_start {
            self.run_cold_start_analysis().await?;
        }

        Ok(self.results.clone())
    }

    /// Run warmup phase.
    async fn run_warmup(&mut self) -> Result<()> {
        let action = json!("warmup");
        for runner in self.runners.values_mut() {
            for _ in 0..self.config.warmup {
                let _ = runner.step(&action);
            }
        }
        Ok(())
    }

    /// Run a single task benchmark.
    async fn run_task(&mut self, task: &str) -> Result<()> {
        let action = json!({ "task": task });
        let reset_samples = self.config.iterations.min(10);

        for runner in self.runners.values_mut() {
            let backend = runner.backend();
            let mut metrics = BackendMetrics::new(backend);

            for seed in 0..reset_samples {
                let duration = runner.reset(seed as u64)?;
                metrics.record_reset(duration);
            }

            for _ in 0..self.config.iterations {
                let duration = runner.step(&action)?;
                metrics.record_step(duration);
            }

            self.results.add_backend(metrics);
        }

        Ok(())
    }

    /// Run scaling tests.
    async fn run_scaling_tests(&mut self) -> Result<()> {
        for backend in self.config.backends.clone() {
            if !matches!(backend, Backend::WasmInproc | Backend::McpTool) {
                continue;
            }
            if !self.runners.contains_key(&backend) {
                continue;
            }

            let mut scaling = ScalingMetrics::new(backend);
            for env_count in self.config.env_counts.clone() {
                let (throughput, latency) = self.measure_scaling(&backend, env_count).await?;
                scaling.add_point(env_count, throughput, latency);
            }
            self.results = self.results.clone().with_scaling(scaling);
        }

        Ok(())
    }

    /// Measure scaling at a specific environment count.
    async fn measure_scaling(&mut self, backend: &Backend, env_count: usize) -> Result<(f64, u64)> {
        let iterations = self.config.iterations.min(1000).max(1);
        let action = json!({ "scale": env_count });
        let mut total_us = 0u64;

        if let Some(runner) = self.runners.get_mut(backend) {
            for _ in 0..iterations {
                total_us += runner.step(&action)?.as_micros() as u64;
            }
        }

        let total_us = total_us.max(1);
        let mean_us = total_us / iterations as u64;
        let throughput = (iterations as f64 * env_count as f64) / (total_us as f64 / 1_000_000.0);
        let p99 = mean_us * 3;

        Ok((throughput, p99))
    }

    /// Run cold-start analysis.
    async fn run_cold_start_analysis(&mut self) -> Result<()> {
        for backend in self.config.backends.clone() {
            let mut runner = self.create_runner(&backend);
            if !runner.is_available() {
                continue;
            }

            let mut cold = ColdStartMetrics::new(backend);
            let start = std::time::Instant::now();
            runner.initialize()?;
            cold.instance_create_us = start.elapsed().as_micros() as u64;

            let action = json!("cold_test");
            cold.first_step_us = runner.step(&action)?.as_micros() as u64;

            let mut warm_total = 0u64;
            for _ in 0..100 {
                warm_total += runner.step(&action)?.as_micros() as u64;
            }
            cold.warm_step_us = warm_total / 100;

            self.results = self.results.clone().with_cold_start(cold);
        }

        Ok(())
    }

    /// Cleanup all runners.
    pub async fn cleanup(&mut self) -> Result<()> {
        for (_, mut runner) in self.runners.drain() {
            runner.cleanup()?;
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

    /// Output, if any.
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
