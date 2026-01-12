// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Configuration for comparison benchmarks.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::backend::Backend;

/// Configuration for comparison benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Backends to compare.
    pub backends: Vec<Backend>,

    /// Tasks to run.
    pub tasks: Vec<String>,

    /// Number of warmup iterations.
    pub warmup: usize,

    /// Number of measurement iterations.
    pub iterations: usize,

    /// Environment counts to test scaling.
    pub env_counts: Vec<usize>,

    /// Batch sizes to test.
    pub batch_sizes: Vec<usize>,

    /// Output directory for results.
    pub output_dir: Option<String>,

    /// Hardware configuration.
    pub hardware: HardwareConfig,

    /// Whether to run cold-start tests.
    pub test_cold_start: bool,

    /// Whether to run scaling tests.
    pub test_scaling: bool,

    /// Verbose output.
    pub verbose: bool,
}

impl ComparisonConfig {
    /// Create a new comparison configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a backend to compare.
    pub fn add_backend(mut self, backend: Backend) -> Self {
        if !self.backends.contains(&backend) {
            self.backends.push(backend);
        }
        self
    }

    /// Add multiple backends.
    pub fn with_backends(mut self, backends: Vec<Backend>) -> Self {
        for backend in backends {
            if !self.backends.contains(&backend) {
                self.backends.push(backend);
            }
        }
        self
    }

    /// Add a task to run.
    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        let task = task.into();
        if !self.tasks.contains(&task) {
            self.tasks.push(task);
        }
        self
    }

    /// Add multiple tasks.
    pub fn with_tasks(mut self, tasks: Vec<String>) -> Self {
        for task in tasks {
            if !self.tasks.contains(&task) {
                self.tasks.push(task);
            }
        }
        self
    }

    /// Set warmup iterations.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set measurement iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set environment counts for scaling tests.
    pub fn with_env_counts(mut self, counts: Vec<usize>) -> Self {
        self.env_counts = counts;
        self
    }

    /// Set batch sizes.
    pub fn with_batch_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.batch_sizes = sizes;
        self
    }

    /// Set output directory.
    pub fn with_output(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = Some(dir.into());
        self
    }

    /// Set hardware configuration.
    pub fn with_hardware(mut self, hardware: HardwareConfig) -> Self {
        self.hardware = hardware;
        self
    }

    /// Enable cold-start testing.
    pub fn with_cold_start(mut self, enable: bool) -> Self {
        self.test_cold_start = enable;
        self
    }

    /// Enable scaling tests.
    pub fn with_scaling(mut self, enable: bool) -> Self {
        self.test_scaling = enable;
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Create a minimal config for quick tests.
    pub fn minimal() -> Self {
        Self::new()
            .add_backend(Backend::WasmInproc)
            .with_task("counter")
            .with_iterations(100)
            .with_warmup(10)
    }

    /// Create a full comparison config.
    pub fn full() -> Self {
        Self::new()
            .with_backends(Backend::common())
            .with_tasks(vec![
                "counter".to_string(),
                "file-counter".to_string(),
                "json-transform".to_string(),
            ])
            .with_iterations(10000)
            .with_warmup(100)
            .with_env_counts(vec![1, 4, 16, 64, 256])
            .with_batch_sizes(vec![1, 8, 32])
            .with_cold_start(true)
            .with_scaling(true)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.backends.is_empty() {
            return Err("No backends specified".to_string());
        }
        if self.tasks.is_empty() {
            return Err("No tasks specified".to_string());
        }
        if self.iterations == 0 {
            return Err("Iterations must be > 0".to_string());
        }
        Ok(())
    }
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            backends: Vec::new(),
            tasks: Vec::new(),
            warmup: 100,
            iterations: 10000,
            env_counts: vec![1, 4, 16, 64, 256],
            batch_sizes: vec![1, 8, 32],
            output_dir: None,
            hardware: HardwareConfig::default(),
            test_cold_start: true,
            test_scaling: true,
            verbose: false,
        }
    }
}

/// Task-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Task name.
    pub name: String,

    /// Task description.
    pub description: String,

    /// Wasm component path (for WasmRL).
    pub wasm_path: Option<String>,

    /// Docker image (for SETA-ENV/CRAB).
    pub docker_image: Option<String>,

    /// Native command (for subprocess).
    pub native_command: Option<String>,

    /// Environment variables.
    pub env_vars: std::collections::HashMap<String, String>,

    /// Expected output for verification.
    pub expected_output: Option<serde_json::Value>,
}

impl TaskConfig {
    /// Create a new task configuration.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            wasm_path: None,
            docker_image: None,
            native_command: None,
            env_vars: std::collections::HashMap::new(),
            expected_output: None,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set Wasm component path.
    pub fn with_wasm(mut self, path: impl Into<String>) -> Self {
        self.wasm_path = Some(path.into());
        self
    }

    /// Set Docker image.
    pub fn with_docker(mut self, image: impl Into<String>) -> Self {
        self.docker_image = Some(image.into());
        self
    }

    /// Set native command.
    pub fn with_native(mut self, cmd: impl Into<String>) -> Self {
        self.native_command = Some(cmd.into());
        self
    }

    /// Add environment variable.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Set expected output.
    pub fn with_expected(mut self, output: serde_json::Value) -> Self {
        self.expected_output = Some(output);
        self
    }
}

/// Hardware configuration for benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Number of CPU cores to use.
    pub cpu_cores: usize,

    /// Memory limit in GB.
    pub memory_gb: usize,

    /// Whether to pin CPU cores.
    pub cpu_pinning: bool,

    /// CPU model (for documentation).
    pub cpu_model: Option<String>,

    /// OS information.
    pub os_info: Option<String>,
}

impl HardwareConfig {
    /// Create a new hardware configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set CPU cores.
    pub fn with_cores(mut self, cores: usize) -> Self {
        self.cpu_cores = cores;
        self
    }

    /// Set memory limit.
    pub fn with_memory_gb(mut self, gb: usize) -> Self {
        self.memory_gb = gb;
        self
    }

    /// Enable CPU pinning.
    pub fn with_pinning(mut self, pin: bool) -> Self {
        self.cpu_pinning = pin;
        self
    }

    /// Detect current hardware.
    pub fn detect() -> Self {
        Self {
            cpu_cores: num_cpus(),
            memory_gb: total_memory_gb(),
            cpu_pinning: false,
            cpu_model: Some(cpu_model()),
            os_info: Some(os_info()),
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            cpu_cores: 8,
            memory_gb: 32,
            cpu_pinning: false,
            cpu_model: None,
            os_info: None,
        }
    }
}

// Helper functions for hardware detection

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

fn total_memory_gb() -> usize {
    // Simplified - in real impl would read from /proc/meminfo
    32
}

fn cpu_model() -> String {
    "Unknown CPU".to_string()
}

fn os_info() -> String {
    std::env::consts::OS.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_config_default() {
        let config = ComparisonConfig::default();
        assert!(config.backends.is_empty());
        assert_eq!(config.warmup, 100);
        assert_eq!(config.iterations, 10000);
    }

    #[test]
    fn test_comparison_config_builder() {
        let config = ComparisonConfig::new()
            .add_backend(Backend::WasmInproc)
            .add_backend(Backend::DockerTask)
            .with_task("counter")
            .with_iterations(5000)
            .with_output("results/");

        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.tasks, vec!["counter"]);
        assert_eq!(config.iterations, 5000);
        assert_eq!(config.output_dir, Some("results/".to_string()));
    }

    #[test]
    fn test_comparison_config_no_duplicates() {
        let config = ComparisonConfig::new()
            .add_backend(Backend::WasmInproc)
            .add_backend(Backend::WasmInproc)
            .with_task("counter")
            .with_task("counter");

        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.tasks.len(), 1);
    }

    #[test]
    fn test_comparison_config_validate() {
        let empty = ComparisonConfig::new();
        assert!(empty.validate().is_err());

        let no_tasks = ComparisonConfig::new().add_backend(Backend::WasmInproc);
        assert!(no_tasks.validate().is_err());

        let valid = ComparisonConfig::minimal();
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_task_config_builder() {
        let config = TaskConfig::new("file-counter")
            .with_description("Count lines in files")
            .with_wasm("file_counter.wasm")
            .with_docker("camel-ai/seta-env:file-counter")
            .with_env("DEBUG", "1");

        assert_eq!(config.name, "file-counter");
        assert!(config.wasm_path.is_some());
        assert!(config.docker_image.is_some());
        assert_eq!(config.env_vars.get("DEBUG"), Some(&"1".to_string()));
    }

    #[test]
    fn test_hardware_config_detect() {
        let hw = HardwareConfig::detect();
        assert!(hw.cpu_cores >= 1);
        assert!(hw.cpu_model.is_some());
    }

    #[test]
    fn test_minimal_config() {
        let config = ComparisonConfig::minimal();
        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.tasks.len(), 1);
        assert_eq!(config.iterations, 100);
    }

    #[test]
    fn test_full_config() {
        let config = ComparisonConfig::full();
        assert!(config.backends.len() >= 3);
        assert!(config.tasks.len() >= 3);
        assert!(config.test_cold_start);
        assert!(config.test_scaling);
    }
}
