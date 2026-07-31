// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL Comparison Framework
//!
//! This crate provides tools for comparing WasmRL against CAMEL-AI baselines
//! including SETA-ENV and CRAB backends.
//!
//! # Overview
//!
//! The comparison framework supports multiple execution backends:
//!
//! - **WasmRL In-Process**: Direct Wasm execution (data plane)
//! - **WasmRL MCP**: MCP tool-based execution (control plane)
//! - **SETA-ENV Docker**: Docker-based task execution
//! - **CRAB Docker/VM**: CRAB benchmark backends
//! - **Subprocess**: Native subprocess baseline
//!
//! # Fairness Protocol
//!
//! All comparisons follow the fairness protocol documented in
//! `docs/CAMEL_COMPARISON.md`:
//!
//! 1. Semantic equivalence (same outputs for same inputs)
//! 2. Hardware normalization (same machine, CPU pinning)
//! 3. Statistical rigor (warmup, sufficient samples)
//! 4. Separate cold-start and steady-state metrics
//!
//! # Quick Start
//!
//! ```ignore
//! use wasmrl_comparison::{ComparisonConfig, ComparisonRunner, Backend};
//!
//! let config = ComparisonConfig::new()
//!     .add_backend(Backend::WasmInproc)
//!     .add_backend(Backend::DockerTask)
//!     .with_task("file-counter")
//!     .with_iterations(10000);
//!
//! let runner = ComparisonRunner::new(config)?;
//! let results = runner.run()?;
//! results.generate_report("results/comparison.md")?;
//! ```

#![warn(missing_docs)]

mod backend;
mod config;
mod error;
mod metrics;
mod report;
mod runner;
mod tasks;

// Re-export main types
pub use backend::{Backend, BackendRunner, DockerBackend, SubprocessBackend, WasmBackend};
pub use config::{ComparisonConfig, HardwareConfig, TaskConfig};
pub use error::{ComparisonError, ComparisonResult};
pub use metrics::{BackendMetrics, ComparisonMetrics, ScalingMetrics};
pub use report::{ReportFormat, ReportGenerator};
pub use runner::{ComparisonRunner, RunResult};
pub use tasks::{Task, TaskRegistry, TaskVerifier};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_config_new() {
        let config = ComparisonConfig::new();
        assert!(config.backends.is_empty());
        assert!(config.tasks.is_empty());
    }

    #[test]
    fn test_comparison_config_builder() {
        let config = ComparisonConfig::new()
            .add_backend(Backend::WasmInproc)
            .add_backend(Backend::DockerTask)
            .with_task("counter")
            .with_iterations(1000);

        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.tasks.len(), 1);
        assert_eq!(config.iterations, 1000);
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::WasmInproc.to_string(), "wasm_inproc");
        assert_eq!(Backend::McpTool.to_string(), "mcp_tool");
        assert_eq!(Backend::DockerTask.to_string(), "docker_task");
    }

    #[test]
    fn test_backend_overhead_category() {
        assert!(Backend::WasmInproc.is_inproc());
        assert!(Backend::DockerTask.is_container());
        assert!(Backend::Subprocess.is_process());
    }

    #[test]
    fn test_task_registry() {
        let registry = TaskRegistry::default();
        assert!(registry.get("counter").is_some());
    }

    #[test]
    fn test_comparison_metrics_new() {
        let metrics = ComparisonMetrics::new();
        assert!(metrics.backends.is_empty());
    }

    #[test]
    fn test_backend_metrics() {
        let metrics = BackendMetrics::new(Backend::WasmInproc);
        assert_eq!(metrics.backend, Backend::WasmInproc);
        assert_eq!(metrics.samples, 0);
    }

    #[test]
    fn test_report_format() {
        assert_eq!(ReportFormat::Markdown.extension(), "md");
        assert_eq!(ReportFormat::Json.extension(), "json");
        assert_eq!(ReportFormat::Csv.extension(), "csv");
    }
}
