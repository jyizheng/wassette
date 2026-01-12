// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Backend definitions and runners for comparison benchmarks.

use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{ComparisonError, ComparisonResult};
use crate::metrics::BackendMetrics;

/// Execution backend for comparison benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Backend {
    /// WasmRL in-process execution (data plane).
    WasmInproc,

    /// WasmRL via MCP tools (control plane).
    McpTool,

    /// SETA-ENV Docker-based execution.
    DockerTask,

    /// CRAB Docker backend.
    CrabDocker,

    /// CRAB VM backend.
    CrabVm,

    /// Native subprocess execution.
    Subprocess,

    /// Native in-process (no Wasm).
    Native,
}

impl Backend {
    /// Check if this is an in-process backend.
    pub fn is_inproc(&self) -> bool {
        matches!(self, Self::WasmInproc | Self::Native)
    }

    /// Check if this is a container-based backend.
    pub fn is_container(&self) -> bool {
        matches!(self, Self::DockerTask | Self::CrabDocker)
    }

    /// Check if this is a VM-based backend.
    pub fn is_vm(&self) -> bool {
        matches!(self, Self::CrabVm)
    }

    /// Check if this is a process-based backend.
    pub fn is_process(&self) -> bool {
        matches!(self, Self::Subprocess)
    }

    /// Check if this involves RPC/serialization overhead.
    pub fn has_rpc_overhead(&self) -> bool {
        matches!(self, Self::McpTool | Self::DockerTask | Self::CrabDocker | Self::CrabVm | Self::Subprocess)
    }

    /// Get expected relative overhead factor.
    pub fn expected_overhead_factor(&self) -> f64 {
        match self {
            Self::WasmInproc => 1.0,
            Self::Native => 0.8,  // Slightly faster than Wasm
            Self::McpTool => 5.0,
            Self::Subprocess => 10.0,
            Self::DockerTask => 50.0,
            Self::CrabDocker => 50.0,
            Self::CrabVm => 100.0,
        }
    }

    /// Get human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::WasmInproc => "WasmRL in-process (data plane)",
            Self::McpTool => "WasmRL via MCP (control plane)",
            Self::DockerTask => "SETA-ENV Docker task",
            Self::CrabDocker => "CRAB Docker backend",
            Self::CrabVm => "CRAB VM backend",
            Self::Subprocess => "Native subprocess",
            Self::Native => "Native in-process",
        }
    }

    /// Get all backends for comparison.
    pub fn all() -> Vec<Backend> {
        vec![
            Self::WasmInproc,
            Self::McpTool,
            Self::DockerTask,
            Self::CrabDocker,
            Self::CrabVm,
            Self::Subprocess,
            Self::Native,
        ]
    }

    /// Get commonly compared backends.
    pub fn common() -> Vec<Backend> {
        vec![
            Self::WasmInproc,
            Self::McpTool,
            Self::DockerTask,
            Self::Subprocess,
        ]
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::WasmInproc => "wasm_inproc",
            Self::McpTool => "mcp_tool",
            Self::DockerTask => "docker_task",
            Self::CrabDocker => "crab_docker",
            Self::CrabVm => "crab_vm",
            Self::Subprocess => "subprocess",
            Self::Native => "native",
        };
        write!(f, "{}", name)
    }
}

impl std::str::FromStr for Backend {
    type Err = ComparisonError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wasm_inproc" | "wasm-inproc" | "inproc" => Ok(Self::WasmInproc),
            "mcp_tool" | "mcp-tool" | "mcp" => Ok(Self::McpTool),
            "docker_task" | "docker-task" | "docker" | "seta" | "seta-env" => Ok(Self::DockerTask),
            "crab_docker" | "crab-docker" => Ok(Self::CrabDocker),
            "crab_vm" | "crab-vm" => Ok(Self::CrabVm),
            "subprocess" | "subproc" | "process" => Ok(Self::Subprocess),
            "native" => Ok(Self::Native),
            _ => Err(ComparisonError::unknown_backend(s)),
        }
    }
}

/// Trait for backend runners.
pub trait BackendRunner: Send + Sync {
    /// Get the backend type.
    fn backend(&self) -> Backend;

    /// Initialize the backend.
    fn initialize(&mut self) -> ComparisonResult<()>;

    /// Run a single step and return duration.
    fn step(&mut self, action: &serde_json::Value) -> ComparisonResult<Duration>;

    /// Reset the environment and return duration.
    fn reset(&mut self, seed: u64) -> ComparisonResult<Duration>;

    /// Cleanup the backend.
    fn cleanup(&mut self) -> ComparisonResult<()>;

    /// Check if the backend is available (e.g., Docker installed).
    fn is_available(&self) -> bool;

    /// Get backend-specific metrics.
    fn collect_metrics(&self) -> BackendMetrics;
}

/// WasmRL in-process backend runner.
#[derive(Debug)]
pub struct WasmBackend {
    /// Component path.
    component_path: String,

    /// Whether initialized.
    initialized: bool,

    /// Collected metrics.
    metrics: BackendMetrics,

    /// Step count.
    step_count: u64,
}

impl WasmBackend {
    /// Create a new Wasm backend.
    pub fn new(component_path: impl Into<String>) -> Self {
        Self {
            component_path: component_path.into(),
            initialized: false,
            metrics: BackendMetrics::new(Backend::WasmInproc),
            step_count: 0,
        }
    }
}

impl BackendRunner for WasmBackend {
    fn backend(&self) -> Backend {
        Backend::WasmInproc
    }

    fn initialize(&mut self) -> ComparisonResult<()> {
        // In real implementation, would load the Wasm component
        self.initialized = true;
        Ok(())
    }

    fn step(&mut self, _action: &serde_json::Value) -> ComparisonResult<Duration> {
        if !self.initialized {
            return Err(ComparisonError::not_initialized("WasmBackend"));
        }

        let start = Instant::now();
        // Simulate step (in real impl, would call wasmrl-runtime)
        std::thread::sleep(Duration::from_micros(10));
        let elapsed = start.elapsed();

        self.step_count += 1;
        self.metrics.record_step(elapsed);

        Ok(elapsed)
    }

    fn reset(&mut self, _seed: u64) -> ComparisonResult<Duration> {
        if !self.initialized {
            return Err(ComparisonError::not_initialized("WasmBackend"));
        }

        let start = Instant::now();
        // Simulate reset
        std::thread::sleep(Duration::from_micros(50));
        let elapsed = start.elapsed();

        self.metrics.record_reset(elapsed);

        Ok(elapsed)
    }

    fn cleanup(&mut self) -> ComparisonResult<()> {
        self.initialized = false;
        Ok(())
    }

    fn is_available(&self) -> bool {
        // Wasm is always available if wasmtime is built
        true
    }

    fn collect_metrics(&self) -> BackendMetrics {
        self.metrics.clone()
    }
}

/// Docker-based backend runner (for SETA-ENV/CRAB).
#[derive(Debug)]
pub struct DockerBackend {
    /// Docker image name.
    image: String,

    /// Container ID (when running).
    container_id: Option<String>,

    /// Backend type.
    backend_type: Backend,

    /// Collected metrics.
    metrics: BackendMetrics,
}

impl DockerBackend {
    /// Create a new Docker backend.
    pub fn new(image: impl Into<String>, backend_type: Backend) -> Self {
        let backend_type = if backend_type.is_container() {
            backend_type
        } else {
            Backend::DockerTask
        };

        Self {
            image: image.into(),
            container_id: None,
            backend_type,
            metrics: BackendMetrics::new(backend_type),
        }
    }

    /// Create a SETA-ENV Docker backend.
    pub fn seta_env(task: &str) -> Self {
        Self::new(format!("camel-ai/seta-env:{}", task), Backend::DockerTask)
    }

    /// Create a CRAB Docker backend.
    pub fn crab_docker(task: &str) -> Self {
        Self::new(format!("crab-benchmark/{}:latest", task), Backend::CrabDocker)
    }
}

impl BackendRunner for DockerBackend {
    fn backend(&self) -> Backend {
        self.backend_type
    }

    fn initialize(&mut self) -> ComparisonResult<()> {
        // In real implementation, would start Docker container
        self.container_id = Some("mock-container-id".to_string());
        Ok(())
    }

    fn step(&mut self, _action: &serde_json::Value) -> ComparisonResult<Duration> {
        if self.container_id.is_none() {
            return Err(ComparisonError::not_initialized("DockerBackend"));
        }

        let start = Instant::now();
        // Simulate Docker step (much slower due to container overhead)
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = start.elapsed();

        self.metrics.record_step(elapsed);

        Ok(elapsed)
    }

    fn reset(&mut self, _seed: u64) -> ComparisonResult<Duration> {
        if self.container_id.is_none() {
            return Err(ComparisonError::not_initialized("DockerBackend"));
        }

        let start = Instant::now();
        // Simulate Docker reset (container restart)
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = start.elapsed();

        self.metrics.record_reset(elapsed);

        Ok(elapsed)
    }

    fn cleanup(&mut self) -> ComparisonResult<()> {
        self.container_id = None;
        Ok(())
    }

    fn is_available(&self) -> bool {
        // Check if Docker is available
        // In real impl, would run `docker version`
        false  // Disabled in test environment
    }

    fn collect_metrics(&self) -> BackendMetrics {
        self.metrics.clone()
    }
}

/// Subprocess-based backend runner.
#[derive(Debug)]
pub struct SubprocessBackend {
    /// Command to run.
    command: String,

    /// Arguments.
    args: Vec<String>,

    /// Whether initialized.
    initialized: bool,

    /// Collected metrics.
    metrics: BackendMetrics,
}

impl SubprocessBackend {
    /// Create a new subprocess backend.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            initialized: false,
            metrics: BackendMetrics::new(Backend::Subprocess),
        }
    }

    /// Add command arguments.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }
}

impl BackendRunner for SubprocessBackend {
    fn backend(&self) -> Backend {
        Backend::Subprocess
    }

    fn initialize(&mut self) -> ComparisonResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn step(&mut self, _action: &serde_json::Value) -> ComparisonResult<Duration> {
        if !self.initialized {
            return Err(ComparisonError::not_initialized("SubprocessBackend"));
        }

        let start = Instant::now();
        // Simulate subprocess step
        std::thread::sleep(Duration::from_micros(100));
        let elapsed = start.elapsed();

        self.metrics.record_step(elapsed);

        Ok(elapsed)
    }

    fn reset(&mut self, _seed: u64) -> ComparisonResult<Duration> {
        if !self.initialized {
            return Err(ComparisonError::not_initialized("SubprocessBackend"));
        }

        let start = Instant::now();
        // Simulate subprocess reset
        std::thread::sleep(Duration::from_micros(500));
        let elapsed = start.elapsed();

        self.metrics.record_reset(elapsed);

        Ok(elapsed)
    }

    fn cleanup(&mut self) -> ComparisonResult<()> {
        self.initialized = false;
        Ok(())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn collect_metrics(&self) -> BackendMetrics {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_variants() {
        assert!(Backend::WasmInproc.is_inproc());
        assert!(Backend::Native.is_inproc());
        assert!(!Backend::DockerTask.is_inproc());

        assert!(Backend::DockerTask.is_container());
        assert!(Backend::CrabDocker.is_container());
        assert!(!Backend::WasmInproc.is_container());

        assert!(Backend::CrabVm.is_vm());
        assert!(!Backend::DockerTask.is_vm());

        assert!(Backend::Subprocess.is_process());
    }

    #[test]
    fn test_backend_parse() {
        assert_eq!("wasm_inproc".parse::<Backend>().unwrap(), Backend::WasmInproc);
        assert_eq!("docker".parse::<Backend>().unwrap(), Backend::DockerTask);
        assert_eq!("seta-env".parse::<Backend>().unwrap(), Backend::DockerTask);
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::WasmInproc.to_string(), "wasm_inproc");
        assert_eq!(Backend::CrabDocker.to_string(), "crab_docker");
    }

    #[test]
    fn test_backend_overhead() {
        assert!(Backend::WasmInproc.expected_overhead_factor() < Backend::DockerTask.expected_overhead_factor());
        assert!(Backend::McpTool.expected_overhead_factor() < Backend::DockerTask.expected_overhead_factor());
    }

    #[test]
    fn test_wasm_backend() {
        let mut backend = WasmBackend::new("counter.wasm");
        assert!(backend.is_available());

        backend.initialize().unwrap();
        let step_time = backend.step(&serde_json::json!(0)).unwrap();
        assert!(step_time > Duration::ZERO);

        let reset_time = backend.reset(42).unwrap();
        assert!(reset_time > Duration::ZERO);

        backend.cleanup().unwrap();
    }

    #[test]
    fn test_subprocess_backend() {
        let mut backend = SubprocessBackend::new("python")
            .with_args(vec!["-c".to_string(), "print('hello')".to_string()]);

        backend.initialize().unwrap();
        let step_time = backend.step(&serde_json::json!({})).unwrap();
        assert!(step_time > Duration::ZERO);

        backend.cleanup().unwrap();
    }

    #[test]
    fn test_backend_not_initialized() {
        let mut backend = WasmBackend::new("test.wasm");
        let result = backend.step(&serde_json::json!(0));
        assert!(result.is_err());
    }
}
