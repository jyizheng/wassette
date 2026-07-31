// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Task definitions for CAMEL-AI comparison benchmarks.
//!
//! This module defines the benchmark tasks selected from SETA-ENV and CRAB.

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::error::ComparisonError;

/// A benchmark task definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Task identifier.
    pub id: String,

    /// Human-readable name.
    pub name: String,

    /// Description of what the task does.
    pub description: String,

    /// Task category.
    pub category: TaskCategory,

    /// Source framework (SETA-ENV or CRAB).
    pub source: TaskSource,

    /// Input data for the task.
    pub input: Vec<u8>,

    /// Expected output (for verification).
    pub expected_output: Option<Vec<u8>>,

    /// Verification method.
    pub verification: VerificationMethod,

    /// Estimated complexity (1-10).
    pub complexity: u8,

    /// Tags for filtering.
    pub tags: Vec<String>,
}

impl Task {
    /// Create a new task.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            category: TaskCategory::FileOps,
            source: TaskSource::Custom,
            input: Vec::new(),
            expected_output: None,
            verification: VerificationMethod::OutputMatch,
            complexity: 1,
            tags: Vec::new(),
        }
    }

    /// Builder: set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Builder: set category.
    pub fn with_category(mut self, category: TaskCategory) -> Self {
        self.category = category;
        self
    }

    /// Builder: set source.
    pub fn with_source(mut self, source: TaskSource) -> Self {
        self.source = source;
        self
    }

    /// Builder: set input.
    pub fn with_input(mut self, input: Vec<u8>) -> Self {
        self.input = input;
        self
    }

    /// Builder: set expected output.
    pub fn with_expected(mut self, expected: Vec<u8>) -> Self {
        self.expected_output = Some(expected);
        self
    }

    /// Builder: set complexity.
    pub fn with_complexity(mut self, complexity: u8) -> Self {
        self.complexity = complexity.min(10);
        self
    }

    /// Builder: add tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// Task categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskCategory {
    /// File operations (read, write, transform).
    FileOps,
    /// JSON processing.
    JsonTransform,
    /// Code analysis/linting.
    CodeLint,
    /// Unit testing.
    UnitTest,
    /// API mocking/serving.
    ApiMock,
    /// Computation tasks.
    Compute,
    /// Web/HTTP operations.
    WebFetch,
    /// General purpose.
    General,
}

impl std::fmt::Display for TaskCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileOps => write!(f, "file-ops"),
            Self::JsonTransform => write!(f, "json-transform"),
            Self::CodeLint => write!(f, "code-lint"),
            Self::UnitTest => write!(f, "unit-test"),
            Self::ApiMock => write!(f, "api-mock"),
            Self::Compute => write!(f, "compute"),
            Self::WebFetch => write!(f, "web-fetch"),
            Self::General => write!(f, "general"),
        }
    }
}

/// Task source framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskSource {
    /// From SETA-ENV (Docker-based task automation).
    SetaEnv,
    /// From CRAB (Cross-platform Agent Benchmark).
    Crab,
    /// Custom task for WasmRL.
    Custom,
}

impl std::fmt::Display for TaskSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SetaEnv => write!(f, "SETA-ENV"),
            Self::Crab => write!(f, "CRAB"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Output verification method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Exact output match.
    OutputMatch,
    /// JSON schema validation.
    JsonSchema(String),
    /// Regex pattern match.
    Regex(String),
    /// Custom verifier function name.
    Custom(String),
    /// No verification (timing only).
    None,
}

/// Registry of benchmark tasks.
pub struct TaskRegistry {
    /// Tasks by ID.
    tasks: HashMap<String, Task>,
}

impl TaskRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    /// Create registry with standard SETA-ENV and CRAB tasks.
    pub fn with_standard_tasks() -> Self {
        let mut registry = Self::new();

        // SETA-ENV Tasks
        registry.register(seta_file_counter());
        registry.register(seta_json_transform());
        registry.register(seta_code_lint());
        registry.register(seta_unit_test());
        registry.register(seta_api_mock());

        // CRAB Tasks
        registry.register(crab_file_ops());
        registry.register(crab_compute());
        registry.register(crab_web_fetch());

        registry
    }

    /// Register a task.
    pub fn register(&mut self, task: Task) {
        self.tasks.insert(task.id.clone(), task);
    }

    /// Get a task by ID.
    pub fn get(&self, id: &str) -> Option<&Task> {
        self.tasks.get(id)
    }

    /// Get all tasks.
    pub fn all(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values()
    }

    /// Filter tasks by source.
    pub fn by_source(&self, source: TaskSource) -> Vec<&Task> {
        self.tasks.values().filter(|t| t.source == source).collect()
    }

    /// Filter tasks by category.
    pub fn by_category(&self, category: TaskCategory) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Get task count.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        let mut registry = Self::with_standard_tasks();
        registry.register(counter_smoke_task());
        registry
    }
}

/// Minimal WasmRL counter smoke task.
fn counter_smoke_task() -> Task {
    Task::new("counter", "Counter Smoke")
        .with_description("Minimal counter environment smoke benchmark")
        .with_category(TaskCategory::Compute)
        .with_source(TaskSource::Custom)
        .with_input(br#"{"action": 1}"#.to_vec())
        .with_expected(br#"{"counter": 1}"#.to_vec())
        .with_complexity(1)
        .with_tag("smoke")
        .with_tag("wasmrl")
}

// ============================================================================
// SETA-ENV Standard Tasks
// ============================================================================

/// SETA-ENV file-counter task.
fn seta_file_counter() -> Task {
    Task::new("seta-file-counter", "File Counter")
        .with_description("Count files in a directory structure matching patterns")
        .with_category(TaskCategory::FileOps)
        .with_source(TaskSource::SetaEnv)
        .with_input(br#"{"path": "/tmp/test", "pattern": "*.txt"}"#.to_vec())
        .with_complexity(2)
        .with_tag("filesystem")
        .with_tag("baseline")
}

/// SETA-ENV json-transform task.
fn seta_json_transform() -> Task {
    Task::new("seta-json-transform", "JSON Transform")
        .with_description("Transform JSON documents according to rules")
        .with_category(TaskCategory::JsonTransform)
        .with_source(TaskSource::SetaEnv)
        .with_input(br#"{"data": {"a": 1, "b": 2}, "transform": "double"}"#.to_vec())
        .with_expected(br#"{"a": 2, "b": 4}"#.to_vec())
        .with_complexity(3)
        .with_tag("json")
        .with_tag("transform")
}

/// SETA-ENV code-lint task.
fn seta_code_lint() -> Task {
    Task::new("seta-code-lint", "Code Lint")
        .with_description("Lint code and report issues")
        .with_category(TaskCategory::CodeLint)
        .with_source(TaskSource::SetaEnv)
        .with_input(br#"{"code": "def foo():\n  pass", "language": "python"}"#.to_vec())
        .with_complexity(5)
        .with_tag("lint")
        .with_tag("python")
}

/// SETA-ENV unit-test task.
fn seta_unit_test() -> Task {
    Task::new("seta-unit-test", "Unit Test Runner")
        .with_description("Run unit tests and collect results")
        .with_category(TaskCategory::UnitTest)
        .with_source(TaskSource::SetaEnv)
        .with_input(br#"{"test_file": "test_example.py", "timeout": 30}"#.to_vec())
        .with_complexity(6)
        .with_tag("testing")
        .with_tag("python")
}

/// SETA-ENV api-mock task.
fn seta_api_mock() -> Task {
    Task::new("seta-api-mock", "API Mock Server")
        .with_description("Create and serve mock API endpoints")
        .with_category(TaskCategory::ApiMock)
        .with_source(TaskSource::SetaEnv)
        .with_input(br#"{"routes": [{"path": "/api/test", "response": {"ok": true}}]}"#.to_vec())
        .with_complexity(4)
        .with_tag("http")
        .with_tag("mock")
}

// ============================================================================
// CRAB Standard Tasks
// ============================================================================

/// CRAB file-ops task.
fn crab_file_ops() -> Task {
    Task::new("crab-file-ops", "File Operations")
        .with_description("Perform file operations across platform")
        .with_category(TaskCategory::FileOps)
        .with_source(TaskSource::Crab)
        .with_input(br#"{"operation": "copy", "src": "/tmp/a.txt", "dst": "/tmp/b.txt"}"#.to_vec())
        .with_complexity(2)
        .with_tag("filesystem")
        .with_tag("cross-platform")
}

/// CRAB compute task.
fn crab_compute() -> Task {
    Task::new("crab-compute", "Compute Task")
        .with_description("CPU-bound computation task")
        .with_category(TaskCategory::Compute)
        .with_source(TaskSource::Crab)
        .with_input(br#"{"operation": "fibonacci", "n": 30}"#.to_vec())
        .with_expected(br#"{"result": 832040}"#.to_vec())
        .with_complexity(4)
        .with_tag("compute")
        .with_tag("cpu-bound")
}

/// CRAB web-fetch task.
fn crab_web_fetch() -> Task {
    Task::new("crab-web-fetch", "Web Fetch")
        .with_description("Fetch data from web endpoints")
        .with_category(TaskCategory::WebFetch)
        .with_source(TaskSource::Crab)
        .with_input(br#"{"url": "https://httpbin.org/get", "method": "GET"}"#.to_vec())
        .with_complexity(3)
        .with_tag("http")
        .with_tag("network")
}

/// Task verifier for validating outputs.
pub struct TaskVerifier;

impl TaskVerifier {
    /// Verify task output.
    pub fn verify(task: &Task, output: &[u8]) -> Result<bool> {
        match &task.verification {
            VerificationMethod::OutputMatch => {
                if let Some(expected) = &task.expected_output {
                    Ok(output == expected.as_slice())
                } else {
                    // No expected output, pass if any output
                    Ok(!output.is_empty())
                }
            }
            VerificationMethod::JsonSchema(schema) => {
                // Parse output as JSON
                let _value: serde_json::Value = serde_json::from_slice(output)
                    .map_err(|e| ComparisonError::execution(format!("Invalid JSON: {}", e)))?;

                // TODO: Actual schema validation
                let _ = schema;
                Ok(true)
            }
            VerificationMethod::Regex(pattern) => {
                let re = regex::Regex::new(pattern)
                    .map_err(|e| ComparisonError::config(format!("Invalid regex: {}", e)))?;
                let output_str = std::str::from_utf8(output)
                    .map_err(|e| ComparisonError::execution(format!("Invalid UTF-8: {}", e)))?;
                Ok(re.is_match(output_str))
            }
            VerificationMethod::Custom(_name) => {
                // TODO: Lookup custom verifier
                Ok(true)
            }
            VerificationMethod::None => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_builder() {
        let task = Task::new("test-1", "Test Task")
            .with_description("A test task")
            .with_category(TaskCategory::Compute)
            .with_source(TaskSource::Custom)
            .with_complexity(5)
            .with_tag("test");

        assert_eq!(task.id, "test-1");
        assert_eq!(task.name, "Test Task");
        assert_eq!(task.category, TaskCategory::Compute);
        assert_eq!(task.complexity, 5);
        assert!(task.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_task_registry_standard() {
        let registry = TaskRegistry::with_standard_tasks();
        assert_eq!(registry.len(), 8); // 5 SETA-ENV + 3 CRAB
    }

    #[test]
    fn test_registry_by_source() {
        let registry = TaskRegistry::with_standard_tasks();

        let seta_tasks = registry.by_source(TaskSource::SetaEnv);
        assert_eq!(seta_tasks.len(), 5);

        let crab_tasks = registry.by_source(TaskSource::Crab);
        assert_eq!(crab_tasks.len(), 3);
    }

    #[test]
    fn test_registry_by_category() {
        let registry = TaskRegistry::with_standard_tasks();

        let file_ops = registry.by_category(TaskCategory::FileOps);
        assert_eq!(file_ops.len(), 2); // seta-file-counter + crab-file-ops
    }

    #[test]
    fn test_registry_get() {
        let registry = TaskRegistry::with_standard_tasks();

        let task = registry.get("seta-json-transform");
        assert!(task.is_some());
        assert_eq!(task.unwrap().source, TaskSource::SetaEnv);

        let missing = registry.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_task_category_display() {
        assert_eq!(TaskCategory::FileOps.to_string(), "file-ops");
        assert_eq!(TaskCategory::JsonTransform.to_string(), "json-transform");
    }

    #[test]
    fn test_task_source_display() {
        assert_eq!(TaskSource::SetaEnv.to_string(), "SETA-ENV");
        assert_eq!(TaskSource::Crab.to_string(), "CRAB");
    }

    #[test]
    fn test_verifier_output_match() {
        let task = Task::new("test", "Test").with_expected(b"hello".to_vec());

        assert!(TaskVerifier::verify(&task, b"hello").unwrap());
        assert!(!TaskVerifier::verify(&task, b"world").unwrap());
    }

    #[test]
    fn test_verifier_regex() {
        let mut task = Task::new("test", "Test");
        task.verification = VerificationMethod::Regex(r"\d+".to_string());

        assert!(TaskVerifier::verify(&task, b"123").unwrap());
        assert!(!TaskVerifier::verify(&task, b"abc").unwrap());
    }

    #[test]
    fn test_verifier_none() {
        let mut task = Task::new("test", "Test");
        task.verification = VerificationMethod::None;

        assert!(TaskVerifier::verify(&task, b"anything").unwrap());
    }

    #[test]
    fn test_complexity_max() {
        let task = Task::new("test", "Test").with_complexity(15);
        assert_eq!(task.complexity, 10); // Capped at 10
    }
}
