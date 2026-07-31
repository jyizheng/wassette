// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Error types for Python bindings.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// WasmRL Python error type.
#[derive(Debug, Error)]
pub enum PyWasmRLError {
    /// Environment not initialized.
    #[error("Environment not initialized. Call reset() first.")]
    NotInitialized,

    /// Invalid action shape or type.
    #[error("Invalid action: {0}")]
    InvalidAction(String),

    /// Invalid observation shape or type.
    #[error("Invalid observation: {0}")]
    InvalidObservation(String),

    /// Component loading error.
    #[error("Failed to load component: {0}")]
    ComponentLoadError(String),

    /// Runtime error.
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Policy error.
    #[error("Policy error: {0}")]
    PolicyError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Environment closed.
    #[error("Environment has been closed")]
    EnvClosed,

    /// Shape mismatch.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Type error.
    #[error("Type error: {0}")]
    TypeError(String),
}

impl From<PyWasmRLError> for PyErr {
    fn from(err: PyWasmRLError) -> Self {
        match err {
            PyWasmRLError::NotInitialized => PyRuntimeError::new_err(err.to_string()),
            PyWasmRLError::InvalidAction(_) => PyValueError::new_err(err.to_string()),
            PyWasmRLError::InvalidObservation(_) => PyValueError::new_err(err.to_string()),
            PyWasmRLError::ComponentLoadError(_) => PyRuntimeError::new_err(err.to_string()),
            PyWasmRLError::RuntimeError(_) => PyRuntimeError::new_err(err.to_string()),
            PyWasmRLError::PolicyError(_) => PyRuntimeError::new_err(err.to_string()),
            PyWasmRLError::ConfigError(_) => PyValueError::new_err(err.to_string()),
            PyWasmRLError::EnvClosed => PyRuntimeError::new_err(err.to_string()),
            PyWasmRLError::ShapeMismatch { .. } => PyValueError::new_err(err.to_string()),
            PyWasmRLError::TypeError(_) => PyValueError::new_err(err.to_string()),
        }
    }
}

impl From<anyhow::Error> for PyWasmRLError {
    fn from(err: anyhow::Error) -> Self {
        PyWasmRLError::RuntimeError(err.to_string())
    }
}

impl From<serde_json::Error> for PyWasmRLError {
    fn from(err: serde_json::Error) -> Self {
        PyWasmRLError::ConfigError(err.to_string())
    }
}

/// Result type for Python operations.
pub type PyWasmRLResult<T> = Result<T, PyWasmRLError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PyWasmRLError::NotInitialized;
        assert!(err.to_string().contains("not initialized"));
    }

    #[test]
    fn test_error_variants() {
        let errors = vec![
            PyWasmRLError::NotInitialized,
            PyWasmRLError::InvalidAction("bad action".to_string()),
            PyWasmRLError::ComponentLoadError("not found".to_string()),
            PyWasmRLError::RuntimeError("crash".to_string()),
            PyWasmRLError::EnvClosed,
            PyWasmRLError::ShapeMismatch {
                expected: "(4,)".to_string(),
                got: "(8,)".to_string(),
            },
        ];

        for err in errors {
            // All should produce non-empty messages
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("test error");
        let py_err: PyWasmRLError = anyhow_err.into();
        assert!(matches!(py_err, PyWasmRLError::RuntimeError(_)));
    }
}
