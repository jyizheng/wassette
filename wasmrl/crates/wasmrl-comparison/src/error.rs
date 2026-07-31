// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Error types for comparison framework.

use std::fmt;

use thiserror::Error;

/// Result type alias for comparison operations.
pub type ComparisonResult<T> = Result<T, ComparisonError>;

/// Errors that can occur in comparison benchmarks.
#[derive(Error, Debug)]
pub enum ComparisonError {
    /// Unknown backend.
    #[error("Unknown backend: {name}")]
    UnknownBackend {
        /// Backend name.
        name: String,
    },

    /// Backend not available.
    #[error("Backend not available: {backend} (reason: {reason})")]
    BackendNotAvailable {
        /// Backend name.
        backend: String,
        /// Reason.
        reason: String,
    },

    /// Backend not initialized.
    #[error("Backend not initialized: {backend}")]
    NotInitialized {
        /// Backend name.
        backend: String,
    },

    /// Task not found.
    #[error("Task not found: {name}")]
    TaskNotFound {
        /// Task name.
        name: String,
    },

    /// Configuration error.
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message.
        message: String,
    },

    /// Verification failed.
    #[error("Verification failed: expected {expected}, got {actual}")]
    VerificationFailed {
        /// Expected value.
        expected: String,
        /// Actual value.
        actual: String,
    },

    /// Execution error.
    #[error("Execution error: {message}")]
    ExecutionError {
        /// Error message.
        message: String,
    },

    /// Timeout.
    #[error("Timeout after {timeout_ms}ms")]
    Timeout {
        /// Timeout in milliseconds.
        timeout_ms: u64,
    },

    /// IO error.
    #[error("IO error: {message}")]
    IoError {
        /// Error message.
        message: String,
    },

    /// Report generation error.
    #[error("Report error: {message}")]
    ReportError {
        /// Error message.
        message: String,
    },
}

impl ComparisonError {
    /// Create an unknown backend error.
    pub fn unknown_backend(name: impl Into<String>) -> Self {
        Self::UnknownBackend { name: name.into() }
    }

    /// Create a backend not available error.
    pub fn backend_not_available(backend: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::BackendNotAvailable {
            backend: backend.into(),
            reason: reason.into(),
        }
    }

    /// Create a not initialized error.
    pub fn not_initialized(backend: impl Into<String>) -> Self {
        Self::NotInitialized {
            backend: backend.into(),
        }
    }

    /// Create a task not found error.
    pub fn task_not_found(name: impl Into<String>) -> Self {
        Self::TaskNotFound { name: name.into() }
    }

    /// Create a configuration error.
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Create a verification failed error.
    pub fn verification_failed(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::VerificationFailed {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an execution error.
    pub fn execution(message: impl Into<String>) -> Self {
        Self::ExecutionError {
            message: message.into(),
        }
    }

    /// Create a timeout error.
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Create an IO error.
    pub fn io(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    /// Create a report error.
    pub fn report(message: impl Into<String>) -> Self {
        Self::ReportError {
            message: message.into(),
        }
    }

    /// Check if this is a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::Timeout { .. } | Self::VerificationFailed { .. })
    }
}

impl From<std::io::Error> for ComparisonError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for ComparisonError {
    fn from(err: serde_json::Error) -> Self {
        Self::ConfigError {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_unknown_backend() {
        let err = ComparisonError::unknown_backend("invalid");
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn test_error_not_initialized() {
        let err = ComparisonError::not_initialized("WasmBackend");
        assert!(err.to_string().contains("WasmBackend"));
    }

    #[test]
    fn test_error_verification_failed() {
        let err = ComparisonError::verification_failed("10", "20");
        assert!(err.to_string().contains("expected"));
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("20"));
    }

    #[test]
    fn test_error_recoverable() {
        assert!(ComparisonError::timeout(1000).is_recoverable());
        assert!(ComparisonError::verification_failed("a", "b").is_recoverable());
        assert!(!ComparisonError::config("bad config").is_recoverable());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: ComparisonError = io_err.into();
        assert!(matches!(err, ComparisonError::IoError { .. }));
    }
}
