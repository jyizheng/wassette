// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Error types for the WasmRL runtime.

use thiserror::Error;

/// Errors that can occur in the WasmRL runtime.
#[derive(Error, Debug)]
pub enum RuntimeError {
    /// Error loading a component.
    #[error("Failed to load component: {0}")]
    ComponentLoad(String),

    /// Error instantiating a component.
    #[error("Failed to instantiate component: {0}")]
    Instantiation(String),

    /// Error executing a function.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Instance not found.
    #[error("Instance not found: {0}")]
    InstanceNotFound(u64),

    /// Instance pool exhausted.
    #[error("Instance pool exhausted, max capacity: {0}")]
    PoolExhausted(usize),

    /// Instance crashed or trapped.
    #[error("Instance {instance_id} trapped: {reason}")]
    InstanceTrapped {
        /// The instance that trapped.
        instance_id: u64,
        /// The reason for the trap.
        reason: String,
    },

    /// Tensor shape/dtype mismatch.
    #[error("Tensor mismatch: expected {expected}, got {actual}")]
    TensorMismatch {
        /// Expected tensor specification.
        expected: String,
        /// Actual tensor specification.
        actual: String,
    },

    /// Batch size mismatch.
    #[error("Batch size mismatch: expected {expected}, got {actual}")]
    BatchSizeMismatch {
        /// Expected batch size.
        expected: usize,
        /// Actual batch size.
        actual: usize,
    },

    /// Timeout exceeded.
    #[error("Timeout exceeded: {operation} took {elapsed_ms}ms, limit was {limit_ms}ms")]
    Timeout {
        /// The operation that timed out.
        operation: String,
        /// Elapsed time in milliseconds.
        elapsed_ms: u64,
        /// Time limit in milliseconds.
        limit_ms: u64,
    },

    /// Fuel exhausted.
    #[error("Fuel exhausted during {operation}")]
    FuelExhausted {
        /// The operation that exhausted fuel.
        operation: String,
    },

    /// Memory limit exceeded.
    #[error("Memory limit exceeded: attempted {attempted_mb}MB, limit is {limit_mb}MB")]
    MemoryLimitExceeded {
        /// Attempted memory allocation in MB.
        attempted_mb: usize,
        /// Memory limit in MB.
        limit_mb: usize,
    },

    /// Configuration error.
    #[error("Invalid configuration: {0}")]
    Configuration(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl RuntimeError {
    /// Create a new component load error.
    pub fn component_load(msg: impl Into<String>) -> Self {
        Self::ComponentLoad(msg.into())
    }

    /// Create a new instantiation error.
    pub fn instantiation(msg: impl Into<String>) -> Self {
        Self::Instantiation(msg.into())
    }

    /// Create a new execution error.
    pub fn execution(msg: impl Into<String>) -> Self {
        Self::Execution(msg.into())
    }

    /// Create a new instance not found error.
    pub fn instance_not_found(id: u64) -> Self {
        Self::InstanceNotFound(id)
    }

    /// Create a new pool exhausted error.
    pub fn pool_exhausted(capacity: usize) -> Self {
        Self::PoolExhausted(capacity)
    }

    /// Create a new instance trapped error.
    pub fn instance_trapped(instance_id: u64, reason: impl Into<String>) -> Self {
        Self::InstanceTrapped {
            instance_id,
            reason: reason.into(),
        }
    }

    /// Create a new timeout error.
    pub fn timeout(operation: impl Into<String>, elapsed_ms: u64, limit_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            elapsed_ms,
            limit_ms,
        }
    }

    /// Create a new fuel exhausted error.
    pub fn fuel_exhausted(operation: impl Into<String>) -> Self {
        Self::FuelExhausted {
            operation: operation.into(),
        }
    }
}

/// Result type for runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RuntimeError::component_load("missing export");
        assert_eq!(err.to_string(), "Failed to load component: missing export");

        let err = RuntimeError::instance_not_found(42);
        assert_eq!(err.to_string(), "Instance not found: 42");

        let err = RuntimeError::pool_exhausted(256);
        assert_eq!(
            err.to_string(),
            "Instance pool exhausted, max capacity: 256"
        );

        let err = RuntimeError::instance_trapped(1, "infinite loop");
        assert_eq!(err.to_string(), "Instance 1 trapped: infinite loop");
    }

    #[test]
    fn test_timeout_error() {
        let err = RuntimeError::timeout("step", 1500, 1000);
        assert_eq!(
            err.to_string(),
            "Timeout exceeded: step took 1500ms, limit was 1000ms"
        );
    }

    #[test]
    fn test_fuel_exhausted_error() {
        let err = RuntimeError::fuel_exhausted("step_batch");
        assert_eq!(err.to_string(), "Fuel exhausted during step_batch");
    }

    #[test]
    fn test_batch_size_mismatch() {
        let err = RuntimeError::BatchSizeMismatch {
            expected: 10,
            actual: 8,
        };
        assert_eq!(err.to_string(), "Batch size mismatch: expected 10, got 8");
    }

    #[test]
    fn test_memory_limit_exceeded() {
        let err = RuntimeError::MemoryLimitExceeded {
            attempted_mb: 1024,
            limit_mb: 512,
        };
        assert_eq!(
            err.to_string(),
            "Memory limit exceeded: attempted 1024MB, limit is 512MB"
        );
    }
}
