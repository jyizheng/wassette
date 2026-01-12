// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Policy error types.

use thiserror::Error;

/// Policy-related errors.
#[derive(Debug, Error)]
pub enum PolicyError {
    /// Error parsing policy from TOML or JSON.
    #[error("Policy parse error: {0}")]
    ParseError(String),

    /// Error serializing policy.
    #[error("Policy serialization error: {0}")]
    SerializeError(String),

    /// Policy validation failed.
    #[error("Policy validation error: {0}")]
    ValidationError(String),

    /// I/O error reading policy file.
    #[error("Policy I/O error: {0}")]
    IoError(String),

    /// Budget exceeded error.
    #[error("Budget exceeded: {budget_type} (limit: {limit}, used: {used})")]
    BudgetExceeded {
        /// Type of budget exceeded.
        budget_type: String,
        /// Budget limit.
        limit: u64,
        /// Amount used.
        used: u64,
    },

    /// Capability denied error.
    #[error("Capability denied: {capability}")]
    CapabilityDenied {
        /// Denied capability name.
        capability: String,
    },

    /// Timeout exceeded error.
    #[error("Timeout exceeded: {operation} took {elapsed_ms}ms (limit: {limit_ms}ms)")]
    TimeoutExceeded {
        /// Operation that timed out.
        operation: String,
        /// Time elapsed in milliseconds.
        elapsed_ms: u64,
        /// Timeout limit in milliseconds.
        limit_ms: u64,
    },

    /// Memory limit exceeded.
    #[error("Memory limit exceeded: {used_mb}MB > {limit_mb}MB")]
    MemoryExceeded {
        /// Memory used in MB.
        used_mb: u32,
        /// Memory limit in MB.
        limit_mb: u32,
    },

    /// Enforcement error.
    #[error("Enforcement error: {0}")]
    EnforcementError(String),
}

impl PolicyError {
    /// Create a budget exceeded error.
    pub fn budget_exceeded(budget_type: &str, limit: u64, used: u64) -> Self {
        Self::BudgetExceeded {
            budget_type: budget_type.to_string(),
            limit,
            used,
        }
    }

    /// Create a capability denied error.
    pub fn capability_denied(capability: &str) -> Self {
        Self::CapabilityDenied {
            capability: capability.to_string(),
        }
    }

    /// Create a timeout exceeded error.
    pub fn timeout_exceeded(operation: &str, elapsed_ms: u64, limit_ms: u64) -> Self {
        Self::TimeoutExceeded {
            operation: operation.to_string(),
            elapsed_ms,
            limit_ms,
        }
    }

    /// Create a memory exceeded error.
    pub fn memory_exceeded(used_mb: u32, limit_mb: u32) -> Self {
        Self::MemoryExceeded { used_mb, limit_mb }
    }

    /// Check if this is a budget-related error.
    pub fn is_budget_error(&self) -> bool {
        matches!(
            self,
            Self::BudgetExceeded { .. }
                | Self::TimeoutExceeded { .. }
                | Self::MemoryExceeded { .. }
        )
    }

    /// Check if this is a capability-related error.
    pub fn is_capability_error(&self) -> bool {
        matches!(self, Self::CapabilityDenied { .. })
    }
}

/// Result type for policy operations.
pub type PolicyResult<T> = Result<T, PolicyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_exceeded_error() {
        let err = PolicyError::budget_exceeded("fuel", 1_000_000, 1_500_000);
        assert!(err.is_budget_error());
        assert!(!err.is_capability_error());
        assert!(err.to_string().contains("fuel"));
    }

    #[test]
    fn test_capability_denied_error() {
        let err = PolicyError::capability_denied("network");
        assert!(err.is_capability_error());
        assert!(!err.is_budget_error());
        assert!(err.to_string().contains("network"));
    }

    #[test]
    fn test_timeout_exceeded_error() {
        let err = PolicyError::timeout_exceeded("step", 150, 100);
        assert!(err.is_budget_error());
        assert!(err.to_string().contains("150ms"));
        assert!(err.to_string().contains("100ms"));
    }

    #[test]
    fn test_memory_exceeded_error() {
        let err = PolicyError::memory_exceeded(512, 256);
        assert!(err.is_budget_error());
        assert!(err.to_string().contains("512MB"));
        assert!(err.to_string().contains("256MB"));
    }

    #[test]
    fn test_parse_error() {
        let err = PolicyError::ParseError("invalid syntax".to_string());
        assert!(err.to_string().contains("invalid syntax"));
    }

    #[test]
    fn test_validation_error() {
        let err = PolicyError::ValidationError("max_mb must be > 0".to_string());
        assert!(err.to_string().contains("max_mb"));
    }
}
