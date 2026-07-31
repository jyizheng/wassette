// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Error types for the MCP bridge.

use std::fmt;

use thiserror::Error;

/// Result type alias for bridge operations.
pub type BridgeResult<T> = Result<T, BridgeError>;

/// Errors that can occur in the MCP bridge.
#[derive(Error, Debug)]
pub enum BridgeError {
    /// Session not found.
    #[error("Session not found: {session_id}")]
    SessionNotFound {
        /// The session ID that was not found.
        session_id: String,
    },

    /// Maximum sessions exceeded.
    #[error("Maximum sessions exceeded: limit is {max_sessions}")]
    MaxSessionsExceeded {
        /// The maximum allowed sessions.
        max_sessions: usize,
    },

    /// Component load failure.
    #[error("Failed to load component: {message}")]
    ComponentLoadError {
        /// Error message.
        message: String,
    },

    /// Environment error during step/reset.
    #[error("Environment error: {message}")]
    EnvironmentError {
        /// Error message.
        message: String,
    },

    /// Invalid action format.
    #[error("Invalid action: {message}")]
    InvalidAction {
        /// Error message.
        message: String,
    },

    /// Tool call timeout.
    #[error("Tool call timed out after {timeout_ms}ms")]
    Timeout {
        /// Timeout in milliseconds.
        timeout_ms: u64,
    },

    /// Serialization error.
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message.
        message: String,
    },

    /// Runtime error from wasmrl-runtime.
    #[error("Runtime error: {message}")]
    RuntimeError {
        /// Error message.
        message: String,
    },

    /// Policy violation.
    #[error("Policy violation: {message}")]
    PolicyViolation {
        /// Error message.
        message: String,
    },

    /// Invalid tool name.
    #[error("Unknown tool: {tool_name}")]
    UnknownTool {
        /// The tool name that was not found.
        tool_name: String,
    },
}

impl BridgeError {
    /// Create a session not found error.
    pub fn session_not_found(session_id: impl Into<String>) -> Self {
        Self::SessionNotFound {
            session_id: session_id.into(),
        }
    }

    /// Create a max sessions exceeded error.
    pub fn max_sessions_exceeded(max_sessions: usize) -> Self {
        Self::MaxSessionsExceeded { max_sessions }
    }

    /// Create a component load error.
    pub fn component_load(message: impl Into<String>) -> Self {
        Self::ComponentLoadError {
            message: message.into(),
        }
    }

    /// Create an environment error.
    pub fn environment(message: impl Into<String>) -> Self {
        Self::EnvironmentError {
            message: message.into(),
        }
    }

    /// Create an invalid action error.
    pub fn invalid_action(message: impl Into<String>) -> Self {
        Self::InvalidAction {
            message: message.into(),
        }
    }

    /// Create a timeout error.
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Create a serialization error.
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    /// Create a runtime error.
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::RuntimeError {
            message: message.into(),
        }
    }

    /// Create a policy violation error.
    pub fn policy_violation(message: impl Into<String>) -> Self {
        Self::PolicyViolation {
            message: message.into(),
        }
    }

    /// Create an unknown tool error.
    pub fn unknown_tool(tool_name: impl Into<String>) -> Self {
        Self::UnknownTool {
            tool_name: tool_name.into(),
        }
    }

    /// Check if this is a recoverable error (session can continue).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::InvalidAction { .. } | Self::Timeout { .. } | Self::SerializationError { .. }
        )
    }

    /// Check if this is a fatal error (session should be closed).
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::ComponentLoadError { .. }
                | Self::RuntimeError { .. }
                | Self::PolicyViolation { .. }
        )
    }

    /// Get the error code for MCP error response.
    pub fn error_code(&self) -> i32 {
        match self {
            Self::SessionNotFound { .. } => -32001,
            Self::MaxSessionsExceeded { .. } => -32002,
            Self::ComponentLoadError { .. } => -32003,
            Self::EnvironmentError { .. } => -32004,
            Self::InvalidAction { .. } => -32005,
            Self::Timeout { .. } => -32006,
            Self::SerializationError { .. } => -32007,
            Self::RuntimeError { .. } => -32008,
            Self::PolicyViolation { .. } => -32009,
            Self::UnknownTool { .. } => -32601,
        }
    }
}

impl From<anyhow::Error> for BridgeError {
    fn from(err: anyhow::Error) -> Self {
        Self::RuntimeError {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for BridgeError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_session_not_found() {
        let err = BridgeError::session_not_found("session-abc");
        assert_eq!(err.to_string(), "Session not found: session-abc");
        assert_eq!(err.error_code(), -32001);
        assert!(!err.is_recoverable());
        assert!(!err.is_fatal());
    }

    #[test]
    fn test_error_max_sessions() {
        let err = BridgeError::max_sessions_exceeded(16);
        assert!(err.to_string().contains("16"));
        assert_eq!(err.error_code(), -32002);
    }

    #[test]
    fn test_error_recoverable() {
        assert!(BridgeError::invalid_action("bad format").is_recoverable());
        assert!(BridgeError::timeout(1000).is_recoverable());
        assert!(BridgeError::serialization("parse error").is_recoverable());
        assert!(!BridgeError::environment("env crashed").is_recoverable());
    }

    #[test]
    fn test_error_fatal() {
        assert!(BridgeError::component_load("file not found").is_fatal());
        assert!(BridgeError::runtime("trap").is_fatal());
        assert!(BridgeError::policy_violation("denied").is_fatal());
        assert!(!BridgeError::invalid_action("bad").is_fatal());
    }

    #[test]
    fn test_error_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("Something went wrong");
        let bridge_err: BridgeError = anyhow_err.into();
        assert!(matches!(bridge_err, BridgeError::RuntimeError { .. }));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_result: Result<(), serde_json::Error> =
            serde_json::from_str::<serde_json::Value>("not json {{").map(|_| ());
        let bridge_err: BridgeError = json_result.unwrap_err().into();
        assert!(matches!(bridge_err, BridgeError::SerializationError { .. }));
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(BridgeError::session_not_found("x").error_code(), -32001);
        assert_eq!(BridgeError::max_sessions_exceeded(10).error_code(), -32002);
        assert_eq!(BridgeError::component_load("x").error_code(), -32003);
        assert_eq!(BridgeError::environment("x").error_code(), -32004);
        assert_eq!(BridgeError::invalid_action("x").error_code(), -32005);
        assert_eq!(BridgeError::timeout(100).error_code(), -32006);
        assert_eq!(BridgeError::serialization("x").error_code(), -32007);
        assert_eq!(BridgeError::runtime("x").error_code(), -32008);
        assert_eq!(BridgeError::policy_violation("x").error_code(), -32009);
        assert_eq!(BridgeError::unknown_tool("x").error_code(), -32601);
    }
}
