// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL MCP Bridge - Exposes RL environments as MCP tools.
//!
//! This crate provides a bridge between WasmRL's high-performance in-process
//! runtime and the MCP (Model Context Protocol) control plane. It allows
//! RL environments to be used as MCP tools for orchestration and debugging.
//!
//! # Overview
//!
//! The bridge exposes each WasmRL environment as a set of MCP tools:
//!
//! - `env_create`: Create a new environment instance
//! - `env_reset`: Reset an environment with a seed
//! - `env_step`: Execute a step with an action
//! - `env_close`: Close and cleanup an environment
//! - `env_info`: Get environment metadata (obs/action spaces)
//!
//! # Design Philosophy
//!
//! MCP is the **control plane** - suitable for:
//! - Loading/unloading environments
//! - Debugging and inspection
//! - Low-frequency orchestration
//!
//! WasmRL in-process is the **data plane** - suitable for:
//! - High-throughput RL rollouts
//! - Batched stepping (thousands of envs)
//! - Training loops
//!
//! This bridge demonstrates the overhead gap and provides a clear narrative
//! for when to use each approach.
//!
//! # Quick Start
//!
//! ```ignore
//! use wasmrl_mcp_bridge::{McpBridgeConfig, EnvMcpBridge};
//!
//! // Create bridge with component path
//! let config = McpBridgeConfig::new("path/to/env.wasm");
//! let bridge = EnvMcpBridge::new(config)?;
//!
//! // Get tools to expose via MCP
//! let tools = bridge.get_tools();
//!
//! // Handle tool calls
//! let result = bridge.call_tool("env_reset", json!({"seed": 42}))?;
//! ```
//!
//! # Benchmarking
//!
//! The bridge includes timing instrumentation to measure:
//! - RPC/serialization overhead
//! - Runtime execution time
//! - Environment compute time
//!
//! This enables paper-quality comparisons between MCP and in-proc approaches.

#![warn(missing_docs)]

mod config;
mod error;
mod overhead;
mod session;
mod tools;

// Re-export main types
pub use config::{McpBridgeConfig, SessionConfig};
pub use error::{BridgeError, BridgeResult};
pub use overhead::{OverheadMetrics, TimingBreakdown};
pub use session::{EnvSession, SessionId, SessionManager, SessionState};
pub use tools::{EnvMcpBridge, McpTool, ToolResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config_new() {
        let config = McpBridgeConfig::new("path/to/env.wasm");
        assert_eq!(config.component_path, "path/to/env.wasm");
    }

    #[test]
    fn test_bridge_config_default() {
        let config = McpBridgeConfig::default();
        assert!(config.component_path.is_empty());
        assert_eq!(config.max_sessions, 16);
    }

    #[test]
    fn test_session_config() {
        let config = SessionConfig::new().with_seed(42).with_auto_reset(true);
        assert_eq!(config.seed, Some(42));
        assert!(config.auto_reset);
    }

    #[test]
    fn test_session_id() {
        let id1 = SessionId::new();
        let id2 = SessionId::new();
        assert_ne!(id1, id2);
        assert!(format!("{}", id1).starts_with("session-"));
    }

    #[test]
    fn test_overhead_metrics() {
        let mut metrics = OverheadMetrics::new();
        assert_eq!(metrics.total_calls, 0);
        metrics.record_call(
            std::time::Duration::from_micros(10),
            std::time::Duration::from_micros(100),
            std::time::Duration::from_micros(80),
        );
        assert_eq!(metrics.total_calls, 1);
    }

    #[test]
    fn test_timing_breakdown() {
        let breakdown = TimingBreakdown {
            rpc_serialization_us: 10,
            runtime_overhead_us: 5,
            env_compute_us: 85,
            total_us: 100,
        };
        assert_eq!(breakdown.overhead_ratio(), 0.15);
    }

    #[test]
    fn test_bridge_error_display() {
        let err = BridgeError::session_not_found("session-123");
        assert!(err.to_string().contains("session-123"));
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success(serde_json::json!({"obs": [1, 2, 3]}));
        assert!(result.is_ok);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(!result.is_ok);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }
}
