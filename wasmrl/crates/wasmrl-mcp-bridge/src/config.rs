// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Configuration for the MCP bridge.

use serde::{Deserialize, Serialize};

/// Configuration for the WasmRL MCP bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBridgeConfig {
    /// Path to the WebAssembly component file.
    pub component_path: String,

    /// Maximum number of concurrent sessions.
    pub max_sessions: usize,

    /// Timeout for tool calls in milliseconds.
    pub timeout_ms: u64,

    /// Whether to collect overhead metrics.
    pub collect_metrics: bool,

    /// Optional policy configuration path.
    pub policy_path: Option<String>,

    /// Environment name (used in tool naming).
    pub env_name: Option<String>,
}

impl McpBridgeConfig {
    /// Create a new bridge configuration with a component path.
    pub fn new(component_path: impl Into<String>) -> Self {
        Self {
            component_path: component_path.into(),
            max_sessions: 16,
            timeout_ms: 30_000,
            collect_metrics: true,
            policy_path: None,
            env_name: None,
        }
    }

    /// Set the maximum number of concurrent sessions.
    pub fn with_max_sessions(mut self, max_sessions: usize) -> Self {
        self.max_sessions = max_sessions;
        self
    }

    /// Set the timeout for tool calls.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Enable or disable metrics collection.
    pub fn with_metrics(mut self, collect_metrics: bool) -> Self {
        self.collect_metrics = collect_metrics;
        self
    }

    /// Set the policy configuration path.
    pub fn with_policy(mut self, policy_path: impl Into<String>) -> Self {
        self.policy_path = Some(policy_path.into());
        self
    }

    /// Set the environment name for tool naming.
    pub fn with_env_name(mut self, env_name: impl Into<String>) -> Self {
        self.env_name = Some(env_name.into());
        self
    }

    /// Get the environment name, defaulting to component filename.
    pub fn get_env_name(&self) -> String {
        self.env_name.clone().unwrap_or_else(|| {
            std::path::Path::new(&self.component_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("env")
                .to_string()
        })
    }
}

impl Default for McpBridgeConfig {
    fn default() -> Self {
        Self {
            component_path: String::new(),
            max_sessions: 16,
            timeout_ms: 30_000,
            collect_metrics: true,
            policy_path: None,
            env_name: None,
        }
    }
}

/// Configuration for an individual environment session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Optional seed for environment reset.
    pub seed: Option<u64>,

    /// Whether to auto-reset on episode termination.
    pub auto_reset: bool,

    /// Maximum steps before forced reset.
    pub max_steps: Option<u64>,

    /// Whether to record trajectory data.
    pub record_trajectory: bool,

    /// Custom environment configuration as JSON.
    pub env_config: Option<serde_json::Value>,
}

impl SessionConfig {
    /// Create a new session configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the seed for environment reset.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable or disable auto-reset.
    pub fn with_auto_reset(mut self, auto_reset: bool) -> Self {
        self.auto_reset = auto_reset;
        self
    }

    /// Set the maximum number of steps.
    pub fn with_max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Enable trajectory recording.
    pub fn with_recording(mut self, record: bool) -> Self {
        self.record_trajectory = record;
        self
    }

    /// Set custom environment configuration.
    pub fn with_env_config(mut self, config: serde_json::Value) -> Self {
        self.env_config = Some(config);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config_builder() {
        let config = McpBridgeConfig::new("env.wasm")
            .with_max_sessions(32)
            .with_timeout_ms(60_000)
            .with_metrics(false)
            .with_policy("policy.toml")
            .with_env_name("counter");

        assert_eq!(config.component_path, "env.wasm");
        assert_eq!(config.max_sessions, 32);
        assert_eq!(config.timeout_ms, 60_000);
        assert!(!config.collect_metrics);
        assert_eq!(config.policy_path, Some("policy.toml".to_string()));
        assert_eq!(config.get_env_name(), "counter");
    }

    #[test]
    fn test_bridge_config_env_name_from_path() {
        let config = McpBridgeConfig::new("/path/to/counter_env.wasm");
        assert_eq!(config.get_env_name(), "counter_env");
    }

    #[test]
    fn test_session_config_builder() {
        let config = SessionConfig::new()
            .with_seed(12345)
            .with_auto_reset(true)
            .with_max_steps(1000)
            .with_recording(true)
            .with_env_config(serde_json::json!({"size": 10}));

        assert_eq!(config.seed, Some(12345));
        assert!(config.auto_reset);
        assert_eq!(config.max_steps, Some(1000));
        assert!(config.record_trajectory);
        assert!(config.env_config.is_some());
    }

    #[test]
    fn test_config_serialization() {
        let config = McpBridgeConfig::new("test.wasm").with_max_sessions(8);

        let json = serde_json::to_string(&config).unwrap();
        let parsed: McpBridgeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.component_path, "test.wasm");
        assert_eq!(parsed.max_sessions, 8);
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert!(config.seed.is_none());
        assert!(!config.auto_reset);
        assert!(config.max_steps.is_none());
        assert!(!config.record_trajectory);
    }
}
