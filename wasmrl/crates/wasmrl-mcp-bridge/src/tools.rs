// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! MCP tool definitions and bridge implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, info, instrument, warn};

use crate::config::{McpBridgeConfig, SessionConfig};
use crate::error::{BridgeError, BridgeResult};
use crate::overhead::{OverheadMetrics, TimingBreakdown};
use crate::session::{SessionId, SessionManager, SessionState};

/// Result of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the call succeeded.
    pub is_ok: bool,

    /// The result data (if successful).
    pub data: Option<Value>,

    /// Error message (if failed).
    pub error: Option<String>,

    /// Timing breakdown for overhead analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingBreakdown>,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(data: Value) -> Self {
        Self {
            is_ok: true,
            data: Some(data),
            error: None,
            timing: None,
        }
    }

    /// Create a successful result with timing.
    pub fn success_with_timing(data: Value, timing: TimingBreakdown) -> Self {
        Self {
            is_ok: true,
            data: Some(data),
            error: None,
            timing: Some(timing),
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            is_ok: false,
            data: None,
            error: Some(message.into()),
            timing: None,
        }
    }

    /// Create an error result from a BridgeError.
    pub fn from_error(err: BridgeError) -> Self {
        Self::error(err.to_string())
    }
}

/// Definition of an MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Tool name.
    pub name: String,

    /// Human-readable description.
    pub description: String,

    /// JSON schema for input parameters.
    pub input_schema: Value,
}

impl McpTool {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    /// Set the input schema.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.input_schema = schema;
        self
    }
}

/// The main MCP bridge for WasmRL environments.
#[derive(Debug)]
pub struct EnvMcpBridge {
    /// Bridge configuration.
    config: McpBridgeConfig,

    /// Session manager.
    sessions: SessionManager,

    /// Overhead metrics.
    metrics: OverheadMetrics,

    /// Tool definitions.
    tools: Vec<McpTool>,
}

impl EnvMcpBridge {
    /// Create a new MCP bridge.
    pub fn new(config: McpBridgeConfig) -> BridgeResult<Self> {
        let env_name = config.get_env_name();
        let tools = Self::create_tool_definitions(&env_name);

        Ok(Self {
            sessions: SessionManager::new(config.max_sessions),
            metrics: OverheadMetrics::new(),
            config,
            tools,
        })
    }

    /// Create tool definitions for an environment.
    fn create_tool_definitions(env_name: &str) -> Vec<McpTool> {
        vec![
            McpTool::new(
                format!("{}_create", env_name),
                format!("Create a new {} environment session", env_name),
            )
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility"
                    },
                    "auto_reset": {
                        "type": "boolean",
                        "description": "Automatically reset on episode end"
                    },
                    "config": {
                        "type": "object",
                        "description": "Environment-specific configuration"
                    }
                },
                "required": []
            })),
            McpTool::new(
                format!("{}_reset", env_name),
                format!("Reset a {} environment to initial state", env_name),
            )
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to reset"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for this reset"
                    }
                },
                "required": ["session_id"]
            })),
            McpTool::new(
                format!("{}_step", env_name),
                format!("Execute one step in {} environment", env_name),
            )
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to step"
                    },
                    "action": {
                        "description": "Action to execute (format depends on environment)"
                    }
                },
                "required": ["session_id", "action"]
            })),
            McpTool::new(
                format!("{}_close", env_name),
                format!("Close a {} environment session", env_name),
            )
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to close"
                    }
                },
                "required": ["session_id"]
            })),
            McpTool::new(
                format!("{}_info", env_name),
                format!("Get information about {} environment or session", env_name),
            )
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID. If omitted, returns env info."
                    }
                },
                "required": []
            })),
            McpTool::new(
                format!("{}_list", env_name),
                format!("List all active {} sessions", env_name),
            )
            .with_schema(json!({
                "type": "object",
                "properties": {},
                "required": []
            })),
            McpTool::new(
                format!("{}_metrics", env_name),
                "Get overhead metrics for MCP bridge",
            )
            .with_schema(json!({
                "type": "object",
                "properties": {},
                "required": []
            })),
        ]
    }

    /// Get the list of tool definitions.
    pub fn get_tools(&self) -> &[McpTool] {
        &self.tools
    }

    /// Get the bridge configuration.
    pub fn config(&self) -> &McpBridgeConfig {
        &self.config
    }

    /// Get current overhead metrics summary.
    pub fn get_metrics(&self) -> crate::overhead::OverheadSummary {
        self.metrics.summary()
    }

    /// Call a tool by name with arguments.
    #[instrument(skip(self, args), fields(tool_name = %tool_name))]
    pub fn call_tool(&mut self, tool_name: &str, args: Value) -> ToolResult {
        let start = Instant::now();
        let rpc_start = Instant::now();

        // Parse the tool name to extract operation
        let env_name = self.config.get_env_name();
        let operation = if let Some(op) = tool_name.strip_prefix(&format!("{}_", env_name)) {
            op
        } else {
            return ToolResult::from_error(BridgeError::unknown_tool(tool_name));
        };

        // RPC overhead (argument parsing)
        let rpc_time = rpc_start.elapsed();
        let runtime_start = Instant::now();

        let result = match operation {
            "create" => self.handle_create(args),
            "reset" => self.handle_reset(args),
            "step" => self.handle_step(args),
            "close" => self.handle_close(args),
            "info" => self.handle_info(args),
            "list" => self.handle_list(args),
            "metrics" => self.handle_metrics(args),
            _ => Err(BridgeError::unknown_tool(tool_name)),
        };

        let runtime_time = runtime_start.elapsed();
        let total_time = start.elapsed();

        // Estimate env compute time (total - rpc - runtime overhead)
        let env_time = total_time.saturating_sub(rpc_time + runtime_time);

        // Record metrics if enabled
        if self.config.collect_metrics {
            self.metrics.record_call(rpc_time, runtime_time, env_time);
        }

        match result {
            Ok(data) => {
                debug!(tool_name = %tool_name, duration_us = %total_time.as_micros(), "Tool call succeeded");
                if self.config.collect_metrics {
                    ToolResult::success_with_timing(
                        data,
                        TimingBreakdown::from_durations(rpc_time, runtime_time, env_time),
                    )
                } else {
                    ToolResult::success(data)
                }
            }
            Err(err) => {
                warn!(tool_name = %tool_name, error = %err, "Tool call failed");
                ToolResult::from_error(err)
            }
        }
    }

    /// Handle create session tool.
    fn handle_create(&mut self, args: Value) -> BridgeResult<Value> {
        let seed = args.get("seed").and_then(|v| v.as_u64());
        let auto_reset = args
            .get("auto_reset")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let env_config = args.get("config").cloned();

        let mut config = SessionConfig::new().with_auto_reset(auto_reset);
        if let Some(s) = seed {
            config = config.with_seed(s);
        }
        if let Some(c) = env_config {
            config = config.with_env_config(c);
        }

        let session_id = self.sessions.create_session(config)?;
        info!(session_id = %session_id, "Created new session");

        Ok(json!({
            "session_id": session_id.as_str(),
            "status": "created"
        }))
    }

    /// Handle reset tool.
    fn handle_reset(&mut self, args: Value) -> BridgeResult<Value> {
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BridgeError::invalid_action("session_id is required"))?;
        let session_id = SessionId::from_string(session_id);

        let seed = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);

        // Get session and verify it can be reset
        let session = self.sessions.get_mut(&session_id)?;
        if !session.state.can_reset() {
            return Err(BridgeError::environment(format!(
                "Session cannot be reset in state {:?}",
                session.state
            )));
        }

        // Simulate reset (in real impl, this would call the runtime)
        let observation = Self::simulate_reset(seed);
        session.mark_reset(observation.clone());

        debug!(session_id = %session_id, seed = %seed, "Session reset");

        Ok(json!({
            "session_id": session_id.as_str(),
            "observation": observation,
            "info": {}
        }))
    }

    /// Handle step tool.
    fn handle_step(&mut self, args: Value) -> BridgeResult<Value> {
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BridgeError::invalid_action("session_id is required"))?;
        let session_id = SessionId::from_string(session_id);

        let action = args
            .get("action")
            .ok_or_else(|| BridgeError::invalid_action("action is required"))?
            .clone();

        // Get session and verify it can step
        let session = self.sessions.get_mut(&session_id)?;
        if !session.state.can_step() {
            return Err(BridgeError::environment(format!(
                "Session cannot step in state {:?}",
                session.state
            )));
        }

        // Simulate step (in real impl, this would call the runtime)
        let (observation, reward, terminated, truncated) = Self::simulate_step(&action);
        session.mark_step(observation.clone(), reward, terminated || truncated);

        debug!(
            session_id = %session_id,
            reward = %reward,
            terminated = %terminated,
            "Session stepped"
        );

        Ok(json!({
            "session_id": session_id.as_str(),
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "episode_steps": session.episode_steps,
                "episode_reward": session.episode_reward
            }
        }))
    }

    /// Handle close tool.
    fn handle_close(&mut self, args: Value) -> BridgeResult<Value> {
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BridgeError::invalid_action("session_id is required"))?;
        let session_id = SessionId::from_string(session_id);

        self.sessions.close_session(&session_id)?;
        info!(session_id = %session_id, "Session closed");

        Ok(json!({
            "session_id": session_id.as_str(),
            "status": "closed"
        }))
    }

    /// Handle info tool.
    fn handle_info(&mut self, args: Value) -> BridgeResult<Value> {
        if let Some(session_id) = args.get("session_id").and_then(|v| v.as_str()) {
            let session_id = SessionId::from_string(session_id);
            let session = self.sessions.get(&session_id)?;
            Ok(session.info())
        } else {
            // Return environment info
            Ok(json!({
                "env_name": self.config.get_env_name(),
                "component_path": self.config.component_path,
                "max_sessions": self.config.max_sessions,
                "observation_space": {
                    "type": "box",
                    "shape": [4],
                    "dtype": "float32"
                },
                "action_space": {
                    "type": "discrete",
                    "n": 2
                }
            }))
        }
    }

    /// Handle list tool.
    fn handle_list(&mut self, _args: Value) -> BridgeResult<Value> {
        let sessions: Vec<Value> = self
            .sessions
            .list_sessions()
            .iter()
            .map(|s| s.info())
            .collect();

        Ok(json!({
            "sessions": sessions,
            "count": sessions.len(),
            "stats": self.sessions.stats()
        }))
    }

    /// Handle metrics tool.
    fn handle_metrics(&mut self, _args: Value) -> BridgeResult<Value> {
        let summary = self.metrics.summary();
        Ok(serde_json::to_value(summary)?)
    }

    /// Simulate a reset (placeholder for actual runtime call).
    fn simulate_reset(seed: u64) -> Value {
        // In real implementation, this would call wasmrl-runtime
        json!([seed as f64 % 1.0, 0.0, 0.0, 0.0])
    }

    /// Simulate a step (placeholder for actual runtime call).
    fn simulate_step(_action: &Value) -> (Value, f64, bool, bool) {
        // In real implementation, this would call wasmrl-runtime
        let observation = json!([0.1, 0.2, 0.3, 0.4]);
        let reward = 1.0;
        let terminated = false;
        let truncated = false;
        (observation, reward, terminated, truncated)
    }
}

/// Thread-safe wrapper for EnvMcpBridge.
#[derive(Debug, Clone)]
pub struct SharedEnvMcpBridge(Arc<Mutex<EnvMcpBridge>>);

impl SharedEnvMcpBridge {
    /// Create a new shared bridge.
    pub fn new(config: McpBridgeConfig) -> BridgeResult<Self> {
        let bridge = EnvMcpBridge::new(config)?;
        Ok(Self(Arc::new(Mutex::new(bridge))))
    }

    /// Get tool definitions.
    pub fn get_tools(&self) -> Vec<McpTool> {
        self.0.lock().unwrap().get_tools().to_vec()
    }

    /// Call a tool.
    pub fn call_tool(&self, tool_name: &str, args: Value) -> ToolResult {
        self.0.lock().unwrap().call_tool(tool_name, args)
    }

    /// Get metrics.
    pub fn get_metrics(&self) -> crate::overhead::OverheadSummary {
        self.0.lock().unwrap().get_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success(json!({"key": "value"}));
        assert!(result.is_ok);
        assert!(result.data.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something failed");
        assert!(!result.is_ok);
        assert!(result.data.is_none());
        assert_eq!(result.error, Some("Something failed".to_string()));
    }

    #[test]
    fn test_mcp_tool_creation() {
        let tool =
            McpTool::new("my_tool", "Does something useful").with_schema(json!({"type": "object"}));
        assert_eq!(tool.name, "my_tool");
        assert_eq!(tool.description, "Does something useful");
    }

    #[test]
    fn test_bridge_creation() {
        let config = McpBridgeConfig::new("counter.wasm");
        let bridge = EnvMcpBridge::new(config).unwrap();
        assert!(!bridge.get_tools().is_empty());
    }

    #[test]
    fn test_bridge_tool_names() {
        let config = McpBridgeConfig::new("test_env.wasm");
        let bridge = EnvMcpBridge::new(config).unwrap();
        let tools = bridge.get_tools();

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"test_env_create"));
        assert!(names.contains(&"test_env_reset"));
        assert!(names.contains(&"test_env_step"));
        assert!(names.contains(&"test_env_close"));
        assert!(names.contains(&"test_env_info"));
        assert!(names.contains(&"test_env_list"));
        assert!(names.contains(&"test_env_metrics"));
    }

    #[test]
    fn test_bridge_create_session() {
        let config = McpBridgeConfig::new("env.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        let result = bridge.call_tool("env_create", json!({"seed": 42}));
        assert!(result.is_ok);
        assert!(result.data.unwrap().get("session_id").is_some());
    }

    #[test]
    fn test_bridge_session_workflow() {
        let config = McpBridgeConfig::new("env.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        // Create session
        let create_result = bridge.call_tool("env_create", json!({}));
        assert!(create_result.is_ok);
        let session_id = create_result.data.unwrap()["session_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Reset session
        let reset_result =
            bridge.call_tool("env_reset", json!({"session_id": session_id, "seed": 123}));
        assert!(reset_result.is_ok);

        // Step session
        let step_result =
            bridge.call_tool("env_step", json!({"session_id": session_id, "action": 1}));
        assert!(step_result.is_ok);
        let step_data = step_result.data.unwrap();
        assert!(step_data.get("observation").is_some());
        assert!(step_data.get("reward").is_some());

        // Close session
        let close_result = bridge.call_tool("env_close", json!({"session_id": session_id}));
        assert!(close_result.is_ok);
    }

    #[test]
    fn test_bridge_unknown_tool() {
        let config = McpBridgeConfig::new("env.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        let result = bridge.call_tool("unknown_tool", json!({}));
        assert!(!result.is_ok);
        assert!(result.error.unwrap().contains("Unknown tool"));
    }

    #[test]
    fn test_bridge_session_not_found() {
        let config = McpBridgeConfig::new("env.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        let result = bridge.call_tool("env_reset", json!({"session_id": "nonexistent"}));
        assert!(!result.is_ok);
        assert!(result.error.unwrap().contains("not found"));
    }

    #[test]
    fn test_bridge_metrics() {
        let config = McpBridgeConfig::new("env.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        // Perform some operations
        bridge.call_tool("env_create", json!({}));
        bridge.call_tool("env_list", json!({}));

        let result = bridge.call_tool("env_metrics", json!({}));
        assert!(result.is_ok);
        let metrics = result.data.unwrap();
        assert!(metrics.get("total_calls").is_some());
    }

    #[test]
    fn test_bridge_info_env() {
        let config = McpBridgeConfig::new("counter.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        let result = bridge.call_tool("counter_info", json!({}));
        assert!(result.is_ok);
        let info = result.data.unwrap();
        assert_eq!(info["env_name"], "counter");
    }

    #[test]
    fn test_bridge_list_sessions() {
        let config = McpBridgeConfig::new("env.wasm");
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        // Create some sessions
        bridge.call_tool("env_create", json!({}));
        bridge.call_tool("env_create", json!({}));

        let result = bridge.call_tool("env_list", json!({}));
        assert!(result.is_ok);
        let data = result.data.unwrap();
        assert_eq!(data["count"], 2);
    }

    #[test]
    fn test_shared_bridge() {
        let config = McpBridgeConfig::new("env.wasm");
        let bridge = SharedEnvMcpBridge::new(config).unwrap();

        let tools = bridge.get_tools();
        assert!(!tools.is_empty());

        let result = bridge.call_tool("env_create", json!({}));
        assert!(result.is_ok);
    }

    #[test]
    fn test_timing_in_result() {
        let config = McpBridgeConfig::new("env.wasm").with_metrics(true);
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        let result = bridge.call_tool("env_list", json!({}));
        assert!(result.is_ok);
        assert!(result.timing.is_some());
    }
}
