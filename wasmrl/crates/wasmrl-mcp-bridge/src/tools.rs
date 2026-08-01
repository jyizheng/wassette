// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! MCP tool definitions and bridge implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, info, instrument, warn};
use wasmrl_policy::Policy;
use wasmrl_runtime::{
    ComponentRef, EnvRuntime, InstanceHandle, PolicyConfig, RuntimeConfig, WasmEnvFactory,
};
use wasmrl_wit::{DType, EnvConfig, Tensor};

use crate::config::{McpBridgeConfig, SessionConfig};
use crate::error::{BridgeError, BridgeResult};
use crate::overhead::{OverheadMetrics, TimingBreakdown};
use crate::session::{SessionId, SessionManager};

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
pub struct EnvMcpBridge {
    /// Bridge configuration.
    config: McpBridgeConfig,

    /// Session manager.
    sessions: SessionManager,

    /// Overhead metrics.
    metrics: OverheadMetrics,

    /// Tool definitions.
    tools: Vec<McpTool>,

    /// Lazily loaded component runtime.
    runtime: Option<EnvRuntime>,

    /// Runtime handles keyed by MCP session ID.
    handles: HashMap<SessionId, InstanceHandle>,

    /// Cached observation-space descriptor from the component.
    observation_space: Option<Tensor>,

    /// Cached action-space descriptor from the component.
    action_space: Option<Tensor>,
}

impl std::fmt::Debug for EnvMcpBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvMcpBridge")
            .field("config", &self.config)
            .field("sessions", &self.sessions)
            .field("metrics", &self.metrics)
            .field("tools", &self.tools)
            .field("runtime_loaded", &self.runtime.is_some())
            .field("handles", &self.handles)
            .field("observation_space", &self.observation_space)
            .field("action_space", &self.action_space)
            .finish()
    }
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
            runtime: None,
            handles: HashMap::new(),
            observation_space: None,
            action_space: None,
        })
    }

    fn load_policy(&self) -> BridgeResult<PolicyConfig> {
        let Some(path) = self.config.policy_path.as_deref() else {
            return Ok(PolicyConfig::default());
        };

        let policy = Policy::from_file(std::path::Path::new(path))
            .map_err(|error| BridgeError::policy_violation(error.to_string()))?;
        policy
            .validate()
            .map_err(|error| BridgeError::policy_violation(error.to_string()))?;

        Ok(PolicyConfig {
            max_memory_mb: policy.memory.enabled.then_some(policy.memory.max_mb as u64),
            fuel_per_step: policy.fuel.enabled.then_some(policy.fuel.per_step),
            fuel_per_reset: policy.fuel.enabled.then_some(policy.fuel.per_reset),
            fuel_per_batch: policy.fuel.enabled.then_some(policy.fuel.per_batch),
            timeout_ms_step: policy.timeout.enabled.then_some(policy.timeout.step_ms),
            timeout_ms_reset: policy.timeout.enabled.then_some(policy.timeout.reset_ms),
            fs_read_paths: policy
                .wasi
                .filesystem_read
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            fs_write_paths: policy
                .wasi
                .filesystem_write
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            network_enabled: policy.wasi.network,
        })
    }

    fn ensure_runtime(&mut self) -> BridgeResult<()> {
        if self.runtime.is_some() {
            return Ok(());
        }
        if self.config.component_path.is_empty() {
            return Err(BridgeError::component_load("component path is empty"));
        }

        let policy = self.load_policy()?;
        let runtime_config = RuntimeConfig::new()
            .with_max_instances(self.config.max_sessions)
            .with_step_timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .with_reset_timeout(std::time::Duration::from_millis(self.config.timeout_ms));
        let factory = WasmEnvFactory::with_config(
            ComponentRef::from_file(&self.config.component_path),
            policy,
            runtime_config,
        )
        .map_err(|error| BridgeError::component_load(error.to_string()))?;
        self.runtime = Some(EnvRuntime::new(Arc::new(factory)));
        Ok(())
    }

    fn runtime_mut(&mut self) -> BridgeResult<&mut EnvRuntime> {
        self.runtime
            .as_mut()
            .ok_or_else(|| BridgeError::runtime("component runtime is not loaded"))
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

        self.ensure_runtime()?;
        let env_config = config
            .env_config
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?
            .unwrap_or_else(|| "{}".to_string());

        let handle = self
            .runtime
            .as_ref()
            .expect("runtime was loaded above")
            .factory()
            .spawn_one()
            .map_err(|error| BridgeError::runtime(error.to_string()))?;

        let init_result = self.runtime_mut()?.init(handle, EnvConfig::new(env_config));
        if let Err(error) = init_result {
            let _ = self
                .runtime
                .as_ref()
                .expect("runtime was loaded above")
                .factory()
                .release(handle);
            return Err(BridgeError::environment(error.to_string()));
        }

        let observation_space = match self.runtime_mut()?.observation_space(handle) {
            Ok(space) => space,
            Err(error) => {
                let _ = self.runtime_mut()?.close(handle);
                return Err(BridgeError::environment(error.to_string()));
            }
        };
        let action_space = match self.runtime_mut()?.action_space(handle) {
            Ok(space) => space,
            Err(error) => {
                let _ = self.runtime_mut()?.close(handle);
                return Err(BridgeError::environment(error.to_string()));
            }
        };

        let session_id = match self.sessions.create_session(config) {
            Ok(session_id) => session_id,
            Err(error) => {
                let _ = self.runtime_mut()?.close(handle);
                return Err(error);
            }
        };
        self.sessions
            .get_mut(&session_id)?
            .set_instance_handle(handle.id);
        self.handles.insert(session_id.clone(), handle);
        self.observation_space = Some(observation_space);
        self.action_space = Some(action_space);
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

        let seed = {
            let session = self.sessions.get(&session_id)?;
            if !session.state.can_reset() {
                return Err(BridgeError::environment(format!(
                    "Session cannot be reset in state {:?}",
                    session.state
                )));
            }
            args.get("seed")
                .and_then(|value| value.as_u64())
                .or(session.config.seed)
                .unwrap_or(0)
        };
        let handle = self.handle_for_session(&session_id)?;
        let observation = match self.runtime_mut()?.reset(handle, seed) {
            Ok(observation) => Self::tensor_to_json(&observation)?,
            Err(error) => {
                self.sessions.get_mut(&session_id)?.mark_error();
                return Err(BridgeError::environment(error.to_string()));
            }
        };
        self.sessions
            .get_mut(&session_id)?
            .mark_reset(observation.clone());

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

        let action_json = args
            .get("action")
            .ok_or_else(|| BridgeError::invalid_action("action is required"))?
            .clone();

        let (auto_reset, reset_seed, max_steps, episode_steps) = {
            let session = self.sessions.get(&session_id)?;
            if !session.state.can_step() {
                return Err(BridgeError::environment(format!(
                    "Session cannot step in state {:?}",
                    session.state
                )));
            }
            (
                session.config.auto_reset,
                session.config.seed.unwrap_or(0),
                session.config.max_steps,
                session.episode_steps,
            )
        };

        let handle = self.handle_for_session(&session_id)?;
        let action_space = self
            .action_space
            .clone()
            .ok_or_else(|| BridgeError::runtime("action space is unavailable"))?;
        let action = Self::json_to_action_tensor(&action_json, &action_space)?;
        let result = match self.runtime_mut()?.step(handle, &action) {
            Ok(result) => result,
            Err(error) => {
                self.sessions.get_mut(&session_id)?.mark_error();
                return Err(BridgeError::environment(error.to_string()));
            }
        };

        let final_observation = Self::tensor_to_json(&result.observation)?;
        let forced_truncation = max_steps.is_some_and(|limit| episode_steps + 1 >= limit);
        let terminated = result.terminated;
        let truncated = result.truncated || forced_truncation;
        let done = terminated || truncated;
        let guest_info = result
            .info
            .as_deref()
            .map(serde_json::from_str::<Value>)
            .transpose()?
            .unwrap_or(Value::Null);

        let (completed_steps, completed_reward) = {
            let session = self.sessions.get_mut(&session_id)?;
            session.mark_step(final_observation.clone(), result.reward, done);
            (session.episode_steps, session.episode_reward)
        };

        let (observation, auto_reset_observation) = if done && auto_reset {
            let reset_observation = match self.runtime_mut()?.reset(handle, reset_seed) {
                Ok(observation) => Self::tensor_to_json(&observation)?,
                Err(error) => {
                    self.sessions.get_mut(&session_id)?.mark_error();
                    return Err(BridgeError::environment(error.to_string()));
                }
            };
            self.sessions
                .get_mut(&session_id)?
                .mark_reset(reset_observation.clone());
            (reset_observation, Some(final_observation.clone()))
        } else {
            (final_observation, None)
        };

        debug!(
            session_id = %session_id,
            reward = %result.reward,
            terminated = %terminated,
            "Session stepped"
        );

        Ok(json!({
            "session_id": session_id.as_str(),
            "observation": observation,
            "reward": result.reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "episode_steps": completed_steps,
                "episode_reward": completed_reward,
                "guest": guest_info,
                "final_observation": auto_reset_observation
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

        self.sessions.get(&session_id)?;
        if let Some(handle) = self.handles.remove(&session_id) {
            self.runtime_mut()?
                .close(handle)
                .map_err(|error| BridgeError::environment(error.to_string()))?;
        }
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
            Ok(json!({
                "env_name": self.config.get_env_name(),
                "component_path": self.config.component_path,
                "max_sessions": self.config.max_sessions,
                "component_loaded": self.runtime.is_some(),
                "observation_space": self.observation_space.as_ref().map(Self::space_to_json),
                "action_space": self.action_space.as_ref().map(Self::space_to_json)
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

    fn handle_for_session(&self, session_id: &SessionId) -> BridgeResult<InstanceHandle> {
        self.handles.get(session_id).copied().ok_or_else(|| {
            BridgeError::runtime(format!("session {} has no runtime instance", session_id))
        })
    }

    fn space_to_json(space: &Tensor) -> Value {
        let discrete_count = Self::discrete_action_count(space);
        json!({
            "type": if discrete_count.is_some() { "discrete" } else { "box" },
            "shape": space.shape,
            "dtype": space.dtype.to_string(),
            "n": discrete_count,
        })
    }

    fn discrete_action_count(space: &Tensor) -> Option<i64> {
        if space.shape.as_slice() != [1] {
            return None;
        }
        let count = match space.dtype {
            DType::Int32 if space.data.len() == 4 => {
                i32::from_le_bytes(space.data.as_slice().try_into().ok()?) as i64
            }
            DType::Int64 if space.data.len() == 8 => {
                i64::from_le_bytes(space.data.as_slice().try_into().ok()?)
            }
            DType::Uint8 if space.data.len() == 1 => space.data[0] as i64,
            _ => return None,
        };
        (count > 0).then_some(count)
    }

    fn json_to_action_tensor(action: &Value, space: &Tensor) -> BridgeResult<Tensor> {
        if action.is_object() {
            if let Ok(tensor) = serde_json::from_value::<Tensor>(action.clone()) {
                if !tensor.is_valid() {
                    return Err(BridgeError::invalid_action("tensor byte length is invalid"));
                }
                return Ok(tensor);
            }
        }

        let values: Vec<&Value> = match action {
            Value::Array(values) => values.iter().collect(),
            value => vec![value],
        };
        let expected: usize = space.shape.iter().map(|&dim| dim as usize).product();
        if values.len() != expected {
            return Err(BridgeError::invalid_action(format!(
                "expected {} values for shape {:?}, got {}",
                expected,
                space.shape,
                values.len()
            )));
        }

        if let Some(count) = Self::discrete_action_count(space) {
            let value = values[0]
                .as_i64()
                .ok_or_else(|| BridgeError::invalid_action("discrete action must be an integer"))?;
            if !(0..count).contains(&value) {
                return Err(BridgeError::invalid_action(format!(
                    "discrete action must be in range 0..{}",
                    count
                )));
            }
        }

        let mut data = Vec::with_capacity(expected * space.dtype.element_size());
        for value in values {
            match space.dtype {
                DType::Float32 => data.extend_from_slice(
                    &(value.as_f64().ok_or_else(|| {
                        BridgeError::invalid_action("float32 action must contain numbers")
                    })? as f32)
                        .to_le_bytes(),
                ),
                DType::Float64 => data.extend_from_slice(
                    &value
                        .as_f64()
                        .ok_or_else(|| {
                            BridgeError::invalid_action("float64 action must contain numbers")
                        })?
                        .to_le_bytes(),
                ),
                DType::Int32 => {
                    let number = value.as_i64().ok_or_else(|| {
                        BridgeError::invalid_action("int32 action must contain integers")
                    })?;
                    let number = i32::try_from(number)
                        .map_err(|_| BridgeError::invalid_action("int32 action is out of range"))?;
                    data.extend_from_slice(&number.to_le_bytes());
                }
                DType::Int64 => data.extend_from_slice(
                    &value
                        .as_i64()
                        .ok_or_else(|| {
                            BridgeError::invalid_action("int64 action must contain integers")
                        })?
                        .to_le_bytes(),
                ),
                DType::Uint8 => {
                    let number = value.as_u64().ok_or_else(|| {
                        BridgeError::invalid_action("uint8 action must contain unsigned integers")
                    })?;
                    data.push(u8::try_from(number).map_err(|_| {
                        BridgeError::invalid_action("uint8 action is out of range")
                    })?);
                }
                DType::Boolean => data.push(
                    value
                        .as_bool()
                        .ok_or_else(|| {
                            BridgeError::invalid_action("boolean action must contain booleans")
                        })?
                        .into(),
                ),
            }
        }

        Ok(Tensor::new(space.dtype, space.shape.clone(), data))
    }

    fn tensor_to_json(tensor: &Tensor) -> BridgeResult<Value> {
        if !tensor.is_valid() {
            return Err(BridgeError::serialization(format!(
                "invalid tensor byte length for {:?} with shape {:?}",
                tensor.dtype, tensor.shape
            )));
        }

        let values = match tensor.dtype {
            DType::Float32 => tensor
                .data
                .chunks_exact(4)
                .map(|bytes| {
                    let number = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
                    serde_json::Number::from_f64(number as f64)
                        .map(Value::Number)
                        .ok_or_else(|| BridgeError::serialization("non-finite float32 value"))
                })
                .collect::<BridgeResult<Vec<_>>>()?,
            DType::Float64 => tensor
                .data
                .chunks_exact(8)
                .map(|bytes| {
                    let number = f64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"));
                    serde_json::Number::from_f64(number)
                        .map(Value::Number)
                        .ok_or_else(|| BridgeError::serialization("non-finite float64 value"))
                })
                .collect::<BridgeResult<Vec<_>>>()?,
            DType::Int32 => tensor
                .data
                .chunks_exact(4)
                .map(|bytes| {
                    Value::from(i32::from_le_bytes(
                        bytes.try_into().expect("four-byte chunk"),
                    ))
                })
                .collect(),
            DType::Int64 => tensor
                .data
                .chunks_exact(8)
                .map(|bytes| {
                    Value::from(i64::from_le_bytes(
                        bytes.try_into().expect("eight-byte chunk"),
                    ))
                })
                .collect(),
            DType::Uint8 => tensor.data.iter().copied().map(Value::from).collect(),
            DType::Boolean => tensor
                .data
                .iter()
                .map(|value| Value::Bool(*value != 0))
                .collect(),
        };
        Ok(Value::Array(values))
    }
}

impl Drop for EnvMcpBridge {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.as_mut() {
            for (_, handle) in self.handles.drain() {
                let _ = runtime.close(handle);
            }
        }
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

    #[cfg(feature = "integration")]
    fn counter_config() -> McpBridgeConfig {
        let component_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/wasm32-wasip2/release/counter_env.wasm");
        assert!(component_path.is_file());
        McpBridgeConfig::new(component_path.to_string_lossy()).with_env_name("env")
    }

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
    #[cfg(feature = "integration")]
    fn test_bridge_create_session() {
        let config = counter_config();
        let mut bridge = EnvMcpBridge::new(config).unwrap();

        let result = bridge.call_tool("env_create", json!({"seed": 42}));
        assert!(result.is_ok);
        assert!(result.data.unwrap().get("session_id").is_some());
    }

    #[test]
    #[cfg(feature = "integration")]
    fn test_bridge_session_workflow() {
        let config = counter_config();
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
    #[cfg(feature = "integration")]
    fn test_bridge_list_sessions() {
        let config = counter_config();
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
    #[cfg(feature = "integration")]
    fn test_shared_bridge() {
        let config = counter_config();
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
