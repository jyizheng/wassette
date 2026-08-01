// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Counter Environment - A minimal RL environment for testing.
//!
//! This environment implements a simple counter that the agent must
//! learn to increment to a target value. It demonstrates:
//!
//! - Deterministic behavior with seeded RNG
//! - Tensor-based observation and action spaces
//! - Episode termination logic
//! - Snapshot/restore capability
//!
//! # Environment Specification
//!
//! - **Observation**: Single f32 value (current counter value)
//! - **Action**: Single i32 value (0 = decrement, 1 = increment, 2 = no-op)
//! - **Reward**: +1 for reaching target, -0.01 per step (encourages efficiency)
//! - **Termination**: When counter reaches target value
//! - **Truncation**: After max_steps (default: 100)

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use wasmrl_sdk_rust::{DeterministicRng, SnapshotHelper, TensorEncoder};

#[cfg(target_arch = "wasm32")]
#[allow(warnings)]
mod bindings;

/// Configuration for the counter environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterConfig {
    /// Target value to reach.
    #[serde(default = "default_target")]
    pub target: i32,
    /// Maximum steps before truncation.
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,
    /// Initial counter value (if None, randomized based on seed).
    #[serde(default)]
    pub initial_value: Option<i32>,
}

fn default_target() -> i32 {
    10
}

fn default_max_steps() -> u32 {
    100
}

impl Default for CounterConfig {
    fn default() -> Self {
        Self {
            target: default_target(),
            max_steps: default_max_steps(),
            initial_value: None,
        }
    }
}

/// Internal state of the counter environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterState {
    /// Current counter value.
    pub counter: i32,
    /// Current step number.
    pub step_count: u32,
    /// RNG state for reproducibility.
    rng_state: u64,
}

/// Counter environment implementation.
#[derive(Debug)]
pub struct CounterEnv {
    /// Environment configuration.
    config: CounterConfig,
    /// Current state (None if not initialized).
    state: Option<CounterState>,
    /// RNG for initialization.
    rng: DeterministicRng,
}

impl CounterEnv {
    /// Create a new counter environment with default configuration.
    pub fn new() -> Self {
        Self {
            config: CounterConfig::default(),
            state: None,
            rng: DeterministicRng::new(0),
        }
    }

    /// Initialize environment with JSON configuration.
    pub fn init(&mut self, config_json: &str) -> Result<u64, String> {
        self.config = if config_json.is_empty() || config_json == "{}" {
            CounterConfig::default()
        } else {
            serde_json::from_str(config_json)
                .map_err(|e| format!("init failed: invalid config JSON: {}", e))?
        };
        Ok(1) // Return handle ID
    }

    /// Reset environment with given seed.
    pub fn reset(&mut self, seed: u64) -> Result<Vec<u8>, String> {
        self.rng = DeterministicRng::new(seed);

        let initial_value = self.config.initial_value.unwrap_or_else(|| {
            // Random initial value between -5 and 5
            self.rng.next_i32_range(-5, 6)
        });

        self.state = Some(CounterState {
            counter: initial_value,
            step_count: 0,
            rng_state: seed,
        });

        Ok(self.get_observation())
    }

    /// Execute one step with given action.
    pub fn step(&mut self, action: &[u8]) -> Result<StepOutput, String> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| "step failed: environment not reset".to_string())?;

        // Decode action (expecting single i32)
        if action.len() != 4 {
            return Err(format!(
                "step failed: invalid action size {}, expected 4",
                action.len()
            ));
        }
        let action_value = i32::from_le_bytes(action.try_into().unwrap());

        // Apply action
        match action_value {
            0 => state.counter -= 1, // Decrement
            1 => state.counter += 1, // Increment
            2 => {}                  // No-op
            _ => {
                return Err(format!(
                    "step failed: invalid action {}, expected 0, 1, or 2",
                    action_value
                ))
            }
        }

        state.step_count += 1;

        // Calculate reward and termination
        let terminated = state.counter == self.config.target;
        let truncated = state.step_count >= self.config.max_steps;
        let reward = if terminated {
            1.0
        } else {
            -0.01 // Small penalty per step
        };

        Ok(StepOutput {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated,
            info: None,
        })
    }

    /// Get current observation as bytes.
    fn get_observation(&self) -> Vec<u8> {
        let counter = self.state.as_ref().map(|s| s.counter).unwrap_or(0);
        TensorEncoder::encode_f32(&[counter as f32])
    }

    /// Get observation space specification.
    pub fn observation_space(&self) -> Vec<u8> {
        // Shape: [1] (single value)
        // Return a sample observation
        TensorEncoder::encode_f32(&[0.0])
    }

    /// Get action space specification.
    pub fn action_space(&self) -> Vec<u8> {
        // Shape: [1] (single discrete action: 0, 1, or 2)
        TensorEncoder::encode_i32(&[3]) // 3 possible actions
    }

    /// Capture environment state as snapshot.
    pub fn snapshot(&self) -> Result<Vec<u8>, String> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| "snapshot failed: environment not reset".to_string())?;

        SnapshotHelper::serialize(state)
    }

    /// Restore environment from snapshot.
    pub fn restore(&mut self, snapshot: &[u8]) -> Result<(), String> {
        let state: CounterState = SnapshotHelper::deserialize(snapshot)?;
        self.state = Some(state);
        Ok(())
    }

    /// Close environment and release resources.
    pub fn close(&mut self) -> Result<(), String> {
        self.state = None;
        Ok(())
    }
}

impl Default for CounterEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Output of a single step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// Observation bytes.
    pub observation: Vec<u8>,
    /// Reward value.
    pub reward: f64,
    /// Whether episode terminated.
    pub terminated: bool,
    /// Whether episode was truncated.
    pub truncated: bool,
    /// Optional info string.
    pub info: Option<String>,
}

impl StepOutput {
    /// Check if episode is done.
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// WebAssembly component implementation of the WasmRL environment world.
#[cfg(target_arch = "wasm32")]
mod component {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use bindings::exports::wasmrl::env::{batch, environment, snapshot};

    use super::{bindings, CounterEnv};

    struct Component;

    #[derive(Default)]
    struct ComponentEnvironments {
        next_id: u64,
        environments: HashMap<u64, CounterEnv>,
    }

    static COMPONENT_ENVIRONMENTS: OnceLock<Mutex<ComponentEnvironments>> = OnceLock::new();

    fn component_environments() -> &'static Mutex<ComponentEnvironments> {
        COMPONENT_ENVIRONMENTS.get_or_init(|| {
            Mutex::new(ComponentEnvironments {
                next_id: 1,
                environments: HashMap::new(),
            })
        })
    }

    fn with_component_env<T>(
        handle: environment::EnvHandle,
        operation: impl FnOnce(&mut CounterEnv) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut registry = component_environments()
            .lock()
            .map_err(|_| "environment registry lock poisoned".to_string())?;
        let env = registry
            .environments
            .get_mut(&handle.id)
            .ok_or_else(|| format!("environment handle {} not found", handle.id))?;
        operation(env)
    }

    fn observation_tensor(data: Vec<u8>) -> environment::Tensor {
        environment::Tensor {
            dtype: environment::Dtype::Float32,
            shape: vec![1],
            data,
        }
    }

    fn action_space_tensor(data: Vec<u8>) -> environment::Tensor {
        environment::Tensor {
            dtype: environment::Dtype::Int32,
            shape: vec![1],
            data,
        }
    }

    impl environment::Guest for Component {
        fn init(config: environment::EnvConfig) -> Result<environment::EnvHandle, String> {
            let mut env = CounterEnv::new();
            env.init(&config.config_json)?;

            let mut registry = component_environments()
                .lock()
                .map_err(|_| "environment registry lock poisoned".to_string())?;
            let id = registry.next_id;
            registry.next_id = registry.next_id.wrapping_add(1).max(1);
            registry.environments.insert(id, env);
            Ok(environment::EnvHandle { id })
        }

        fn reset(handle: environment::EnvHandle, seed: u64) -> Result<environment::Tensor, String> {
            with_component_env(handle, |env| env.reset(seed).map(observation_tensor))
        }

        fn step(
            handle: environment::EnvHandle,
            action: environment::Tensor,
        ) -> Result<environment::StepResult, String> {
            if action.dtype != environment::Dtype::Int32
                || action.shape.as_slice() != [1]
                || action.data.len() != std::mem::size_of::<i32>()
            {
                return Err("counter action must be an int32 tensor with shape [1]".to_string());
            }

            with_component_env(handle, |env| {
                let result = env.step(&action.data)?;
                Ok(environment::StepResult {
                    observation: observation_tensor(result.observation),
                    reward: result.reward,
                    terminated: result.terminated,
                    truncated: result.truncated,
                    info: result.info,
                })
            })
        }

        fn observation_space(
            handle: environment::EnvHandle,
        ) -> Result<environment::Tensor, String> {
            with_component_env(handle, |env| {
                Ok(observation_tensor(env.observation_space()))
            })
        }

        fn action_space(handle: environment::EnvHandle) -> Result<environment::Tensor, String> {
            with_component_env(handle, |env| Ok(action_space_tensor(env.action_space())))
        }

        fn close(handle: environment::EnvHandle) -> Result<(), String> {
            let mut registry = component_environments()
                .lock()
                .map_err(|_| "environment registry lock poisoned".to_string())?;
            let mut env = registry
                .environments
                .remove(&handle.id)
                .ok_or_else(|| format!("environment handle {} not found", handle.id))?;
            env.close()
        }
    }

    impl batch::Guest for Component {
        fn reset_batch(
            handles: Vec<environment::EnvHandle>,
            seeds: Vec<u64>,
        ) -> Result<Vec<environment::Tensor>, String> {
            if handles.len() != seeds.len() {
                return Err(format!(
                    "batch size mismatch: {} handles, {} seeds",
                    handles.len(),
                    seeds.len()
                ));
            }

            handles
                .into_iter()
                .zip(seeds)
                .map(|(handle, seed)| <Self as environment::Guest>::reset(handle, seed))
                .collect()
        }

        fn step_batch(
            handles: Vec<environment::EnvHandle>,
            actions: Vec<environment::Tensor>,
        ) -> Result<batch::BatchStepResult, String> {
            if handles.len() != actions.len() {
                return Err(format!(
                    "batch size mismatch: {} handles, {} actions",
                    handles.len(),
                    actions.len()
                ));
            }

            let mut result = batch::BatchStepResult {
                observations: Vec::with_capacity(handles.len()),
                rewards: Vec::with_capacity(handles.len()),
                terminated: Vec::with_capacity(handles.len()),
                truncated: Vec::with_capacity(handles.len()),
                infos: Vec::with_capacity(handles.len()),
            };

            for (handle, action) in handles.into_iter().zip(actions) {
                let step = <Self as environment::Guest>::step(handle, action)?;
                result.observations.push(step.observation);
                result.rewards.push(step.reward);
                result.terminated.push(step.terminated);
                result.truncated.push(step.truncated);
                result.infos.push(step.info);
            }

            Ok(result)
        }
    }

    impl snapshot::Guest for Component {
        fn snapshot(handle: environment::EnvHandle) -> Result<snapshot::SnapshotData, String> {
            with_component_env(handle, |env| {
                Ok(snapshot::SnapshotData {
                    version: 1,
                    data: env.snapshot()?,
                })
            })
        }

        fn restore(
            handle: environment::EnvHandle,
            state: snapshot::SnapshotData,
        ) -> Result<(), String> {
            if state.version != 1 {
                return Err(format!("unsupported snapshot version {}", state.version));
            }
            with_component_env(handle, |env| env.restore(&state.data))
        }
    }

    bindings::export!(Component with_types_in bindings);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_env_new() {
        let env = CounterEnv::new();
        assert!(env.state.is_none());
    }

    #[test]
    fn test_counter_env_init() {
        let mut env = CounterEnv::new();
        let result = env.init(r#"{"target": 5, "max_steps": 50}"#);
        assert!(result.is_ok());
        assert_eq!(env.config.target, 5);
        assert_eq!(env.config.max_steps, 50);
    }

    #[test]
    fn test_counter_env_reset() {
        let mut env = CounterEnv::new();
        env.init("{}").unwrap();
        let obs = env.reset(42).unwrap();
        assert_eq!(obs.len(), 4); // Single f32
        assert!(env.state.is_some());
    }

    #[test]
    fn test_counter_env_step() {
        let mut env = CounterEnv::new();
        env.init(r#"{"initial_value": 0, "target": 1}"#).unwrap();
        env.reset(42).unwrap();

        // Increment action
        let action = TensorEncoder::encode_i32(&[1]);
        let result = env.step(&action).unwrap();

        assert!(result.terminated); // Should reach target
        assert_eq!(result.reward, 1.0);
    }

    #[test]
    fn test_counter_env_decrement() {
        let mut env = CounterEnv::new();
        env.init(r#"{"initial_value": 0, "target": -1}"#).unwrap();
        env.reset(42).unwrap();

        // Decrement action
        let action = TensorEncoder::encode_i32(&[0]);
        let result = env.step(&action).unwrap();

        assert!(result.terminated);
    }

    #[test]
    fn test_counter_env_truncation() {
        let mut env = CounterEnv::new();
        env.init(r#"{"initial_value": 0, "target": 1000, "max_steps": 3}"#)
            .unwrap();
        env.reset(42).unwrap();

        // No-op actions until truncation
        let action = TensorEncoder::encode_i32(&[2]);
        for _ in 0..2 {
            let result = env.step(&action).unwrap();
            assert!(!result.done());
        }
        let result = env.step(&action).unwrap();
        assert!(result.truncated);
        assert!(!result.terminated);
    }

    #[test]
    fn test_counter_env_determinism() {
        // Same seed should produce same initial state
        let mut env1 = CounterEnv::new();
        let mut env2 = CounterEnv::new();

        env1.init("{}").unwrap();
        env2.init("{}").unwrap();

        let obs1 = env1.reset(12345).unwrap();
        let obs2 = env2.reset(12345).unwrap();

        assert_eq!(obs1, obs2, "Same seed must produce same initial state");
    }

    #[test]
    fn test_counter_env_trajectory_determinism() {
        // Same seed + same actions = same trajectory
        let actions = vec![1, 1, 0, 1, 2, 1]; // Sequence of actions

        let mut trajectories = Vec::new();
        for _ in 0..5 {
            let mut env = CounterEnv::new();
            env.init(r#"{"initial_value": 0, "target": 100}"#).unwrap();
            env.reset(42).unwrap();

            let mut trajectory = Vec::new();
            for &a in &actions {
                let action = TensorEncoder::encode_i32(&[a]);
                let result = env.step(&action).unwrap();
                trajectory.push((result.observation.clone(), result.reward));
            }
            trajectories.push(trajectory);
        }

        // All trajectories should be identical
        for i in 1..trajectories.len() {
            assert_eq!(
                trajectories[0], trajectories[i],
                "Trajectory {} differs from trajectory 0",
                i
            );
        }
    }

    #[test]
    fn test_counter_env_snapshot_restore() {
        let mut env = CounterEnv::new();
        env.init(r#"{"initial_value": 0, "target": 10}"#).unwrap();
        env.reset(42).unwrap();

        // Take a few steps
        let action = TensorEncoder::encode_i32(&[1]);
        env.step(&action).unwrap();
        env.step(&action).unwrap();

        // Snapshot
        let snapshot = env.snapshot().unwrap();
        let state_before = env.state.clone();

        // Take more steps
        env.step(&action).unwrap();
        env.step(&action).unwrap();

        // Restore
        env.restore(&snapshot).unwrap();

        // State should match snapshot
        assert_eq!(
            env.state.as_ref().unwrap().counter,
            state_before.as_ref().unwrap().counter
        );
        assert_eq!(
            env.state.as_ref().unwrap().step_count,
            state_before.as_ref().unwrap().step_count
        );
    }

    #[test]
    fn test_counter_env_invalid_action() {
        let mut env = CounterEnv::new();
        env.init("{}").unwrap();
        env.reset(42).unwrap();

        // Invalid action value
        let action = TensorEncoder::encode_i32(&[5]);
        let result = env.step(&action);
        assert!(result.is_err());

        // Invalid action size
        let action = vec![1, 2]; // Wrong size
        let result = env.step(&action);
        assert!(result.is_err());
    }

    #[test]
    fn test_counter_env_not_reset_error() {
        let mut env = CounterEnv::new();
        env.init("{}").unwrap();

        // Step without reset should fail
        let action = TensorEncoder::encode_i32(&[1]);
        let result = env.step(&action);
        assert!(result.is_err());
    }
}
