// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Malicious Memory Environment - Security test for memory exhaustion.
//!
//! This environment is intentionally malicious and will attempt to:
//! - Allocate excessive memory in step()
//! - Create memory bombs
//! - Exhaust available heap
//!
//! It is used to test that the WasmRL runtime properly enforces:
//! - Memory limits (max_memory_mb)
//! - Linear memory bounds
//!
//! # WARNING
//!
//! This environment should NEVER be run without proper memory limits.

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use wasmrl_sdk_rust::TensorEncoder;

#[cfg(target_arch = "wasm32")]
#[allow(warnings)]
mod bindings;

/// Configuration for malicious memory behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaliciousMemoryConfig {
    /// How much memory to allocate per step (in MB).
    #[serde(default = "default_alloc_mb")]
    pub alloc_mb_per_step: usize,
    /// Whether to keep allocations (vs. drop them).
    #[serde(default = "default_keep")]
    pub keep_allocations: bool,
}

fn default_alloc_mb() -> usize {
    100
}

fn default_keep() -> bool {
    true
}

impl Default for MaliciousMemoryConfig {
    fn default() -> Self {
        Self {
            alloc_mb_per_step: default_alloc_mb(),
            keep_allocations: default_keep(),
        }
    }
}

/// Malicious environment that attempts to exhaust memory.
#[derive(Debug)]
pub struct MaliciousMemoryEnv {
    /// Configuration.
    config: MaliciousMemoryConfig,
    /// Whether environment is initialized.
    initialized: bool,
    /// Accumulated allocations (memory bomb).
    allocations: Vec<Vec<u8>>,
    /// Step counter.
    step_count: u32,
}

impl MaliciousMemoryEnv {
    /// Create a new malicious memory environment.
    pub fn new() -> Self {
        Self {
            config: MaliciousMemoryConfig::default(),
            initialized: false,
            allocations: Vec::new(),
            step_count: 0,
        }
    }

    /// Initialize environment with configuration.
    pub fn init(&mut self, config_json: &str) -> Result<u64, String> {
        self.config = if config_json.is_empty() || config_json == "{}" {
            MaliciousMemoryConfig::default()
        } else {
            serde_json::from_str(config_json).map_err(|e| format!("init failed: {}", e))?
        };

        self.initialized = true;
        self.allocations.clear();
        Ok(1)
    }

    /// Reset environment.
    pub fn reset(&mut self, _seed: u64) -> Result<Vec<u8>, String> {
        if !self.initialized {
            return Err("reset failed: not initialized".to_string());
        }

        self.step_count = 0;
        self.allocations.clear();
        Ok(self.get_observation())
    }

    /// Step environment - allocates memory.
    pub fn step(&mut self, _action: &[u8]) -> Result<StepOutput, String> {
        if !self.initialized {
            return Err("step failed: not initialized".to_string());
        }

        self.step_count += 1;

        // Attempt to allocate large memory block
        let bytes_to_alloc = self.config.alloc_mb_per_step * 1024 * 1024;
        let allocation = vec![0xABu8; bytes_to_alloc];

        if self.config.keep_allocations {
            // Keep allocation (memory bomb)
            self.allocations.push(allocation);
        }
        // If not keeping, allocation is dropped here

        Ok(StepOutput {
            observation: self.get_observation(),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: Some(format!(
                "allocated {} MB, total kept: {} MB",
                self.config.alloc_mb_per_step,
                self.allocations.len() * self.config.alloc_mb_per_step
            )),
        })
    }

    /// Get observation.
    fn get_observation(&self) -> Vec<u8> {
        let total_mb = self.allocations.len() * self.config.alloc_mb_per_step;
        TensorEncoder::encode_f32(&[total_mb as f32])
    }

    /// Snapshot.
    pub fn snapshot(&self) -> Result<Vec<u8>, String> {
        Ok(vec![0, 0, 0, 0])
    }

    /// Restore.
    pub fn restore(&mut self, _snapshot: &[u8]) -> Result<(), String> {
        Ok(())
    }

    /// Close environment.
    pub fn close(&mut self) -> Result<(), String> {
        self.initialized = false;
        self.allocations.clear();
        Ok(())
    }
}

impl Default for MaliciousMemoryEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Step output.
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// Observation.
    pub observation: Vec<u8>,
    /// Reward.
    pub reward: f64,
    /// Terminated flag.
    pub terminated: bool,
    /// Truncated flag.
    pub truncated: bool,
    /// Info.
    pub info: Option<String>,
}

#[cfg(target_arch = "wasm32")]
mod component {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use bindings::exports::wasmrl::env::{batch, environment, snapshot};

    use super::{bindings, MaliciousMemoryEnv};

    struct Component;

    #[derive(Default)]
    struct Environments {
        next_id: u64,
        environments: HashMap<u64, MaliciousMemoryEnv>,
    }

    static ENVIRONMENTS: OnceLock<Mutex<Environments>> = OnceLock::new();

    fn environments() -> &'static Mutex<Environments> {
        ENVIRONMENTS.get_or_init(|| {
            Mutex::new(Environments {
                next_id: 1,
                environments: HashMap::new(),
            })
        })
    }

    fn with_env<T>(
        handle: environment::EnvHandle,
        operation: impl FnOnce(&mut MaliciousMemoryEnv) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut registry = environments()
            .lock()
            .map_err(|_| "environment registry lock poisoned".to_string())?;
        let env = registry
            .environments
            .get_mut(&handle.id)
            .ok_or_else(|| format!("environment handle {} not found", handle.id))?;
        operation(env)
    }

    fn observation(data: Vec<u8>) -> environment::Tensor {
        environment::Tensor {
            dtype: environment::Dtype::Float32,
            shape: vec![1],
            data,
        }
    }

    fn action_space() -> environment::Tensor {
        environment::Tensor {
            dtype: environment::Dtype::Int32,
            shape: vec![1],
            data: 1_i32.to_le_bytes().to_vec(),
        }
    }

    impl environment::Guest for Component {
        fn init(config: environment::EnvConfig) -> Result<environment::EnvHandle, String> {
            let mut env = MaliciousMemoryEnv::new();
            env.init(&config.config_json)?;

            let mut registry = environments()
                .lock()
                .map_err(|_| "environment registry lock poisoned".to_string())?;
            let id = registry.next_id;
            registry.next_id = registry.next_id.wrapping_add(1).max(1);
            registry.environments.insert(id, env);
            Ok(environment::EnvHandle { id })
        }

        fn reset(handle: environment::EnvHandle, seed: u64) -> Result<environment::Tensor, String> {
            with_env(handle, |env| env.reset(seed).map(observation))
        }

        fn step(
            handle: environment::EnvHandle,
            action: environment::Tensor,
        ) -> Result<environment::StepResult, String> {
            if action.dtype != environment::Dtype::Int32
                || action.shape.as_slice() != [1]
                || action.data.len() != 4
            {
                return Err("action must be an int32 tensor with shape [1]".to_string());
            }

            with_env(handle, |env| {
                let result = env.step(&action.data)?;
                Ok(environment::StepResult {
                    observation: observation(result.observation),
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
            with_env(handle, |_| Ok(observation(vec![0; 4])))
        }

        fn action_space(handle: environment::EnvHandle) -> Result<environment::Tensor, String> {
            with_env(handle, |_| Ok(action_space()))
        }

        fn close(handle: environment::EnvHandle) -> Result<(), String> {
            let mut registry = environments()
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
                return Err("batch size mismatch".to_string());
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
                return Err("batch size mismatch".to_string());
            }
            let mut batch = batch::BatchStepResult {
                observations: Vec::with_capacity(handles.len()),
                rewards: Vec::with_capacity(handles.len()),
                terminated: Vec::with_capacity(handles.len()),
                truncated: Vec::with_capacity(handles.len()),
                infos: Vec::with_capacity(handles.len()),
            };
            for (handle, action) in handles.into_iter().zip(actions) {
                let result = <Self as environment::Guest>::step(handle, action)?;
                batch.observations.push(result.observation);
                batch.rewards.push(result.reward);
                batch.terminated.push(result.terminated);
                batch.truncated.push(result.truncated);
                batch.infos.push(result.info);
            }
            Ok(batch)
        }
    }

    impl snapshot::Guest for Component {
        fn snapshot(handle: environment::EnvHandle) -> Result<snapshot::SnapshotData, String> {
            with_env(handle, |env| {
                Ok(snapshot::SnapshotData {
                    version: 1,
                    data: env.snapshot()?,
                })
            })
        }

        fn restore(
            handle: environment::EnvHandle,
            data: snapshot::SnapshotData,
        ) -> Result<(), String> {
            if data.version != 1 {
                return Err(format!("unsupported snapshot version {}", data.version));
            }
            with_env(handle, |env| env.restore(&data.data))
        }
    }

    bindings::export!(Component with_types_in bindings);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malicious_memory_env_new() {
        let env = MaliciousMemoryEnv::new();
        assert!(!env.initialized);
        assert!(env.allocations.is_empty());
    }

    #[test]
    fn test_malicious_memory_env_init() {
        let mut env = MaliciousMemoryEnv::new();
        let result = env.init("{}");
        assert!(result.is_ok());
        assert!(env.initialized);
    }

    #[test]
    fn test_malicious_memory_env_config() {
        let mut env = MaliciousMemoryEnv::new();
        let result = env.init(r#"{"alloc_mb_per_step": 1, "keep_allocations": false}"#);
        assert!(result.is_ok());
        assert_eq!(env.config.alloc_mb_per_step, 1);
        assert!(!env.config.keep_allocations);
    }

    // NOTE: We test with small allocations to avoid actually exhausting memory
    #[test]
    fn test_malicious_memory_env_small_alloc() {
        let mut env = MaliciousMemoryEnv::new();
        env.init(r#"{"alloc_mb_per_step": 1, "keep_allocations": true}"#)
            .unwrap();
        env.reset(42).unwrap();

        // Small allocation should work
        let action = vec![0, 0, 0, 0];
        let result = env.step(&action);
        assert!(result.is_ok());
        assert_eq!(env.allocations.len(), 1);
    }

    #[test]
    fn test_malicious_memory_env_no_keep() {
        let mut env = MaliciousMemoryEnv::new();
        env.init(r#"{"alloc_mb_per_step": 1, "keep_allocations": false}"#)
            .unwrap();
        env.reset(42).unwrap();

        let action = vec![0, 0, 0, 0];
        env.step(&action).unwrap();
        env.step(&action).unwrap();

        // Allocations not kept
        assert!(env.allocations.is_empty());
    }
}
