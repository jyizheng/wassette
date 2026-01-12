// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Environment executor for running WasmRL environments.
//!
//! This module provides the core execution logic for:
//! - Single instance stepping
//! - Batch stepping (sync and async modes)
//! - Reset operations
//! - Snapshot/restore

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use wasmrl_wit::{BatchStepResult, EnvConfig, SnapshotData, StepResult, Tensor};

use crate::config::RuntimeConfig;
use crate::engine::EnvState;
use crate::error::{RuntimeError, RuntimeResult};
use crate::factory::WasmEnvFactory;
use crate::instance::{InstanceHandle, InstanceInfo, InstanceStatus};
use crate::metrics::{RuntimeMetrics, Timer};
use crate::pool::SharedPool;

/// Environment runtime for executing environment operations.
pub struct EnvRuntime {
    /// The factory that created this runtime.
    factory: Arc<WasmEnvFactory>,
    /// Runtime metrics.
    metrics: RuntimeMetrics,
    /// Environment state per instance.
    env_states: HashMap<u64, EnvInstanceState>,
}

/// State for a single environment instance.
#[derive(Debug)]
struct EnvInstanceState {
    /// Environment configuration.
    config: EnvConfig,
    /// Current seed.
    seed: u64,
    /// Whether environment is initialized.
    initialized: bool,
    /// Step count in current episode.
    episode_steps: u64,
    /// Latest snapshot if available.
    latest_snapshot: Option<SnapshotData>,
}

impl EnvInstanceState {
    fn new(config: EnvConfig) -> Self {
        Self {
            config,
            seed: 0,
            initialized: false,
            episode_steps: 0,
            latest_snapshot: None,
        }
    }
}

impl EnvRuntime {
    /// Create a new environment runtime from a factory.
    pub fn new(factory: Arc<WasmEnvFactory>) -> Self {
        Self {
            factory,
            metrics: RuntimeMetrics::new(),
            env_states: HashMap::new(),
        }
    }

    /// Initialize an environment instance with configuration.
    pub fn init(&mut self, handle: InstanceHandle, config: EnvConfig) -> RuntimeResult<()> {
        let state = EnvInstanceState::new(config);
        self.env_states.insert(handle.id, state);
        Ok(())
    }

    /// Reset an environment instance.
    ///
    /// Returns the initial observation.
    pub fn reset(&mut self, handle: InstanceHandle, seed: u64) -> RuntimeResult<Tensor> {
        let timer = Timer::start();

        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        state.seed = seed;
        state.initialized = true;
        state.episode_steps = 0;

        // Record metrics
        self.metrics.record_reset(timer.elapsed());
        self.factory.pool().record_reset(handle)?;

        // Return placeholder observation
        // In real implementation, this would call the Wasm component
        Ok(Tensor::zeros(wasmrl_wit::DType::Float32, vec![1]))
    }

    /// Execute a single step.
    ///
    /// Returns the step result (observation, reward, done, info).
    pub fn step(&mut self, handle: InstanceHandle, action: &Tensor) -> RuntimeResult<StepResult> {
        let timer = Timer::start();

        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        if !state.initialized {
            return Err(RuntimeError::execution(
                "Environment not initialized, call reset first",
            ));
        }

        state.episode_steps += 1;

        // Record metrics
        self.metrics.record_step(timer.elapsed());
        self.factory.pool().record_step(handle)?;

        // Return placeholder result
        // In real implementation, this would call the Wasm component
        let obs = Tensor::zeros(wasmrl_wit::DType::Float32, vec![1]);
        Ok(StepResult::new(obs, 0.0, false, false))
    }

    /// Reset multiple environments in batch.
    pub fn reset_many(
        &mut self,
        handles: &[InstanceHandle],
        seeds: &[u64],
    ) -> RuntimeResult<Vec<Tensor>> {
        if handles.len() != seeds.len() {
            return Err(RuntimeError::BatchSizeMismatch {
                expected: handles.len(),
                actual: seeds.len(),
            });
        }

        let mut observations = Vec::with_capacity(handles.len());

        for (handle, seed) in handles.iter().zip(seeds.iter()) {
            observations.push(self.reset(*handle, *seed)?);
        }

        Ok(observations)
    }

    /// Step multiple environments in batch.
    pub fn step_many(
        &mut self,
        handles: &[InstanceHandle],
        actions: &[Tensor],
    ) -> RuntimeResult<BatchStepResult> {
        let timer = Timer::start();

        if handles.len() != actions.len() {
            return Err(RuntimeError::BatchSizeMismatch {
                expected: handles.len(),
                actual: actions.len(),
            });
        }

        let n = handles.len();
        let mut result = BatchStepResult::with_capacity(n);

        for (handle, action) in handles.iter().zip(actions.iter()) {
            match self.step(*handle, action) {
                Ok(step_result) => {
                    result.observations.push(step_result.observation);
                    result.rewards.push(step_result.reward);
                    result.terminated.push(step_result.terminated);
                    result.truncated.push(step_result.truncated);
                    result.infos.push(step_result.info);
                }
                Err(e) => {
                    // Mark instance as errored
                    self.factory.pool().mark_error(*handle, true)?;
                    self.metrics.record_trap();

                    // Fill with error placeholder
                    result
                        .observations
                        .push(Tensor::zeros(wasmrl_wit::DType::Float32, vec![1]));
                    result.rewards.push(0.0);
                    result.terminated.push(true);
                    result.truncated.push(false);
                    result.infos.push(Some(format!("Error: {}", e)));
                }
            }
        }

        self.metrics.record_batch_step(timer.elapsed(), n);

        Ok(result)
    }

    /// Take a snapshot of an environment instance.
    pub fn snapshot(&self, handle: InstanceHandle) -> RuntimeResult<SnapshotData> {
        let state = self
            .env_states
            .get(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        if !state.initialized {
            return Err(RuntimeError::execution(
                "Cannot snapshot uninitialized environment",
            ));
        }

        // Create snapshot data
        // In real implementation, this would call the Wasm component
        let data = serde_json::to_vec(&serde_json::json!({
            "seed": state.seed,
            "episode_steps": state.episode_steps,
        }))
        .map_err(|e| RuntimeError::execution(e.to_string()))?;

        Ok(SnapshotData::new(data))
    }

    /// Restore an environment instance from a snapshot.
    pub fn restore(&mut self, handle: InstanceHandle, snapshot: &SnapshotData) -> RuntimeResult<()> {
        if !snapshot.is_compatible() {
            return Err(RuntimeError::execution(format!(
                "Incompatible snapshot version: {}",
                snapshot.version
            )));
        }

        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        state.initialized = true;
        state.latest_snapshot = Some(snapshot.clone());

        Ok(())
    }

    /// Close an environment instance.
    pub fn close(&mut self, handle: InstanceHandle) -> RuntimeResult<()> {
        self.env_states.remove(&handle.id);
        self.factory.pool().release(handle)?;
        Ok(())
    }

    /// Get runtime metrics.
    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }

    /// Get mutable metrics reference.
    pub fn metrics_mut(&mut self) -> &mut RuntimeMetrics {
        &mut self.metrics
    }

    /// Get the underlying factory.
    pub fn factory(&self) -> &WasmEnvFactory {
        &self.factory
    }

    /// Get environment state for an instance.
    pub fn get_state(&self, handle: InstanceHandle) -> Option<&EnvInstanceState> {
        self.env_states.get(&handle.id)
    }

    /// Get number of active environments.
    pub fn active_count(&self) -> usize {
        self.env_states.len()
    }
}

/// Batch executor for parallel environment stepping.
pub struct BatchExecutor {
    /// Runtime instance.
    runtime: EnvRuntime,
    /// Active instance handles.
    handles: Vec<InstanceHandle>,
    /// Seeds for each instance.
    seeds: Vec<u64>,
}

impl BatchExecutor {
    /// Create a new batch executor.
    pub fn new(runtime: EnvRuntime, count: usize) -> RuntimeResult<Self> {
        let handles = runtime.factory.spawn(count)?;
        let seeds = vec![0; count];

        Ok(Self {
            runtime,
            handles,
            seeds,
        })
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.handles.len()
    }

    /// Initialize all environments with the same configuration.
    pub fn init_all(&mut self, config: EnvConfig) -> RuntimeResult<()> {
        for handle in &self.handles {
            self.runtime.init(*handle, config.clone())?;
        }
        Ok(())
    }

    /// Reset all environments with given seeds.
    pub fn reset_all(&mut self, seeds: &[u64]) -> RuntimeResult<Vec<Tensor>> {
        if seeds.len() != self.handles.len() {
            return Err(RuntimeError::BatchSizeMismatch {
                expected: self.handles.len(),
                actual: seeds.len(),
            });
        }

        self.seeds.copy_from_slice(seeds);
        self.runtime.reset_many(&self.handles, seeds)
    }

    /// Step all environments with given actions.
    pub fn step_all(&mut self, actions: &[Tensor]) -> RuntimeResult<BatchStepResult> {
        self.runtime.step_many(&self.handles, actions)
    }

    /// Get the runtime.
    pub fn runtime(&self) -> &EnvRuntime {
        &self.runtime
    }

    /// Get mutable runtime.
    pub fn runtime_mut(&mut self) -> &mut EnvRuntime {
        &mut self.runtime
    }

    /// Get instance handles.
    pub fn handles(&self) -> &[InstanceHandle] {
        &self.handles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full runtime tests require actual Wasm components
    // These are tested in integration tests

    #[test]
    fn test_env_instance_state_new() {
        let config = EnvConfig::empty();
        let state = EnvInstanceState::new(config);
        assert!(!state.initialized);
        assert_eq!(state.seed, 0);
        assert_eq!(state.episode_steps, 0);
    }

    #[test]
    fn test_batch_size_mismatch_error() {
        let err = RuntimeError::BatchSizeMismatch {
            expected: 10,
            actual: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }
}
