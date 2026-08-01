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
use std::time::Duration;

use wasmrl_wit::{BatchStepResult, EnvConfig, SnapshotData, StepResult, Tensor};
use wasmtime::Store;

use crate::bindings::exports::wasmrl::env::environment;
use crate::bindings::{self, Env};
use crate::engine::EnvState;
use crate::error::{RuntimeError, RuntimeResult};
use crate::factory::WasmEnvFactory;
use crate::instance::InstanceHandle;
use crate::metrics::{RuntimeMetrics, Timer};

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
pub struct EnvInstanceState {
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
    /// Wasmtime store containing this component instance.
    store: Store<EnvState>,
    /// Type-checked bindings for the component instance.
    bindings: Env,
    /// Handle allocated by the guest environment.
    guest_handle: environment::EnvHandle,
}

impl std::fmt::Debug for EnvInstanceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvInstanceState")
            .field("config", &self.config)
            .field("seed", &self.seed)
            .field("initialized", &self.initialized)
            .field("episode_steps", &self.episode_steps)
            .field("latest_snapshot", &self.latest_snapshot)
            .field("store", &"<wasmtime::Store>")
            .field("bindings", &"<wasmrl::Env>")
            .field("guest_handle", &self.guest_handle)
            .finish()
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

    fn prepare_operation(
        state: &mut EnvInstanceState,
        fuel: u64,
        timeout: Option<Duration>,
        operation: &str,
    ) -> RuntimeResult<()> {
        if fuel > 0 {
            state
                .store
                .set_fuel(fuel)
                .map_err(|_| RuntimeError::fuel_exhausted(operation))?;
        } else {
            // A store may still have fuel enabled for a different operation.
            let _ = state.store.set_fuel(u64::MAX);
        }
        state.store.data_mut().fuel_consumed = 0;
        state.store.data_mut().clear_error();
        let deadline = timeout
            .map(|duration| duration.as_millis().max(1) as u64)
            .unwrap_or(u64::MAX);
        state.store.set_epoch_deadline(deadline);
        Ok(())
    }

    fn map_call_error(
        handle: InstanceHandle,
        operation: &str,
        timeout: Option<Duration>,
        error: anyhow::Error,
    ) -> RuntimeError {
        match error.downcast_ref::<wasmtime::Trap>() {
            Some(wasmtime::Trap::OutOfFuel) => RuntimeError::fuel_exhausted(operation),
            Some(wasmtime::Trap::Interrupt) => {
                let limit_ms = timeout
                    .map(|duration| duration.as_millis() as u64)
                    .unwrap_or_default();
                RuntimeError::timeout(operation, limit_ms, limit_ms)
            }
            _ => RuntimeError::instance_trapped(handle.id, error.to_string()),
        }
    }

    fn record_fuel(state: &mut EnvInstanceState, fuel: u64) {
        if fuel == 0 {
            return;
        }
        if let Ok(remaining) = state.store.get_fuel() {
            state.store.data_mut().fuel_consumed = fuel.saturating_sub(remaining);
        }
    }

    /// Initialize an environment instance with configuration.
    pub fn init(&mut self, handle: InstanceHandle, config: EnvConfig) -> RuntimeResult<()> {
        if self.factory.pool().get_info(handle).is_none() {
            return Err(RuntimeError::instance_not_found(handle.id));
        }

        if self
            .env_states
            .get(&handle.id)
            .is_some_and(|state| state.config.config_json == config.config_json)
        {
            return Ok(());
        }

        if let Some(mut previous) = self.env_states.remove(&handle.id) {
            let _ = Self::prepare_operation(
                &mut previous,
                self.factory.config().fuel_per_reset,
                self.factory.config().reset_timeout,
                "close",
            );
            let _ = previous
                .bindings
                .wasmrl_env_environment()
                .call_close(&mut previous.store, previous.guest_handle);
        }

        let store_state = EnvState::with_memory_limit(
            self.factory
                .config()
                .max_memory_bytes
                .min(usize::MAX as u64) as usize,
        );
        let mut store = if self.factory.config().fuel_enabled() {
            self.factory
                .engine()
                .create_store_with_fuel(store_state, u64::MAX)
                .map_err(|e| RuntimeError::instantiation(e.to_string()))?
        } else {
            self.factory.engine().create_store(store_state)
        };
        let init_fuel = self.factory.config().fuel_per_reset;
        if init_fuel > 0 {
            store
                .set_fuel(init_fuel)
                .map_err(|_| RuntimeError::fuel_exhausted("init"))?;
        }
        store.set_epoch_deadline(
            self.factory
                .config()
                .reset_timeout
                .map(|duration| duration.as_millis().max(1) as u64)
                .unwrap_or(u64::MAX),
        );
        let bindings = self
            .factory
            .bindings()
            .instantiate(&mut store)
            .map_err(|e| RuntimeError::instantiation(e.to_string()))?;
        let guest_config = environment::EnvConfig {
            config_json: config.config_json.clone(),
        };
        let guest_handle = bindings
            .wasmrl_env_environment()
            .call_init(&mut store, &guest_config)
            .map_err(|error| {
                Self::map_call_error(handle, "init", self.factory.config().reset_timeout, error)
            })?
            .map_err(RuntimeError::execution)?;

        let state = EnvInstanceState {
            config,
            seed: 0,
            initialized: false,
            episode_steps: 0,
            latest_snapshot: None,
            store,
            bindings,
            guest_handle,
        };
        self.env_states.insert(handle.id, state);
        Ok(())
    }

    /// Reset an environment instance.
    ///
    /// Returns the initial observation.
    pub fn reset(&mut self, handle: InstanceHandle, seed: u64) -> RuntimeResult<Tensor> {
        let timer = Timer::start();
        let fuel = self.factory.config().fuel_per_reset;
        let timeout = self.factory.config().reset_timeout;
        let result = {
            let state = self
                .env_states
                .get_mut(&handle.id)
                .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;
            Self::prepare_operation(state, fuel, timeout, "reset")?;

            match state.bindings.wasmrl_env_environment().call_reset(
                &mut state.store,
                state.guest_handle,
                seed,
            ) {
                Ok(Ok(observation)) => {
                    Self::record_fuel(state, fuel);
                    state.seed = seed;
                    state.initialized = true;
                    state.episode_steps = 0;
                    Ok(bindings::lift_tensor(observation))
                }
                Ok(Err(message)) => Err(RuntimeError::execution(message)),
                Err(error) => {
                    state.store.data_mut().record_trap(&error.to_string());
                    Err(Self::map_call_error(handle, "reset", timeout, error))
                }
            }
        };

        let elapsed = timer.elapsed();
        self.metrics.record_reset(elapsed);
        match result {
            Ok(observation) => {
                self.factory.pool().record_reset(handle)?;
                Ok(observation)
            }
            Err(error) => {
                if matches!(error, RuntimeError::InstanceTrapped { .. }) {
                    self.metrics.record_trap();
                    self.factory.pool().mark_error(handle, true)?;
                }
                Err(error)
            }
        }
    }

    /// Execute a single step.
    ///
    /// Returns the step result (observation, reward, done, info).
    pub fn step(&mut self, handle: InstanceHandle, action: &Tensor) -> RuntimeResult<StepResult> {
        let timer = Timer::start();
        let fuel = self.factory.config().fuel_per_step;
        let timeout = self.factory.config().step_timeout;
        let lowered_action = bindings::lower_tensor(action);
        let result = {
            let state = self
                .env_states
                .get_mut(&handle.id)
                .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

            if !state.initialized {
                return Err(RuntimeError::execution(
                    "Environment not initialized, call reset first",
                ));
            }
            Self::prepare_operation(state, fuel, timeout, "step")?;

            match state.bindings.wasmrl_env_environment().call_step(
                &mut state.store,
                state.guest_handle,
                &lowered_action,
            ) {
                Ok(Ok(step_result)) => {
                    Self::record_fuel(state, fuel);
                    state.episode_steps += 1;
                    Ok(bindings::lift_step_result(step_result))
                }
                Ok(Err(message)) => Err(RuntimeError::execution(message)),
                Err(error) => {
                    state.store.data_mut().record_trap(&error.to_string());
                    Err(Self::map_call_error(handle, "step", timeout, error))
                }
            }
        };

        self.metrics.record_step(timer.elapsed());
        match result {
            Ok(step_result) => {
                self.factory.pool().record_step(handle)?;
                Ok(step_result)
            }
            Err(error) => {
                if matches!(error, RuntimeError::InstanceTrapped { .. }) {
                    self.metrics.record_trap();
                    self.factory.pool().mark_error(handle, true)?;
                }
                Err(error)
            }
        }
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
                    // Preserve batch shape while surfacing the per-instance error.
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
    pub fn snapshot(&mut self, handle: InstanceHandle) -> RuntimeResult<SnapshotData> {
        let fuel = self.factory.config().fuel_per_reset;
        let timeout = self.factory.config().reset_timeout;
        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        if !state.initialized {
            return Err(RuntimeError::execution(
                "Cannot snapshot uninitialized environment",
            ));
        }
        Self::prepare_operation(state, fuel, timeout, "snapshot")?;

        let snapshot = state
            .bindings
            .wasmrl_env_snapshot()
            .call_snapshot(&mut state.store, state.guest_handle)
            .map_err(|error| Self::map_call_error(handle, "snapshot", timeout, error))?
            .map_err(RuntimeError::execution)?;
        let snapshot = bindings::lift_snapshot(snapshot);
        state.latest_snapshot = Some(snapshot.clone());
        Ok(snapshot)
    }

    /// Restore an environment instance from a snapshot.
    pub fn restore(
        &mut self,
        handle: InstanceHandle,
        snapshot: &SnapshotData,
    ) -> RuntimeResult<()> {
        if !snapshot.is_compatible() {
            return Err(RuntimeError::execution(format!(
                "Incompatible snapshot version: {}",
                snapshot.version
            )));
        }

        let fuel = self.factory.config().fuel_per_reset;
        let timeout = self.factory.config().reset_timeout;
        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        Self::prepare_operation(state, fuel, timeout, "restore")?;
        let lowered = bindings::lower_snapshot(snapshot);
        state
            .bindings
            .wasmrl_env_snapshot()
            .call_restore(&mut state.store, state.guest_handle, &lowered)
            .map_err(|error| Self::map_call_error(handle, "restore", timeout, error))?
            .map_err(RuntimeError::execution)?;

        state.initialized = true;
        state.latest_snapshot = Some(snapshot.clone());
        Ok(())
    }

    /// Query the observation space declared by an environment instance.
    pub fn observation_space(&mut self, handle: InstanceHandle) -> RuntimeResult<Tensor> {
        let fuel = self.factory.config().fuel_per_reset;
        let timeout = self.factory.config().reset_timeout;
        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;
        Self::prepare_operation(state, fuel, timeout, "observation-space")?;
        state
            .bindings
            .wasmrl_env_environment()
            .call_observation_space(&mut state.store, state.guest_handle)
            .map_err(|error| Self::map_call_error(handle, "observation-space", timeout, error))?
            .map(bindings::lift_tensor)
            .map_err(RuntimeError::execution)
    }

    /// Query the action space declared by an environment instance.
    pub fn action_space(&mut self, handle: InstanceHandle) -> RuntimeResult<Tensor> {
        let fuel = self.factory.config().fuel_per_reset;
        let timeout = self.factory.config().reset_timeout;
        let state = self
            .env_states
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;
        Self::prepare_operation(state, fuel, timeout, "action-space")?;
        state
            .bindings
            .wasmrl_env_environment()
            .call_action_space(&mut state.store, state.guest_handle)
            .map_err(|error| Self::map_call_error(handle, "action-space", timeout, error))?
            .map(bindings::lift_tensor)
            .map_err(RuntimeError::execution)
    }

    /// Close an environment instance.
    pub fn close(&mut self, handle: InstanceHandle) -> RuntimeResult<()> {
        let fuel = self.factory.config().fuel_per_reset;
        let timeout = self.factory.config().reset_timeout;
        let mut state = self
            .env_states
            .remove(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;
        Self::prepare_operation(&mut state, fuel, timeout, "close")?;
        let close_result = match state
            .bindings
            .wasmrl_env_environment()
            .call_close(&mut state.store, state.guest_handle)
        {
            Ok(result) => result.map_err(RuntimeError::execution),
            Err(error) => Err(Self::map_call_error(handle, "close", timeout, error)),
        };
        self.factory.pool().release(handle)?;
        close_result
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
    fn test_batch_size_mismatch_error() {
        let err = RuntimeError::BatchSizeMismatch {
            expected: 10,
            actual: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }
}
