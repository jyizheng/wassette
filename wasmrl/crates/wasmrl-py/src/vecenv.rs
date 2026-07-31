// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Vectorized environment for parallel RL training.
//! Implements a Gymnasium-like VecEnv interface.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use wasmrl_runtime::{ComponentRef, EnvRuntime, InstanceHandle, WasmEnvFactory};
use wasmrl_wit::{EnvConfig, SnapshotData, Tensor};

use crate::config::PyEnvConfig;
use crate::error::PyWasmRLError;
use crate::spaces::{make_box_space, make_discrete_space, PySpace};
use crate::tensor::{numpy_batch_to_action_tensors, stack_observations};

/// Vectorized WasmRL environment for parallel training.
#[pyclass(name = "WasmVecEnv")]
pub struct PyWasmVecEnv {
    /// Runtime used for batch execution.
    runtime: EnvRuntime,
    /// Active runtime handles.
    handles: Vec<InstanceHandle>,
    /// Environment configuration.
    config: EnvConfig,
    /// Base seed used when no reset seed is provided.
    seed: u64,
    /// Number of environments.
    #[pyo3(get)]
    num_envs: usize,
    /// Observation space (single env).
    #[pyo3(get)]
    single_observation_space: Py<PySpace>,
    /// Action space (single env).
    #[pyo3(get)]
    single_action_space: Py<PySpace>,
    /// Whether auto-reset is enabled.
    #[pyo3(get)]
    auto_reset: bool,
    /// Whether environments have been initialized.
    initialized: bool,
    /// Per-environment done flags.
    dones: Vec<bool>,
    /// Per-environment episode rewards.
    episode_rewards: Vec<f64>,
    /// Per-environment episode lengths.
    episode_lengths: Vec<u64>,
}

#[pymethods]
impl PyWasmVecEnv {
    /// Create a new vectorized environment.
    #[new]
    #[pyo3(signature = (component_path, config=None))]
    pub fn new(
        py: Python<'_>,
        component_path: &str,
        config: Option<&PyEnvConfig>,
    ) -> PyResult<Self> {
        let component_bytes = std::fs::read(component_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load component: {}", e))
        })?;

        Self::from_bytes(py, component_bytes, config)
    }

    /// Create from raw component bytes.
    #[staticmethod]
    #[pyo3(signature = (component_bytes, config=None))]
    pub fn from_bytes(
        py: Python<'_>,
        component_bytes: Vec<u8>,
        config: Option<&PyEnvConfig>,
    ) -> PyResult<Self> {
        let py_config = config.cloned().unwrap_or_default();
        let env_config = py_config.to_env_config();
        let num_envs = py_config.num_envs.max(1);

        let factory = WasmEnvFactory::with_config(
            ComponentRef::from_bytes(component_bytes),
            py_config.to_policy_config(),
            py_config.to_runtime_config(),
        )
        .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let factory = Arc::new(factory);
        let handles = factory
            .spawn(num_envs)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let mut runtime = EnvRuntime::new(factory);
        for handle in &handles {
            runtime
                .init(*handle, env_config.clone())
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        }

        let (_, observation_space) =
            make_box_space(vec![1], f32::NEG_INFINITY as f64, f32::INFINITY as f64);
        let (_, action_space) = make_discrete_space(2);

        Ok(Self {
            runtime,
            handles,
            config: env_config,
            seed: py_config.seed,
            num_envs,
            single_observation_space: Py::new(py, observation_space)?,
            single_action_space: Py::new(py, action_space)?,
            auto_reset: py_config.auto_reset,
            initialized: false,
            dones: vec![true; num_envs],
            episode_rewards: vec![0.0; num_envs],
            episode_lengths: vec![0; num_envs],
        })
    }

    /// Reset all environments.
    #[pyo3(signature = (seed=None, options=None))]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
        options: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyDict>)> {
        let _ = options;
        let base_seed = seed.unwrap_or(self.seed);
        let seeds: Vec<u64> = (0..self.num_envs)
            .map(|i| base_seed.wrapping_add(i as u64))
            .collect();

        for handle in &self.handles {
            self.runtime
                .init(*handle, self.config.clone())
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        }
        let observations = self
            .runtime
            .reset_many(&self.handles, &seeds)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        self.initialized = true;
        self.dones.fill(false);
        self.episode_rewards.fill(0.0);
        self.episode_lengths.fill(0);

        let obs_array = stack_observations(py, &observations)?;
        let info = PyDict::new(py);
        info.set_item("_final_observation", py.None())?;

        Ok((obs_array, info))
    }

    /// Step all environments with given actions.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyDict>,
    )> {
        if !self.initialized {
            return Err(PyWasmRLError::NotInitialized.into());
        }

        let action_tensors = numpy_batch_to_action_tensors(py, &actions, self.num_envs)?;
        let mut batch = self
            .runtime
            .step_many(&self.handles, &action_tensors)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        let mut final_observations: Vec<Option<Tensor>> = vec![None; self.num_envs];
        let mut final_infos: Vec<Option<String>> = vec![None; self.num_envs];

        for i in 0..self.num_envs {
            let done = batch.terminated[i] || batch.truncated[i];
            self.episode_rewards[i] += batch.rewards[i];
            self.episode_lengths[i] += 1;
            self.dones[i] = done;

            if done && self.auto_reset {
                final_observations[i] = Some(batch.observations[i].clone());
                final_infos[i] = batch.infos[i].clone();

                let seed = self.seed.wrapping_add(i as u64);
                batch.observations[i] = self
                    .runtime
                    .reset(self.handles[i], seed)
                    .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
                self.episode_rewards[i] = 0.0;
                self.episode_lengths[i] = 0;
                self.dones[i] = false;
            }
        }

        let obs_array = stack_observations(py, &batch.observations)?;
        let rewards_array = PyArray1::from_vec(py, batch.rewards);
        let terminateds_array = PyArray1::from_vec(py, batch.terminated);
        let truncateds_array = PyArray1::from_vec(py, batch.truncated);

        let info = PyDict::new(py);
        let final_obs_list = PyList::empty(py);
        for obs in &final_observations {
            if let Some(o) = obs {
                final_obs_list.append(crate::tensor::tensor_to_numpy(py, o)?)?;
            } else {
                final_obs_list.append(py.None())?;
            }
        }
        info.set_item("final_observation", final_obs_list)?;
        info.set_item("final_info", final_infos)?;
        info.set_item("episode_rewards", self.episode_rewards.clone())?;
        info.set_item("episode_lengths", self.episode_lengths.clone())?;

        Ok((
            obs_array,
            rewards_array,
            terminateds_array,
            truncateds_array,
            info,
        ))
    }

    /// Reset specific environments by index.
    #[pyo3(signature = (indices, seed=None))]
    pub fn reset_envs<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mut observations = Vec::with_capacity(indices.len());
        let base_seed = seed.unwrap_or(self.seed);

        for (offset, &i) in indices.iter().enumerate() {
            if i >= self.num_envs {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Environment index {} out of range (0..{})",
                    i, self.num_envs
                )));
            }

            self.runtime
                .init(self.handles[i], self.config.clone())
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
            let obs = self
                .runtime
                .reset(self.handles[i], base_seed.wrapping_add(offset as u64))
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

            observations.push(obs);
            self.dones[i] = false;
            self.episode_rewards[i] = 0.0;
            self.episode_lengths[i] = 0;
        }

        stack_observations(py, &observations)
    }

    /// Close all environments.
    pub fn close(&mut self) {
        for handle in self.handles.drain(..) {
            let _ = self.runtime.close(handle);
        }
        self.initialized = false;
        self.dones.fill(true);
    }

    /// Get environment at specific index.
    pub fn get_env(&self, index: usize) -> PyResult<String> {
        if index >= self.num_envs {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Environment index {} out of range (0..{})",
                index, self.num_envs
            )));
        }
        Ok(format!("WasmEnv[{}]", index))
    }

    /// Sample random discrete actions for all environments.
    pub fn sample_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let mut rng_state = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let actions: Vec<i32> = (0..self.num_envs)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng_state % 2) as i32
            })
            .collect();

        Ok(PyArray1::from_vec(py, actions))
    }

    /// Take snapshots of all environments.
    pub fn snapshot_all(&self) -> PyResult<Vec<Vec<u8>>> {
        self.handles
            .iter()
            .map(|handle| {
                let snapshot = self
                    .runtime
                    .snapshot(*handle)
                    .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
                serde_json::to_vec(&snapshot)
                    .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()).into())
            })
            .collect()
    }

    /// Restore all environments from snapshots.
    pub fn restore_all(&mut self, snapshots: Vec<Vec<u8>>) -> PyResult<()> {
        if snapshots.len() != self.num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} snapshots, got {}",
                self.num_envs,
                snapshots.len()
            )));
        }

        for (handle, snapshot) in self.handles.iter().zip(snapshots) {
            let snapshot = serde_json::from_slice::<SnapshotData>(&snapshot)
                .unwrap_or_else(|_| SnapshotData::new(snapshot));
            self.runtime
                .restore(*handle, &snapshot)
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        }

        self.dones.fill(false);
        Ok(())
    }

    /// Get current episode rewards.
    #[getter]
    pub fn rewards(&self) -> Vec<f64> {
        self.episode_rewards.clone()
    }

    /// Get current episode lengths.
    #[getter]
    pub fn lengths(&self) -> Vec<u64> {
        self.episode_lengths.clone()
    }

    /// Get done flags.
    #[getter]
    pub fn done_flags(&self) -> Vec<bool> {
        self.dones.clone()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "WasmVecEnv(num_envs={}, auto_reset={}, initialized={})",
            self.num_envs, self.auto_reset, self.initialized
        )
    }

    /// Length (number of environments).
    pub fn __len__(&self) -> usize {
        self.num_envs
    }
}

/// Create a vectorized environment from component path.
#[pyfunction]
#[pyo3(signature = (component_path, num_envs=8, config=None))]
pub fn make_vec_env(
    py: Python<'_>,
    component_path: &str,
    num_envs: u32,
    config: Option<&PyEnvConfig>,
) -> PyResult<PyWasmVecEnv> {
    let mut py_config = config.cloned().unwrap_or_default();
    py_config.num_envs = num_envs as usize;

    PyWasmVecEnv::new(py, component_path, Some(&py_config))
}

/// List available environments in a directory.
#[pyfunction]
#[pyo3(signature = (directory="."))]
pub fn list_available_envs(directory: &str) -> PyResult<Vec<String>> {
    let mut envs = Vec::new();

    if let Ok(entries) = std::fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "wasm").unwrap_or(false) {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    envs.push(name.to_string());
                }
            }
        }
    }

    envs.sort();
    Ok(envs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_envs_empty_dir() {
        let envs = list_available_envs("/nonexistent").unwrap();
        assert!(envs.is_empty());
    }

    #[test]
    fn test_make_vec_env_missing_file() {
        Python::initialize();
        Python::attach(|py| {
            let result = make_vec_env(py, "/nonexistent.wasm", 4, None);
            assert!(result.is_err());
        });
    }
}
