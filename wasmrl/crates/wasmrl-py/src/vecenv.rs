// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Vectorized environment for parallel RL training.
//! Implements Gymnasium VecEnv interface for compatibility with SB3, RLlib, etc.

use crate::config::PyEnvConfig;
use crate::error::PyWasmRLError;
use crate::spaces::{make_box_space, make_discrete_space, PySpace};
use crate::tensor::{numpy_batch_to_action_tensors, stack_observations};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use wasmrl_runtime::{EnvConfig, EnvFactory, EnvPool, EnvPoolConfig, WasmEnvInstance};
use wasmrl_wit::Tensor;

/// Vectorized WasmRL environment for parallel training.
/// 
/// This class implements the Gymnasium VecEnv interface, allowing seamless
/// integration with popular RL libraries like Stable-Baselines3.
///
/// Example:
/// ```python
/// from wasmrl_py import WasmVecEnv, EnvConfig
/// 
/// config = EnvConfig(num_envs=8, max_memory_mb=64)
/// env = WasmVecEnv("counter.wasm", config)
/// 
/// obs, info = env.reset()
/// for _ in range(1000):
///     actions = env.action_space.sample()
///     obs, rewards, dones, truncs, infos = env.step(actions)
/// ```
#[pyclass(name = "WasmVecEnv")]
pub struct PyWasmVecEnv {
    /// Environment pool for parallel execution.
    pool: EnvPool,
    /// Individual environment instances.
    instances: Vec<Box<dyn WasmEnvInstance>>,
    /// Factory for creating new instances.
    factory: Arc<EnvFactory>,
    /// Component bytes.
    component_bytes: Vec<u8>,
    /// Configuration.
    config: EnvConfig,
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
    ///
    /// Args:
    ///     component_path: Path to the .wasm component file.
    ///     config: Environment configuration (optional).
    #[new]
    #[pyo3(signature = (component_path, config=None))]
    pub fn new(py: Python<'_>, component_path: &str, config: Option<&PyEnvConfig>) -> PyResult<Self> {
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
        let num_envs = py_config.num_envs as usize;

        // Create factory
        let factory = EnvFactory::new()
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let factory = Arc::new(factory);

        // Create pool config
        let pool_config = EnvPoolConfig {
            num_envs,
            max_memory_per_env: py_config.max_memory_mb as usize * 1024 * 1024,
            enable_snapshots: true,
            auto_reset: py_config.auto_reset,
        };

        // Create pool
        let pool = EnvPool::new(pool_config.clone())
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        // Create individual instances
        let mut instances = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            let instance = factory
                .create(&component_bytes, &env_config)
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
            instances.push(instance);
        }

        // Get spec from first instance
        let spec = instances[0].spec();

        // Create observation space
        let obs_space = if spec.obs_dtype.is_floating_point() {
            make_box_space(
                py,
                -f32::INFINITY,
                f32::INFINITY,
                spec.obs_shape.iter().map(|&x| x as usize).collect(),
            )?
        } else {
            let max_val = spec
                .obs_shape
                .iter()
                .map(|&x| x as i64)
                .product::<i64>()
                .max(1);
            make_discrete_space(py, max_val, 0)?
        };

        // Create action space
        let act_space = if spec.act_dtype.is_floating_point() {
            make_box_space(
                py,
                -1.0,
                1.0,
                spec.act_shape.iter().map(|&x| x as usize).collect(),
            )?
        } else {
            let n = spec.act_shape.first().copied().unwrap_or(1) as i64;
            make_discrete_space(py, n, 0)?
        };

        Ok(Self {
            pool,
            instances,
            factory,
            component_bytes,
            config: env_config,
            num_envs,
            single_observation_space: obs_space,
            single_action_space: act_space,
            auto_reset: py_config.auto_reset,
            initialized: false,
            dones: vec![true; num_envs],
            episode_rewards: vec![0.0; num_envs],
            episode_lengths: vec![0; num_envs],
        })
    }

    /// Reset all environments.
    ///
    /// Args:
    ///     seed: Random seed for reproducibility (optional).
    ///     options: Additional options (optional).
    ///
    /// Returns:
    ///     Tuple of (observations, info_dict).
    #[pyo3(signature = (seed=None, options=None))]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
        options: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyDict>)> {
        let mut observations = Vec::with_capacity(self.num_envs);

        for (i, instance) in self.instances.iter_mut().enumerate() {
            // Optionally update seed per environment
            if let Some(base_seed) = seed {
                let env_seed = base_seed.wrapping_add(i as u64);
                // Re-create instance with new seed
                let new_config = EnvConfig {
                    seed: Some(env_seed),
                    ..self.config.clone()
                };
                *instance = self
                    .factory
                    .create(&self.component_bytes, &new_config)
                    .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
            }

            let init_result = instance
                .init()
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

            observations.push(init_result.observation);
            self.dones[i] = false;
            self.episode_rewards[i] = 0.0;
            self.episode_lengths[i] = 0;
        }

        self.initialized = true;

        let obs_array = stack_observations(py, &observations)?;
        let info = PyDict::new(py);
        info.set_item("_final_observation", py.None())?;

        Ok((obs_array, info))
    }

    /// Step all environments with given actions.
    ///
    /// Args:
    ///     actions: Numpy array of actions, shape (num_envs,) or (num_envs, action_dim).
    ///
    /// Returns:
    ///     Tuple of (observations, rewards, terminateds, truncateds, infos).
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

        let mut observations = Vec::with_capacity(self.num_envs);
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut terminateds = Vec::with_capacity(self.num_envs);
        let mut truncateds = Vec::with_capacity(self.num_envs);
        let mut final_observations: Vec<Option<Tensor>> = vec![None; self.num_envs];
        let mut final_infos: Vec<Option<String>> = vec![None; self.num_envs];

        for (i, (instance, action)) in self.instances.iter_mut().zip(action_tensors).enumerate() {
            let step_result = instance
                .step(&action)
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

            let terminated = step_result.done;
            let truncated = step_result.truncated.unwrap_or(false);
            let done = terminated || truncated;

            // Update episode stats
            self.episode_rewards[i] += step_result.reward;
            self.episode_lengths[i] += 1;

            // Handle auto-reset
            if done && self.auto_reset {
                // Store final observation before reset
                final_observations[i] = Some(step_result.observation.clone());
                final_infos[i] = step_result.info.clone();

                // Reset this environment
                let init_result = instance
                    .init()
                    .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

                observations.push(init_result.observation);

                // Reset episode stats
                self.episode_rewards[i] = 0.0;
                self.episode_lengths[i] = 0;
            } else {
                observations.push(step_result.observation);
            }

            rewards.push(step_result.reward);
            terminateds.push(terminated);
            truncateds.push(truncated);
            self.dones[i] = done;
        }

        let obs_array = stack_observations(py, &observations)?;
        let rewards_array = PyArray1::from_vec(py, rewards);
        let terminateds_array = PyArray1::from_vec(py, terminateds);
        let truncateds_array = PyArray1::from_vec(py, truncateds);

        // Build info dict
        let info = PyDict::new(py);
        
        // Add final observations for auto-reset (Gymnasium VecEnv API)
        let final_obs_list = PyList::empty(py);
        for obs in &final_observations {
            if let Some(o) = obs {
                let arr = crate::tensor::tensor_to_numpy(py, o)?;
                final_obs_list.append(arr)?;
            } else {
                final_obs_list.append(py.None())?;
            }
        }
        info.set_item("final_observation", final_obs_list)?;
        
        // Add episode rewards and lengths
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

        for (offset, &i) in indices.iter().enumerate() {
            if i >= self.num_envs {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Environment index {} out of range (0..{})",
                    i, self.num_envs
                )));
            }

            // Optionally update seed
            if let Some(base_seed) = seed {
                let env_seed = base_seed.wrapping_add(offset as u64);
                let new_config = EnvConfig {
                    seed: Some(env_seed),
                    ..self.config.clone()
                };
                self.instances[i] = self
                    .factory
                    .create(&self.component_bytes, &new_config)
                    .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
            }

            let init_result = self.instances[i]
                .init()
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

            observations.push(init_result.observation);
            self.dones[i] = false;
            self.episode_rewards[i] = 0.0;
            self.episode_lengths[i] = 0;
        }

        stack_observations(py, &observations)
    }

    /// Close all environments.
    pub fn close(&mut self) {
        self.instances.clear();
        self.initialized = false;
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

    /// Sample random actions for all environments.
    pub fn sample_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        // For now, sample discrete actions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Get action space size from first instance
        let spec = self.instances[0].spec();
        let n_actions = spec.act_shape.first().copied().unwrap_or(1) as i32;
        
        let actions: Vec<i32> = (0..self.num_envs)
            .map(|_| rng.gen_range(0..n_actions))
            .collect();
        
        Ok(PyArray1::from_vec(py, actions))
    }

    /// Take snapshots of all environments.
    pub fn snapshot_all(&self) -> PyResult<Vec<Vec<u8>>> {
        self.instances
            .iter()
            .map(|inst| {
                inst.snapshot()
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

        for (instance, snapshot) in self.instances.iter_mut().zip(snapshots) {
            instance
                .restore(&snapshot)
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
///
/// This is the main entry point for creating WasmRL vectorized environments.
///
/// Args:
///     component_path: Path to the .wasm component file.
///     num_envs: Number of parallel environments.
///     config: Additional configuration (optional).
///
/// Returns:
///     A WasmVecEnv instance.
#[pyfunction]
#[pyo3(signature = (component_path, num_envs=8, config=None))]
pub fn make_vec_env(
    py: Python<'_>,
    component_path: &str,
    num_envs: u32,
    config: Option<&PyEnvConfig>,
) -> PyResult<PyWasmVecEnv> {
    let mut py_config = config.cloned().unwrap_or_default();
    py_config.num_envs = num_envs;
    
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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = make_vec_env(py, "/nonexistent.wasm", 4, None);
            assert!(result.is_err());
        });
    }
}
