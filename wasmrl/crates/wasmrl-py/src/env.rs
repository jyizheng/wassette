// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Single environment Python wrapper.

use crate::config::PyEnvConfig;
use crate::error::{PyWasmRLError, PyWasmRLResult};
use crate::spaces::{make_box_space, make_discrete_space, PySpace};
use crate::tensor::{numpy_to_action_tensor, tensor_to_numpy};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use wasmrl_runtime::{EnvConfig, EnvFactory, WasmEnvInstance};
use wasmrl_wit::Tensor;

/// Single WasmRL environment wrapper for Python.
#[pyclass(name = "WasmEnv")]
pub struct PyWasmEnv {
    /// Environment instance.
    instance: Option<Box<dyn WasmEnvInstance>>,
    /// Factory for creating new instances.
    factory: Arc<EnvFactory>,
    /// Component bytes.
    component_bytes: Vec<u8>,
    /// Configuration.
    config: EnvConfig,
    /// Observation space.
    #[pyo3(get)]
    observation_space: Py<PySpace>,
    /// Action space.
    #[pyo3(get)]
    action_space: Py<PySpace>,
    /// Whether the environment has been initialized.
    initialized: bool,
    /// Whether the last episode is done.
    done: bool,
    /// Total rewards accumulated.
    total_reward: f64,
    /// Current step count.
    step_count: u64,
}

#[pymethods]
impl PyWasmEnv {
    /// Create a new WasmEnv from component bytes.
    #[new]
    #[pyo3(signature = (component_bytes, config=None))]
    pub fn new(
        py: Python<'_>,
        component_bytes: Vec<u8>,
        config: Option<&PyEnvConfig>,
    ) -> PyResult<Self> {
        let env_config = config
            .map(|c| c.to_env_config())
            .unwrap_or_default();

        let factory = EnvFactory::new()
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let factory = Arc::new(factory);

        // Create a temporary instance to get space info
        let temp_instance = factory
            .create(&component_bytes, &env_config)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        let spec = temp_instance.spec();

        // Create observation space
        let obs_space = if spec.obs_dtype.is_floating_point() {
            make_box_space(
                py,
                -f32::INFINITY,
                f32::INFINITY,
                spec.obs_shape.iter().map(|&x| x as usize).collect(),
            )?
        } else {
            // Discrete observation
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
            // Discrete action space
            let n = spec.act_shape.first().copied().unwrap_or(1) as i64;
            make_discrete_space(py, n, 0)?
        };

        Ok(Self {
            instance: Some(temp_instance),
            factory,
            component_bytes,
            config: env_config,
            observation_space: obs_space,
            action_space: act_space,
            initialized: false,
            done: true,
            total_reward: 0.0,
            step_count: 0,
        })
    }

    /// Reset the environment and return initial observation.
    #[pyo3(signature = (seed=None, options=None))]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
        options: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyDict>)> {
        // Update seed if provided
        if let Some(s) = seed {
            self.config = EnvConfig {
                seed: Some(s),
                ..self.config.clone()
            };
        }

        // Create fresh instance
        self.instance = Some(
            self.factory
                .create(&self.component_bytes, &self.config)
                .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?,
        );

        let instance = self
            .instance
            .as_mut()
            .ok_or(PyWasmRLError::NotInitialized)?;

        let init_result = instance
            .init()
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        self.initialized = true;
        self.done = false;
        self.total_reward = 0.0;
        self.step_count = 0;

        let obs = tensor_to_numpy(py, &init_result.observation)?;
        let info = PyDict::new(py);
        info.set_item("env_id", init_result.env_id)?;

        Ok((obs, info))
    }

    /// Take a step in the environment.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        f64,
        bool,
        bool,
        Bound<'py, PyDict>,
    )> {
        if !self.initialized {
            return Err(PyWasmRLError::NotInitialized.into());
        }
        if self.done {
            return Err(PyWasmRLError::EnvClosed.into());
        }

        let action_tensor = numpy_to_action_tensor(py, &action)?;

        let instance = self
            .instance
            .as_mut()
            .ok_or(PyWasmRLError::NotInitialized)?;

        let step_result = instance
            .step(&action_tensor)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        self.total_reward += step_result.reward;
        self.step_count += 1;
        self.done = step_result.done;

        let obs = tensor_to_numpy(py, &step_result.observation)?;
        let terminated = step_result.done;
        let truncated = step_result.truncated.unwrap_or(false);

        let info = PyDict::new(py);
        if let Some(info_data) = &step_result.info {
            info.set_item("raw_info", info_data.clone())?;
        }
        info.set_item("total_reward", self.total_reward)?;
        info.set_item("step_count", self.step_count)?;

        Ok((obs, step_result.reward, terminated, truncated, info))
    }

    /// Close the environment.
    pub fn close(&mut self) {
        self.instance = None;
        self.initialized = false;
        self.done = true;
    }

    /// Get the environment spec as a dictionary.
    pub fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(ref instance) = self.instance {
            let spec = instance.spec();
            dict.set_item("env_id", &spec.env_id)?;
            dict.set_item("obs_shape", spec.obs_shape.to_vec())?;
            dict.set_item("act_shape", spec.act_shape.to_vec())?;
            dict.set_item("obs_dtype", format!("{:?}", spec.obs_dtype))?;
            dict.set_item("act_dtype", format!("{:?}", spec.act_dtype))?;
            dict.set_item("max_episode_steps", spec.max_episode_steps)?;
        }
        Ok(dict)
    }

    /// Render the environment (placeholder).
    #[pyo3(signature = (mode=None))]
    pub fn render(&self, mode: Option<&str>) -> PyResult<Option<String>> {
        // WasmRL environments don't support rendering by default
        Ok(Some(format!(
            "WasmEnv(step={}, reward={:.2}, done={})",
            self.step_count, self.total_reward, self.done
        )))
    }

    /// Property: whether the environment is closed.
    #[getter]
    pub fn is_closed(&self) -> bool {
        self.instance.is_none()
    }

    /// Property: current episode length.
    #[getter]
    pub fn episode_length(&self) -> u64 {
        self.step_count
    }

    /// Property: current episode reward.
    #[getter]
    pub fn episode_reward(&self) -> f64 {
        self.total_reward
    }

    /// Take a snapshot of the current state.
    pub fn snapshot(&self) -> PyResult<Vec<u8>> {
        let instance = self
            .instance
            .as_ref()
            .ok_or(PyWasmRLError::NotInitialized)?;

        instance
            .snapshot()
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()).into())
    }

    /// Restore from a snapshot.
    pub fn restore(&mut self, snapshot: Vec<u8>) -> PyResult<()> {
        let instance = self
            .instance
            .as_mut()
            .ok_or(PyWasmRLError::NotInitialized)?;

        instance
            .restore(&snapshot)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        self.done = false;
        Ok(())
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "WasmEnv(initialized={}, done={}, steps={}, reward={:.2})",
            self.initialized, self.done, self.step_count, self.total_reward
        )
    }
}

/// Load a component from file path.
#[pyfunction]
pub fn load_component(path: &str) -> PyResult<Vec<u8>> {
    std::fs::read(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load component: {}", e)))
}

/// Load a component from OCI registry.
#[pyfunction]
pub fn pull_component(registry: &str, name: &str, tag: &str) -> PyResult<Vec<u8>> {
    // Placeholder - would use wassette's OCI registry support
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "OCI registry support requires wassette integration",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_component_not_found() {
        let result = load_component("/nonexistent/path.wasm");
        assert!(result.is_err());
    }

    #[test]
    fn test_pull_component_not_implemented() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = pull_component("registry.io", "test", "v1");
            assert!(result.is_err());
        });
    }
}
