// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Single environment Python wrapper.

use std::sync::Arc;

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use wasmrl_runtime::{ComponentRef, EnvRuntime, InstanceHandle, WasmEnvFactory};
use wasmrl_wit::{EnvConfig, SnapshotData};

use crate::config::PyEnvConfig;
use crate::error::PyWasmRLError;
use crate::spaces::{make_box_space, make_discrete_space, PySpace};
use crate::tensor::{numpy_to_action_tensor, tensor_to_numpy};

/// Single WasmRL environment wrapper for Python.
#[pyclass(name = "WasmEnv")]
pub struct PyWasmEnv {
    /// Runtime used for environment calls.
    runtime: EnvRuntime,
    /// Active instance handle.
    handle: Option<InstanceHandle>,
    /// Environment configuration.
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

impl PyWasmEnv {
    fn handle(&self) -> PyResult<InstanceHandle> {
        self.handle.ok_or_else(|| PyWasmRLError::EnvClosed.into())
    }
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
        let py_config = config.cloned().unwrap_or_default();
        let env_config = py_config.to_env_config();
        let factory = WasmEnvFactory::with_config(
            ComponentRef::from_bytes(component_bytes),
            py_config.to_policy_config(),
            py_config.to_runtime_config(),
        )
        .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let factory = Arc::new(factory);
        let handle = factory
            .spawn_one()
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let mut runtime = EnvRuntime::new(factory);
        runtime
            .init(handle, env_config.clone())
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        let (_, observation_space) =
            make_box_space(vec![1], f32::NEG_INFINITY as f64, f32::INFINITY as f64);
        let (_, action_space) = make_discrete_space(2);

        Ok(Self {
            runtime,
            handle: Some(handle),
            config: env_config,
            observation_space: Py::new(py, observation_space)?,
            action_space: Py::new(py, action_space)?,
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
        let _ = options;
        let handle = self.handle()?;
        self.runtime
            .init(handle, self.config.clone())
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        let observation = self
            .runtime
            .reset(handle, seed.unwrap_or(0))
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        self.initialized = true;
        self.done = false;
        self.total_reward = 0.0;
        self.step_count = 0;

        let obs = tensor_to_numpy(py, &observation)?;
        let info = PyDict::new(py);
        info.set_item("env_id", handle.id)?;

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
        let handle = self.handle()?;
        let step_result = self
            .runtime
            .step(handle, &action_tensor)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;

        self.total_reward += step_result.reward;
        self.step_count += 1;
        self.done = step_result.done();

        let obs = tensor_to_numpy(py, &step_result.observation)?;
        let info = PyDict::new(py);
        if let Some(info_data) = &step_result.info {
            info.set_item("raw_info", info_data.clone())?;
        }
        info.set_item("total_reward", self.total_reward)?;
        info.set_item("step_count", self.step_count)?;

        Ok((
            obs,
            step_result.reward,
            step_result.terminated,
            step_result.truncated,
            info,
        ))
    }

    /// Close the environment.
    pub fn close(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.runtime.close(handle);
        }
        self.initialized = false;
        self.done = true;
    }

    /// Get the environment spec as a dictionary.
    pub fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(handle) = self.handle {
            dict.set_item("env_id", handle.id)?;
            dict.set_item("obs_shape", vec![1usize])?;
            dict.set_item("act_shape", vec![1usize])?;
            dict.set_item("obs_dtype", "Float32")?;
            dict.set_item("act_dtype", "Int32")?;
            dict.set_item("max_episode_steps", py.None())?;
        }
        Ok(dict)
    }

    /// Render the environment.
    #[pyo3(signature = (mode=None))]
    pub fn render(&self, mode: Option<&str>) -> PyResult<Option<String>> {
        let _ = mode;
        Ok(Some(format!(
            "WasmEnv(step={}, reward={:.2}, done={})",
            self.step_count, self.total_reward, self.done
        )))
    }

    /// Property: whether the environment is closed.
    #[getter]
    pub fn is_closed(&self) -> bool {
        self.handle.is_none()
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
        let handle = self.handle()?;
        let snapshot = self
            .runtime
            .snapshot(handle)
            .map_err(|e| PyWasmRLError::RuntimeError(e.to_string()))?;
        serde_json::to_vec(&snapshot).map_err(|e| PyWasmRLError::RuntimeError(e.to_string()).into())
    }

    /// Restore from a snapshot.
    pub fn restore(&mut self, snapshot: Vec<u8>) -> PyResult<()> {
        let handle = self.handle()?;
        let snapshot = serde_json::from_slice::<SnapshotData>(&snapshot)
            .unwrap_or_else(|_| SnapshotData::new(snapshot));
        self.runtime
            .restore(handle, &snapshot)
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
    std::fs::read(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load component: {}", e))
    })
}

/// Load a component from OCI registry.
#[pyfunction]
pub fn pull_component(_registry: &str, _name: &str, _tag: &str) -> PyResult<Vec<u8>> {
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
        Python::initialize();
        Python::attach(|_py| {
            let result = pull_component("registry.io", "test", "v1");
            assert!(result.is_err());
        });
    }
}
