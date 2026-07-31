// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Environment configuration for Python.

use std::time::Duration;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use wasmrl_runtime::{PolicyConfig, RuntimeConfig};
use wasmrl_wit::EnvConfig;

/// Python-exposed environment configuration.
#[pyclass(name = "EnvConfig")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyEnvConfig {
    /// Environment-specific configuration as JSON.
    #[pyo3(get, set)]
    pub config_json: String,

    /// Number of environments (for vectorized).
    #[pyo3(get, set)]
    pub num_envs: usize,

    /// Maximum memory in MB.
    #[pyo3(get, set)]
    pub max_memory_mb: u32,

    /// Fuel per step.
    #[pyo3(get, set)]
    pub fuel_per_step: u64,

    /// Step timeout in milliseconds.
    #[pyo3(get, set)]
    pub timeout_step_ms: u64,

    /// Reset timeout in milliseconds.
    #[pyo3(get, set)]
    pub timeout_reset_ms: u64,

    /// Whether to auto-reset on done.
    #[pyo3(get, set)]
    pub auto_reset: bool,

    /// Random seed (0 = random).
    #[pyo3(get, set)]
    pub seed: u64,
}

#[pymethods]
impl PyEnvConfig {
    /// Create a new configuration with defaults.
    #[new]
    #[pyo3(signature = (
        config_json=None,
        num_envs=1,
        max_memory_mb=256,
        fuel_per_step=1_000_000,
        timeout_step_ms=100,
        timeout_reset_ms=500,
        auto_reset=true,
        seed=0
    ))]
    pub fn new(
        config_json: Option<&str>,
        num_envs: usize,
        max_memory_mb: u32,
        fuel_per_step: u64,
        timeout_step_ms: u64,
        timeout_reset_ms: u64,
        auto_reset: bool,
        seed: u64,
    ) -> Self {
        Self {
            config_json: config_json.unwrap_or("{}").to_string(),
            num_envs,
            max_memory_mb,
            fuel_per_step,
            timeout_step_ms,
            timeout_reset_ms,
            auto_reset,
            seed,
        }
    }

    /// Create from a Python dict.
    #[staticmethod]
    pub fn from_dict(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        let mut config = PyEnvConfig::default();

        if let Some(val) = dict.get_item("config")? {
            if let Ok(s) = val.extract::<String>() {
                config.config_json = s;
            } else if let Ok(d) = val.cast::<pyo3::types::PyDict>() {
                // Convert dict to JSON
                let py = dict.py();
                let json = py.import("json")?;
                let result = json.call_method1("dumps", (d,))?;
                let json_str = result.extract::<String>()?;
                config.config_json = json_str;
            }
        }

        if let Some(val) = dict.get_item("num_envs")? {
            config.num_envs = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_memory_mb")? {
            config.max_memory_mb = val.extract()?;
        }
        if let Some(val) = dict.get_item("fuel_per_step")? {
            config.fuel_per_step = val.extract()?;
        }
        if let Some(val) = dict.get_item("timeout_step_ms")? {
            config.timeout_step_ms = val.extract()?;
        }
        if let Some(val) = dict.get_item("timeout_reset_ms")? {
            config.timeout_reset_ms = val.extract()?;
        }
        if let Some(val) = dict.get_item("auto_reset")? {
            config.auto_reset = val.extract()?;
        }
        if let Some(val) = dict.get_item("seed")? {
            config.seed = val.extract()?;
        }

        Ok(config)
    }

    /// Convert to a Python dict.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("config_json", &self.config_json)?;
        dict.set_item("num_envs", self.num_envs)?;
        dict.set_item("max_memory_mb", self.max_memory_mb)?;
        dict.set_item("fuel_per_step", self.fuel_per_step)?;
        dict.set_item("timeout_step_ms", self.timeout_step_ms)?;
        dict.set_item("timeout_reset_ms", self.timeout_reset_ms)?;
        dict.set_item("auto_reset", self.auto_reset)?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_any().unbind())
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "EnvConfig(num_envs={}, max_memory_mb={}, seed={})",
            self.num_envs, self.max_memory_mb, self.seed
        )
    }
}

impl PyEnvConfig {
    /// Convert to the WIT environment config.
    pub fn to_env_config(&self) -> EnvConfig {
        EnvConfig::new(self.config_json.clone())
    }

    /// Convert to runtime configuration.
    pub fn to_runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig::new()
            .with_max_instances(self.num_envs.max(1))
            .with_max_memory_mb(self.max_memory_mb as u64)
            .with_fuel_per_step(self.fuel_per_step)
            .with_step_timeout(Duration::from_millis(self.timeout_step_ms))
            .with_reset_timeout(Duration::from_millis(self.timeout_reset_ms))
    }

    /// Convert to runtime policy configuration.
    pub fn to_policy_config(&self) -> PolicyConfig {
        PolicyConfig {
            max_memory_mb: Some(self.max_memory_mb as u64),
            fuel_per_step: Some(self.fuel_per_step),
            timeout_ms_step: Some(self.timeout_step_ms),
            timeout_ms_reset: Some(self.timeout_reset_ms),
            network_enabled: false,
            ..Default::default()
        }
    }
}

impl Default for PyEnvConfig {
    fn default() -> Self {
        Self {
            config_json: "{}".to_string(),
            num_envs: 1,
            max_memory_mb: 256,
            fuel_per_step: 1_000_000,
            timeout_step_ms: 100,
            timeout_reset_ms: 500,
            auto_reset: true,
            seed: 0,
        }
    }
}

/// Convert Python dict to JSON string.
pub fn dict_to_json(dict: Option<&Bound<'_, pyo3::types::PyAny>>) -> PyResult<String> {
    match dict {
        None => Ok("{}".to_string()),
        Some(val) => {
            if let Ok(s) = val.extract::<String>() {
                Ok(s)
            } else if val.cast::<pyo3::types::PyDict>().is_ok() {
                let py = val.py();
                let json = py.import("json")?;
                let result = json.call_method1("dumps", (val,))?;
                result.extract::<String>()
            } else {
                Ok("{}".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PyEnvConfig::default();
        assert_eq!(config.num_envs, 1);
        assert_eq!(config.max_memory_mb, 256);
        assert!(config.auto_reset);
    }

    #[test]
    fn test_config_new() {
        let config = PyEnvConfig::new(
            Some(r#"{"target": 10}"#),
            8,
            64,
            500_000,
            50,
            200,
            false,
            42,
        );
        assert_eq!(config.num_envs, 8);
        assert_eq!(config.max_memory_mb, 64);
        assert_eq!(config.seed, 42);
        assert!(!config.auto_reset);
    }

    #[test]
    fn test_config_repr() {
        let config = PyEnvConfig::default();
        let repr = config.__repr__();
        assert!(repr.contains("EnvConfig"));
        assert!(repr.contains("num_envs=1"));
    }
}
