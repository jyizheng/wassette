// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL Python Bindings
//!
//! This crate provides Python bindings for the WasmRL runtime, enabling
//! reinforcement learning practitioners to use WebAssembly environments
//! with standard Python RL libraries like Stable-Baselines3.
//!
//! # Features
//!
//! - **WasmVecEnv**: Gymnasium-compatible vectorized environment
//! - **Batched Operations**: Efficient `reset()` and `step()` for multiple envs
//! - **Policy Integration**: Built-in resource budgets and security
//! - **Numpy Integration**: Zero-copy tensor transfers where possible
//!
//! # Python Usage
//!
//! ```python
//! import wasmrl
//! import numpy as np
//!
//! # Create vectorized environment
//! env = wasmrl.WasmVecEnv(
//!     component_path="counter_env.wasm",
//!     num_envs=8,
//!     config={"target": 10}
//! )
//!
//! # Standard Gymnasium interface
//! obs, info = env.reset()
//! for _ in range(100):
//!     actions = np.random.randint(0, 3, size=(8,))
//!     obs, rewards, terminated, truncated, info = env.step(actions)
//!
//! env.close()
//! ```

#![warn(missing_docs)]

mod config;
mod env;
mod error;
mod spaces;
mod tensor;
mod vecenv;

pub use config::PyEnvConfig;
pub use env::PyWasmEnv;
pub use error::PyWasmRLError;
use pyo3::prelude::*;
pub use spaces::{PyBox, PyDiscrete, PySpace};
pub use tensor::PyTensor;
pub use vecenv::PyWasmVecEnv;

/// WasmRL Python module.
///
/// Provides Python bindings for running WebAssembly RL environments.
#[pymodule]
fn wasmrl_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyWasmVecEnv>()?;
    m.add_class::<PyWasmEnv>()?;
    m.add_class::<PyEnvConfig>()?;
    m.add_class::<PyBox>()?;
    m.add_class::<PyDiscrete>()?;
    m.add_class::<spaces::PyMultiDiscrete>()?;

    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "WasmRL: WebAssembly Runtime for RL Environments")?;

    // Add convenience functions
    m.add_function(wrap_pyfunction!(vecenv::make_vec_env, m)?)?;
    m.add_function(wrap_pyfunction!(vecenv::list_available_envs, m)?)?;
    m.add_function(wrap_pyfunction!(env::load_component, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_version() {
        assert!(!env!("CARGO_PKG_VERSION").is_empty());
    }
}
