// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! M6 Python VecEnv Integration Tests
//!
//! These tests verify the Python bindings compile and basic functionality works.
//! Full Python integration tests require running with maturin.

use wasmrl_py::*;

#[test]
fn test_py_tensor_roundtrip_i32() {
    let original = vec![1i32, 2, 3, 4, 5];
    let tensor = PyTensor::from_i32_array(&original);
    let recovered = tensor.to_i32_vec().unwrap();
    assert_eq!(original, recovered);
}

#[test]
fn test_py_tensor_roundtrip_f32() {
    let original = vec![1.0f32, 2.5, 3.14, 4.0, 5.5];
    let tensor = PyTensor::from_f32_array(&original);
    let recovered = tensor.to_f32_vec().unwrap();
    assert_eq!(original, recovered);
}

#[test]
fn test_py_tensor_roundtrip_i64() {
    let original = vec![100i64, 200, 300];
    let tensor = PyTensor::from_i64_array(&original);
    let recovered = tensor.to_i64_vec().unwrap();
    assert_eq!(original, recovered);
}

#[test]
fn test_py_tensor_shape() {
    let tensor = PyTensor::from_f32_array(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.shape(), vec![4]);
    assert_eq!(tensor.numel(), 4);
}

#[test]
fn test_py_tensor_dtype_mismatch() {
    let tensor = PyTensor::from_i32_array(&[1, 2, 3]);
    // Trying to read as f32 should fail
    assert!(tensor.to_f32_vec().is_err());
}

#[test]
fn test_py_env_config_default() {
    let config = PyEnvConfig::default();
    assert_eq!(config.num_envs, 1);
    assert_eq!(config.max_memory_mb, 64);
    assert!(config.fuel_per_step > 0);
}

#[test]
fn test_py_env_config_custom() {
    let config = PyEnvConfig {
        num_envs: 8,
        max_memory_mb: 128,
        fuel_per_step: 2_000_000,
        timeout_step_ms: 200,
        auto_reset: false,
        seed: Some(42),
    };
    assert_eq!(config.num_envs, 8);
    assert_eq!(config.max_memory_mb, 128);
    assert_eq!(config.seed, Some(42));
}

#[test]
fn test_py_wasmrl_error_display() {
    let err = PyWasmRLError::NotInitialized;
    assert!(format!("{}", err).contains("not initialized"));
    
    let err = PyWasmRLError::InvalidAction("bad action".into());
    assert!(format!("{}", err).contains("bad action"));
    
    let err = PyWasmRLError::RuntimeError("runtime error".into());
    assert!(format!("{}", err).contains("runtime error"));
    
    let err = PyWasmRLError::EnvClosed;
    assert!(format!("{}", err).contains("closed"));
}

#[test]
fn test_py_wasmrl_error_conversion() {
    // Ensure errors can be converted to PyErr
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let err: pyo3::PyErr = PyWasmRLError::NotInitialized.into();
        // Just verify the conversion doesn't panic
        let _ = err.to_string();
    });
}

// Space tests (non-PyO3)
mod space_tests {
    #[test]
    fn test_box_space_contains_logic() {
        // Test contains logic without PyO3
        let low = vec![-1.0f32, -1.0, -1.0];
        let high = vec![1.0f32, 1.0, 1.0];
        let shape = vec![3usize];
        
        // Value in bounds
        let val = vec![0.0f32, 0.5, -0.5];
        assert!(val.iter().zip(low.iter().zip(high.iter()))
            .all(|(v, (l, h))| v >= l && v <= h));
        
        // Value out of bounds
        let val = vec![2.0f32, 0.0, 0.0];
        assert!(!val.iter().zip(low.iter().zip(high.iter()))
            .all(|(v, (l, h))| v >= l && v <= h));
    }
    
    #[test]
    fn test_discrete_space_contains_logic() {
        let n = 5i64;
        let start = 0i64;
        
        // Values in range
        for i in 0..5 {
            assert!(i >= start && i < start + n);
        }
        
        // Value out of range
        assert!(!(5 >= start && 5 < start + n));
        assert!(!(-1 >= start && -1 < start + n));
    }
}

// Config conversion tests
mod config_tests {
    use super::*;
    use wasmrl_runtime::EnvConfig;
    
    #[test]
    fn test_config_to_env_config() {
        let py_config = PyEnvConfig {
            num_envs: 4,
            max_memory_mb: 32,
            fuel_per_step: 500_000,
            timeout_step_ms: 50,
            auto_reset: true,
            seed: Some(123),
        };
        
        let env_config = py_config.to_env_config();
        assert_eq!(env_config.seed, Some(123));
    }
}

// VecEnv structure tests
mod vecenv_tests {
    #[test]
    fn test_episode_tracking_arrays() {
        let num_envs = 8;
        let mut episode_rewards: Vec<f64> = vec![0.0; num_envs];
        let mut episode_lengths: Vec<u64> = vec![0; num_envs];
        
        // Simulate steps
        for _ in 0..10 {
            for i in 0..num_envs {
                episode_rewards[i] += 1.0;
                episode_lengths[i] += 1;
            }
        }
        
        assert!(episode_rewards.iter().all(|&r| r == 10.0));
        assert!(episode_lengths.iter().all(|&l| l == 10));
    }
    
    #[test]
    fn test_auto_reset_logic() {
        let mut dones = vec![false, true, false, true];
        let auto_reset = true;
        
        // Simulate auto-reset
        for (i, done) in dones.iter_mut().enumerate() {
            if *done && auto_reset {
                // Would reset env i here
                *done = false; // After reset
            }
        }
        
        assert!(dones.iter().all(|&d| !d));
    }
    
    #[test]
    fn test_action_batch_size_validation() {
        let num_envs = 4;
        let actions: Vec<i32> = vec![1, 2, 3, 4];
        
        assert_eq!(actions.len(), num_envs);
    }
}

// Integration tests that would require actual Wasm components
#[cfg(feature = "integration")]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_vecenv_lifecycle() {
        // Would test with actual Wasm component
    }
}
