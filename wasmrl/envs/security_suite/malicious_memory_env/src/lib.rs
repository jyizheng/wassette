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
