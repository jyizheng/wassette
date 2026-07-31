// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Malicious Loop Environment - Security test for infinite loops.
//!
//! This environment is intentionally malicious and will attempt to:
//! - Run infinite loops in step()
//! - Consume excessive CPU time
//!
//! It is used to test that the WasmRL runtime properly enforces:
//! - Fuel limits (instruction counting)
//! - Timeout limits
//! - Epoch-based interruption
//!
//! # WARNING
//!
//! This environment should NEVER be run without proper resource limits.
//! It will hang indefinitely if fuel/timeout is not enforced.

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use wasmrl_sdk_rust::TensorEncoder;

/// Configuration for malicious loop behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaliciousConfig {
    /// Which operation should trigger the infinite loop.
    #[serde(default)]
    pub loop_on: LoopTrigger,
    /// Number of iterations before looping (for testing partial execution).
    #[serde(default)]
    pub iterations_before_loop: u32,
}

/// When to trigger the infinite loop.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopTrigger {
    /// Loop on init()
    Init,
    /// Loop on reset()
    Reset,
    /// Loop on step() (default)
    #[default]
    Step,
    /// Loop on snapshot()
    Snapshot,
}

impl Default for MaliciousConfig {
    fn default() -> Self {
        Self {
            loop_on: LoopTrigger::Step,
            iterations_before_loop: 0,
        }
    }
}

/// Malicious environment that intentionally loops forever.
#[derive(Debug)]
pub struct MaliciousLoopEnv {
    /// Configuration.
    config: MaliciousConfig,
    /// Whether environment is initialized.
    initialized: bool,
    /// Step counter.
    step_count: u32,
}

impl MaliciousLoopEnv {
    /// Create a new malicious environment.
    pub fn new() -> Self {
        Self {
            config: MaliciousConfig::default(),
            initialized: false,
            step_count: 0,
        }
    }

    /// Run an infinite loop (will consume fuel/time until interrupted).
    #[inline(never)]
    fn infinite_loop(&self) {
        let mut counter: u64 = 0;
        loop {
            counter = counter.wrapping_add(1);
            // Prevent optimization from removing the loop
            if counter == u64::MAX {
                counter = 0;
            }
            // This volatile-like operation prevents the loop from being optimized away
            std::hint::black_box(counter);
        }
    }

    /// Initialize environment with configuration.
    pub fn init(&mut self, config_json: &str) -> Result<u64, String> {
        self.config = if config_json.is_empty() || config_json == "{}" {
            MaliciousConfig::default()
        } else {
            serde_json::from_str(config_json).map_err(|e| format!("init failed: {}", e))?
        };

        if matches!(self.config.loop_on, LoopTrigger::Init) {
            self.infinite_loop();
        }

        self.initialized = true;
        Ok(1)
    }

    /// Reset environment (may loop infinitely).
    pub fn reset(&mut self, _seed: u64) -> Result<Vec<u8>, String> {
        if !self.initialized {
            return Err("reset failed: not initialized".to_string());
        }

        self.step_count = 0;

        if matches!(self.config.loop_on, LoopTrigger::Reset) {
            self.infinite_loop();
        }

        Ok(self.get_observation())
    }

    /// Step environment (may loop infinitely).
    pub fn step(&mut self, _action: &[u8]) -> Result<StepOutput, String> {
        if !self.initialized {
            return Err("step failed: not initialized".to_string());
        }

        self.step_count += 1;

        // Check if we should start looping
        if matches!(self.config.loop_on, LoopTrigger::Step)
            && self.step_count > self.config.iterations_before_loop
        {
            self.infinite_loop();
        }

        Ok(StepOutput {
            observation: self.get_observation(),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: None,
        })
    }

    /// Get observation.
    fn get_observation(&self) -> Vec<u8> {
        TensorEncoder::encode_f32(&[self.step_count as f32])
    }

    /// Snapshot (may loop infinitely).
    pub fn snapshot(&self) -> Result<Vec<u8>, String> {
        if matches!(self.config.loop_on, LoopTrigger::Snapshot) {
            self.infinite_loop();
        }
        Ok(vec![0, 0, 0, 0])
    }

    /// Restore from snapshot.
    pub fn restore(&mut self, _snapshot: &[u8]) -> Result<(), String> {
        Ok(())
    }

    /// Close environment.
    pub fn close(&mut self) -> Result<(), String> {
        self.initialized = false;
        Ok(())
    }
}

impl Default for MaliciousLoopEnv {
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
    fn test_malicious_env_new() {
        let env = MaliciousLoopEnv::new();
        assert!(!env.initialized);
    }

    #[test]
    fn test_malicious_env_init() {
        let mut env = MaliciousLoopEnv::new();
        // Default config loops on step, so init should succeed
        let result = env.init("{}");
        assert!(result.is_ok());
        assert!(env.initialized);
    }

    #[test]
    fn test_malicious_env_config_parse() {
        let mut env = MaliciousLoopEnv::new();
        let result = env.init(r#"{"loop_on": "reset", "iterations_before_loop": 5}"#);
        assert!(result.is_ok());
        assert!(matches!(env.config.loop_on, LoopTrigger::Reset));
        assert_eq!(env.config.iterations_before_loop, 5);
    }

    // NOTE: We cannot test the actual infinite loop behavior in unit tests
    // because it would hang forever. These tests are for the non-looping paths.

    #[test]
    fn test_malicious_env_reset_no_loop() {
        let mut env = MaliciousLoopEnv::new();
        env.init(r#"{"loop_on": "step"}"#).unwrap();
        // Reset should succeed since loop_on is "step"
        let result = env.reset(42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_malicious_env_iterations_before_loop() {
        let mut env = MaliciousLoopEnv::new();
        env.init(r#"{"loop_on": "step", "iterations_before_loop": 3}"#)
            .unwrap();
        env.reset(42).unwrap();

        // First 3 steps should succeed
        for i in 0..3 {
            let action = vec![0, 0, 0, 0];
            let result = env.step(&action);
            assert!(
                result.is_ok(),
                "Step {} should succeed before loop triggers",
                i
            );
        }

        // Step 4 would trigger the loop, but we can't test that
    }

    #[test]
    fn test_malicious_env_not_initialized() {
        let mut env = MaliciousLoopEnv::new();
        let result = env.reset(42);
        assert!(result.is_err());

        let action = vec![0, 0, 0, 0];
        let result = env.step(&action);
        assert!(result.is_err());
    }
}
