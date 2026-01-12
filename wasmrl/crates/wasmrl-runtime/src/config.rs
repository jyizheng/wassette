// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Runtime configuration for WasmRL environments.

use std::time::Duration;

/// Configuration for the WasmRL runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum number of instances to maintain in the pool.
    pub max_instances: usize,

    /// Maximum memory per instance in bytes.
    pub max_memory_bytes: u64,

    /// Fuel limit per step (0 = unlimited).
    pub fuel_per_step: u64,

    /// Fuel limit per reset (0 = unlimited).
    pub fuel_per_reset: u64,

    /// Timeout for step operations.
    pub step_timeout: Option<Duration>,

    /// Timeout for reset operations.
    pub reset_timeout: Option<Duration>,

    /// Whether to enable epoch-based interruption.
    pub enable_epoch_interruption: bool,

    /// Number of epochs before deadline check.
    pub epoch_deadline: u64,

    /// Whether to pre-warm instances on factory creation.
    pub prewarm_instances: bool,

    /// Number of instances to pre-warm.
    pub prewarm_count: usize,
}

impl RuntimeConfig {
    /// Create a new runtime configuration with default values.
    pub fn new() -> Self {
        Self {
            max_instances: 256,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            fuel_per_step: 0,                    // unlimited
            fuel_per_reset: 0,                   // unlimited
            step_timeout: None,
            reset_timeout: None,
            enable_epoch_interruption: false,
            epoch_deadline: 1000,
            prewarm_instances: false,
            prewarm_count: 0,
        }
    }

    /// Set maximum instances.
    #[must_use]
    pub fn with_max_instances(mut self, max: usize) -> Self {
        self.max_instances = max;
        self
    }

    /// Set maximum memory per instance in megabytes.
    #[must_use]
    pub fn with_max_memory_mb(mut self, mb: u64) -> Self {
        self.max_memory_bytes = mb * 1024 * 1024;
        self
    }

    /// Set fuel limit per step.
    #[must_use]
    pub fn with_fuel_per_step(mut self, fuel: u64) -> Self {
        self.fuel_per_step = fuel;
        self
    }

    /// Set fuel limit per reset.
    #[must_use]
    pub fn with_fuel_per_reset(mut self, fuel: u64) -> Self {
        self.fuel_per_reset = fuel;
        self
    }

    /// Set step timeout.
    #[must_use]
    pub fn with_step_timeout(mut self, timeout: Duration) -> Self {
        self.step_timeout = Some(timeout);
        self
    }

    /// Set reset timeout.
    #[must_use]
    pub fn with_reset_timeout(mut self, timeout: Duration) -> Self {
        self.reset_timeout = Some(timeout);
        self
    }

    /// Enable epoch-based interruption.
    #[must_use]
    pub fn with_epoch_interruption(mut self, deadline: u64) -> Self {
        self.enable_epoch_interruption = true;
        self.epoch_deadline = deadline;
        self
    }

    /// Enable instance pre-warming.
    #[must_use]
    pub fn with_prewarming(mut self, count: usize) -> Self {
        self.prewarm_instances = true;
        self.prewarm_count = count;
        self
    }

    /// Check if fuel metering is enabled.
    pub fn fuel_enabled(&self) -> bool {
        self.fuel_per_step > 0 || self.fuel_per_reset > 0
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy configuration parsed from TOML/JSON.
#[derive(Debug, Clone, Default)]
pub struct PolicyConfig {
    /// Maximum memory in megabytes.
    pub max_memory_mb: Option<u64>,

    /// Fuel budget per step.
    pub fuel_per_step: Option<u64>,

    /// Fuel budget per batch operation.
    pub fuel_per_batch: Option<u64>,

    /// Step timeout in milliseconds.
    pub timeout_ms_step: Option<u64>,

    /// Reset timeout in milliseconds.
    pub timeout_ms_reset: Option<u64>,

    /// Allowed filesystem paths (read-only).
    pub fs_read_paths: Vec<String>,

    /// Allowed filesystem paths (read-write).
    pub fs_write_paths: Vec<String>,

    /// Whether network access is allowed.
    pub network_enabled: bool,
}

impl PolicyConfig {
    /// Create an empty policy config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply this policy to a runtime config.
    pub fn apply_to(&self, config: &mut RuntimeConfig) {
        if let Some(mb) = self.max_memory_mb {
            config.max_memory_bytes = mb * 1024 * 1024;
        }
        if let Some(fuel) = self.fuel_per_step {
            config.fuel_per_step = fuel;
        }
        if let Some(ms) = self.timeout_ms_step {
            config.step_timeout = Some(Duration::from_millis(ms));
        }
        if let Some(ms) = self.timeout_ms_reset {
            config.reset_timeout = Some(Duration::from_millis(ms));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_new() {
        let config = RuntimeConfig::new();
        assert_eq!(config.max_instances, 256);
        assert_eq!(config.max_memory_bytes, 512 * 1024 * 1024);
        assert_eq!(config.fuel_per_step, 0);
        assert!(!config.fuel_enabled());
    }

    #[test]
    fn test_runtime_config_builder() {
        let config = RuntimeConfig::new()
            .with_max_instances(128)
            .with_max_memory_mb(256)
            .with_fuel_per_step(1_000_000)
            .with_step_timeout(Duration::from_secs(1));

        assert_eq!(config.max_instances, 128);
        assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(config.fuel_per_step, 1_000_000);
        assert!(config.fuel_enabled());
        assert_eq!(config.step_timeout, Some(Duration::from_secs(1)));
    }

    #[test]
    fn test_runtime_config_prewarming() {
        let config = RuntimeConfig::new().with_prewarming(16);
        assert!(config.prewarm_instances);
        assert_eq!(config.prewarm_count, 16);
    }

    #[test]
    fn test_runtime_config_epoch() {
        let config = RuntimeConfig::new().with_epoch_interruption(500);
        assert!(config.enable_epoch_interruption);
        assert_eq!(config.epoch_deadline, 500);
    }

    #[test]
    fn test_policy_config_apply() {
        let policy = PolicyConfig {
            max_memory_mb: Some(128),
            fuel_per_step: Some(500_000),
            timeout_ms_step: Some(100),
            ..Default::default()
        };

        let mut config = RuntimeConfig::new();
        policy.apply_to(&mut config);

        assert_eq!(config.max_memory_bytes, 128 * 1024 * 1024);
        assert_eq!(config.fuel_per_step, 500_000);
        assert_eq!(config.step_timeout, Some(Duration::from_millis(100)));
    }

    #[test]
    fn test_policy_config_default() {
        let policy = PolicyConfig::new();
        assert!(policy.max_memory_mb.is_none());
        assert!(!policy.network_enabled);
        assert!(policy.fs_read_paths.is_empty());
    }
}
