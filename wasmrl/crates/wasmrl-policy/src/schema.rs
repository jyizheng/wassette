// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Policy schema definitions and parsing.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Complete policy configuration for WasmRL environments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Memory limits.
    #[serde(default)]
    pub memory: MemoryLimit,
    /// Fuel/instruction budgets.
    #[serde(default)]
    pub fuel: FuelBudget,
    /// Timeout configurations.
    #[serde(default)]
    pub timeout: TimeoutConfig,
    /// WASI capability configuration.
    #[serde(default)]
    pub wasi: WasiConfig,
    /// Additional capability settings.
    #[serde(default)]
    pub capabilities: CapabilityConfig,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            memory: MemoryLimit::default(),
            fuel: FuelBudget::default(),
            timeout: TimeoutConfig::default(),
            wasi: WasiConfig::default(),
            capabilities: CapabilityConfig::default(),
        }
    }
}

impl Policy {
    /// Create a new policy with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse policy from TOML string.
    pub fn from_toml(toml_str: &str) -> crate::PolicyResult<Self> {
        toml::from_str(toml_str).map_err(|e| crate::PolicyError::ParseError(e.to_string()))
    }

    /// Parse policy from JSON string.
    pub fn from_json(json_str: &str) -> crate::PolicyResult<Self> {
        serde_json::from_str(json_str).map_err(|e| crate::PolicyError::ParseError(e.to_string()))
    }

    /// Load policy from file (auto-detects TOML or JSON).
    pub fn from_file(path: &std::path::Path) -> crate::PolicyResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::PolicyError::IoError(e.to_string()))?;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext.to_lowercase().as_str() {
            "json" => Self::from_json(&content),
            "toml" | _ => Self::from_toml(&content),
        }
    }

    /// Serialize policy to TOML string.
    pub fn to_toml(&self) -> crate::PolicyResult<String> {
        toml::to_string_pretty(self).map_err(|e| crate::PolicyError::SerializeError(e.to_string()))
    }

    /// Serialize policy to JSON string.
    pub fn to_json(&self) -> crate::PolicyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::PolicyError::SerializeError(e.to_string()))
    }

    /// Validate the policy configuration.
    pub fn validate(&self) -> crate::PolicyResult<()> {
        // Memory validation
        if self.memory.max_mb == 0 {
            return Err(crate::PolicyError::ValidationError(
                "max_memory_mb must be > 0".to_string(),
            ));
        }
        if self.memory.initial_mb > self.memory.max_mb {
            return Err(crate::PolicyError::ValidationError(
                "initial_mb cannot exceed max_mb".to_string(),
            ));
        }

        // Fuel validation
        if self.fuel.per_step == 0 && self.fuel.enabled {
            return Err(crate::PolicyError::ValidationError(
                "fuel_per_step must be > 0 when fuel is enabled".to_string(),
            ));
        }

        // Timeout validation
        if self.timeout.step_ms == 0 && self.timeout.enabled {
            return Err(crate::PolicyError::ValidationError(
                "timeout_step_ms must be > 0 when timeouts enabled".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if network access is allowed.
    pub fn allows_network(&self) -> bool {
        self.wasi.network
    }

    /// Check if a filesystem path is readable.
    pub fn allows_read(&self, path: &str) -> bool {
        self.wasi
            .filesystem_read
            .iter()
            .any(|allowed| path.starts_with(allowed.to_string_lossy().as_ref()))
    }

    /// Check if a filesystem path is writable.
    pub fn allows_write(&self, path: &str) -> bool {
        self.wasi
            .filesystem_write
            .iter()
            .any(|allowed| path.starts_with(allowed.to_string_lossy().as_ref()))
    }
}

/// Memory limits configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimit {
    /// Maximum memory in megabytes.
    #[serde(default = "default_max_mb")]
    pub max_mb: u32,
    /// Initial memory allocation in megabytes.
    #[serde(default = "default_initial_mb")]
    pub initial_mb: u32,
    /// Maximum linear memory pages (64KB each).
    #[serde(default)]
    pub max_pages: Option<u32>,
    /// Enable memory limiting.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_max_mb() -> u32 {
    256
}
fn default_initial_mb() -> u32 {
    16
}
fn default_true() -> bool {
    true
}

impl Default for MemoryLimit {
    fn default() -> Self {
        Self {
            max_mb: default_max_mb(),
            initial_mb: default_initial_mb(),
            max_pages: None,
            enabled: true,
        }
    }
}

impl MemoryLimit {
    /// Create a new memory limit with specified max.
    pub fn new(max_mb: u32) -> Self {
        Self {
            max_mb,
            ..Default::default()
        }
    }

    /// Convert max_mb to bytes.
    pub fn max_bytes(&self) -> u64 {
        (self.max_mb as u64) * 1024 * 1024
    }

    /// Convert max_mb to Wasm pages (64KB each).
    pub fn max_wasm_pages(&self) -> u32 {
        self.max_pages.unwrap_or_else(|| (self.max_mb * 1024) / 64)
    }
}

/// Fuel/instruction budget configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuelBudget {
    /// Fuel allocated per step call.
    #[serde(default = "default_fuel_per_step")]
    pub per_step: u64,
    /// Fuel allocated per reset call.
    #[serde(default = "default_fuel_per_reset")]
    pub per_reset: u64,
    /// Fuel allocated per batch call.
    #[serde(default = "default_fuel_per_batch")]
    pub per_batch: u64,
    /// Fuel allocated for init.
    #[serde(default = "default_fuel_per_init")]
    pub per_init: u64,
    /// Fuel allocated for snapshot.
    #[serde(default = "default_fuel_per_snapshot")]
    pub per_snapshot: u64,
    /// Fuel allocated for restore.
    #[serde(default = "default_fuel_per_restore")]
    pub per_restore: u64,
    /// Enable fuel metering.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_fuel_per_step() -> u64 {
    1_000_000
}
fn default_fuel_per_reset() -> u64 {
    5_000_000
}
fn default_fuel_per_batch() -> u64 {
    10_000_000
}
fn default_fuel_per_init() -> u64 {
    10_000_000
}
fn default_fuel_per_snapshot() -> u64 {
    2_000_000
}
fn default_fuel_per_restore() -> u64 {
    2_000_000
}

impl Default for FuelBudget {
    fn default() -> Self {
        Self {
            per_step: default_fuel_per_step(),
            per_reset: default_fuel_per_reset(),
            per_batch: default_fuel_per_batch(),
            per_init: default_fuel_per_init(),
            per_snapshot: default_fuel_per_snapshot(),
            per_restore: default_fuel_per_restore(),
            enabled: true,
        }
    }
}

impl FuelBudget {
    /// Create a new fuel budget with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom per-step fuel.
    pub fn with_per_step(per_step: u64) -> Self {
        Self {
            per_step,
            ..Default::default()
        }
    }

    /// Disable fuel metering.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Timeout configuration in milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Timeout for step operations (ms).
    #[serde(default = "default_timeout_step")]
    pub step_ms: u64,
    /// Timeout for reset operations (ms).
    #[serde(default = "default_timeout_reset")]
    pub reset_ms: u64,
    /// Timeout for batch operations (ms).
    #[serde(default = "default_timeout_batch")]
    pub batch_ms: u64,
    /// Timeout for init operations (ms).
    #[serde(default = "default_timeout_init")]
    pub init_ms: u64,
    /// Timeout for snapshot/restore (ms).
    #[serde(default = "default_timeout_snapshot")]
    pub snapshot_ms: u64,
    /// Enable timeout enforcement.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_timeout_step() -> u64 {
    100
}
fn default_timeout_reset() -> u64 {
    500
}
fn default_timeout_batch() -> u64 {
    1000
}
fn default_timeout_init() -> u64 {
    5000
}
fn default_timeout_snapshot() -> u64 {
    200
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            step_ms: default_timeout_step(),
            reset_ms: default_timeout_reset(),
            batch_ms: default_timeout_batch(),
            init_ms: default_timeout_init(),
            snapshot_ms: default_timeout_snapshot(),
            enabled: true,
        }
    }
}

impl TimeoutConfig {
    /// Create a new timeout config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get step timeout as Duration.
    pub fn step_duration(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.step_ms)
    }

    /// Get reset timeout as Duration.
    pub fn reset_duration(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.reset_ms)
    }

    /// Get batch timeout as Duration.
    pub fn batch_duration(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.batch_ms)
    }

    /// Disable all timeouts.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// WASI capability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasiConfig {
    /// Paths allowed for reading.
    #[serde(default)]
    pub filesystem_read: Vec<PathBuf>,
    /// Paths allowed for writing.
    #[serde(default)]
    pub filesystem_write: Vec<PathBuf>,
    /// Allow network access.
    #[serde(default)]
    pub network: bool,
    /// Allowed environment variables.
    #[serde(default)]
    pub env_vars: Vec<String>,
    /// Environment variable values to inject.
    #[serde(default)]
    pub env_values: std::collections::HashMap<String, String>,
    /// Command line arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Allow clock/time operations.
    #[serde(default = "default_true")]
    pub clock: bool,
    /// Allow random number generation.
    #[serde(default = "default_true")]
    pub random: bool,
    /// Inherit stdio from host.
    #[serde(default)]
    pub inherit_stdio: bool,
}

impl Default for WasiConfig {
    fn default() -> Self {
        Self {
            filesystem_read: Vec::new(),
            filesystem_write: Vec::new(),
            network: false, // Deny by default
            env_vars: Vec::new(),
            env_values: std::collections::HashMap::new(),
            args: Vec::new(),
            clock: true,
            random: true,
            inherit_stdio: false,
        }
    }
}

impl WasiConfig {
    /// Create a new WASI config with defaults (deny-by-default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a permissive config for testing.
    pub fn permissive() -> Self {
        Self {
            network: true,
            inherit_stdio: true,
            ..Default::default()
        }
    }

    /// Add a read-only path.
    pub fn add_read_path(&mut self, path: impl Into<PathBuf>) {
        self.filesystem_read.push(path.into());
    }

    /// Add a writable path.
    pub fn add_write_path(&mut self, path: impl Into<PathBuf>) {
        self.filesystem_write.push(path.into());
    }

    /// Set an environment variable.
    pub fn set_env(&mut self, key: &str, value: &str) {
        self.env_values.insert(key.to_string(), value.to_string());
    }
}

/// Additional capability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityConfig {
    /// Allow threading/atomics.
    #[serde(default)]
    pub threading: bool,
    /// Allow SIMD instructions.
    #[serde(default = "default_true")]
    pub simd: bool,
    /// Allow component model.
    #[serde(default = "default_true")]
    pub component_model: bool,
    /// Allow multi-memory.
    #[serde(default)]
    pub multi_memory: bool,
    /// Allow bulk memory operations.
    #[serde(default = "default_true")]
    pub bulk_memory: bool,
    /// Custom capability flags.
    #[serde(default)]
    pub custom: std::collections::HashMap<String, bool>,
}

impl Default for CapabilityConfig {
    fn default() -> Self {
        Self {
            threading: false,
            simd: true,
            component_model: true,
            multi_memory: false,
            bulk_memory: true,
            custom: std::collections::HashMap::new(),
        }
    }
}

impl CapabilityConfig {
    /// Create a new capability config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a custom capability is allowed.
    pub fn allows(&self, capability: &str) -> bool {
        self.custom.get(capability).copied().unwrap_or(false)
    }

    /// Set a custom capability.
    pub fn set(&mut self, capability: &str, allowed: bool) {
        self.custom.insert(capability.to_string(), allowed);
    }
}

/// Builder for constructing policies fluently.
#[derive(Debug, Clone, Default)]
pub struct PolicyBuilder {
    policy: Policy,
}

impl PolicyBuilder {
    /// Create a new policy builder with defaults.
    pub fn new() -> Self {
        Self {
            policy: Policy::default(),
        }
    }

    /// Set maximum memory in megabytes.
    pub fn max_memory_mb(mut self, mb: u32) -> Self {
        self.policy.memory.max_mb = mb;
        self
    }

    /// Set initial memory in megabytes.
    pub fn initial_memory_mb(mut self, mb: u32) -> Self {
        self.policy.memory.initial_mb = mb;
        self
    }

    /// Set fuel per step.
    pub fn fuel_per_step(mut self, fuel: u64) -> Self {
        self.policy.fuel.per_step = fuel;
        self
    }

    /// Set fuel per reset.
    pub fn fuel_per_reset(mut self, fuel: u64) -> Self {
        self.policy.fuel.per_reset = fuel;
        self
    }

    /// Set fuel per batch.
    pub fn fuel_per_batch(mut self, fuel: u64) -> Self {
        self.policy.fuel.per_batch = fuel;
        self
    }

    /// Disable fuel metering.
    pub fn disable_fuel(mut self) -> Self {
        self.policy.fuel.enabled = false;
        self
    }

    /// Set step timeout in milliseconds.
    pub fn timeout_step_ms(mut self, ms: u64) -> Self {
        self.policy.timeout.step_ms = ms;
        self
    }

    /// Set reset timeout in milliseconds.
    pub fn timeout_reset_ms(mut self, ms: u64) -> Self {
        self.policy.timeout.reset_ms = ms;
        self
    }

    /// Set batch timeout in milliseconds.
    pub fn timeout_batch_ms(mut self, ms: u64) -> Self {
        self.policy.timeout.batch_ms = ms;
        self
    }

    /// Disable timeouts.
    pub fn disable_timeouts(mut self) -> Self {
        self.policy.timeout.enabled = false;
        self
    }

    /// Allow reading from specified paths.
    pub fn allow_filesystem_read(mut self, paths: &[&str]) -> Self {
        self.policy.wasi.filesystem_read = paths.iter().map(PathBuf::from).collect();
        self
    }

    /// Allow writing to specified paths.
    pub fn allow_filesystem_write(mut self, paths: &[&str]) -> Self {
        self.policy.wasi.filesystem_write = paths.iter().map(PathBuf::from).collect();
        self
    }

    /// Allow network access.
    pub fn allow_network(mut self) -> Self {
        self.policy.wasi.network = true;
        self
    }

    /// Deny network access.
    pub fn deny_network(mut self) -> Self {
        self.policy.wasi.network = false;
        self
    }

    /// Set environment variables.
    pub fn env_vars(mut self, vars: &[&str]) -> Self {
        self.policy.wasi.env_vars = vars.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Inherit stdio from host.
    pub fn inherit_stdio(mut self) -> Self {
        self.policy.wasi.inherit_stdio = true;
        self
    }

    /// Build the policy.
    pub fn build(self) -> Policy {
        self.policy
    }

    /// Build and validate the policy.
    pub fn build_validated(self) -> crate::PolicyResult<Policy> {
        let policy = self.build();
        policy.validate()?;
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_limit_default() {
        let limit = MemoryLimit::default();
        assert_eq!(limit.max_mb, 256);
        assert_eq!(limit.initial_mb, 16);
        assert!(limit.enabled);
    }

    #[test]
    fn test_memory_limit_max_bytes() {
        let limit = MemoryLimit::new(64);
        assert_eq!(limit.max_bytes(), 64 * 1024 * 1024);
    }

    #[test]
    fn test_memory_limit_wasm_pages() {
        let limit = MemoryLimit::new(64);
        // 64 MB = 64 * 1024 KB = 65536 KB
        // 65536 KB / 64 KB per page = 1024 pages
        assert_eq!(limit.max_wasm_pages(), 1024);
    }

    #[test]
    fn test_fuel_budget_default() {
        let fuel = FuelBudget::default();
        assert_eq!(fuel.per_step, 1_000_000);
        assert_eq!(fuel.per_reset, 5_000_000);
        assert!(fuel.enabled);
    }

    #[test]
    fn test_fuel_budget_disabled() {
        let fuel = FuelBudget::disabled();
        assert!(!fuel.enabled);
    }

    #[test]
    fn test_timeout_config_default() {
        let timeout = TimeoutConfig::default();
        assert_eq!(timeout.step_ms, 100);
        assert_eq!(timeout.reset_ms, 500);
        assert!(timeout.enabled);
    }

    #[test]
    fn test_timeout_duration() {
        let timeout = TimeoutConfig::default();
        assert_eq!(
            timeout.step_duration(),
            std::time::Duration::from_millis(100)
        );
    }

    #[test]
    fn test_wasi_config_default() {
        let wasi = WasiConfig::default();
        assert!(!wasi.network); // Deny by default
        assert!(wasi.filesystem_read.is_empty());
        assert!(wasi.clock);
        assert!(wasi.random);
    }

    #[test]
    fn test_wasi_config_add_paths() {
        let mut wasi = WasiConfig::new();
        wasi.add_read_path("/data");
        wasi.add_write_path("/tmp");
        assert_eq!(wasi.filesystem_read.len(), 1);
        assert_eq!(wasi.filesystem_write.len(), 1);
    }

    #[test]
    fn test_policy_builder() {
        let policy = PolicyBuilder::new()
            .max_memory_mb(128)
            .fuel_per_step(500_000)
            .timeout_step_ms(50)
            .allow_filesystem_read(&["/data"])
            .deny_network()
            .build();

        assert_eq!(policy.memory.max_mb, 128);
        assert_eq!(policy.fuel.per_step, 500_000);
        assert_eq!(policy.timeout.step_ms, 50);
        assert!(!policy.wasi.network);
    }

    #[test]
    fn test_policy_validation_success() {
        let policy = Policy::default();
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_policy_validation_zero_memory() {
        let mut policy = Policy::default();
        policy.memory.max_mb = 0;
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_policy_validation_initial_exceeds_max() {
        let mut policy = Policy::default();
        policy.memory.initial_mb = 512;
        policy.memory.max_mb = 256;
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_policy_allows_read() {
        let policy = PolicyBuilder::new()
            .allow_filesystem_read(&["/data", "/models"])
            .build();

        assert!(policy.allows_read("/data/file.txt"));
        assert!(policy.allows_read("/models/model.bin"));
        assert!(!policy.allows_read("/etc/passwd"));
    }

    #[test]
    fn test_policy_allows_write() {
        let policy = PolicyBuilder::new()
            .allow_filesystem_write(&["/tmp"])
            .build();

        assert!(policy.allows_write("/tmp/output.txt"));
        assert!(!policy.allows_write("/data/file.txt"));
    }

    #[test]
    fn test_policy_from_toml_complete() {
        let toml = r#"
            [memory]
            max_mb = 64
            initial_mb = 8

            [fuel]
            per_step = 500000
            per_reset = 2000000
            enabled = true

            [timeout]
            step_ms = 50
            reset_ms = 200
            enabled = true

            [wasi]
            filesystem_read = ["/data"]
            network = false
            clock = true
        "#;

        let policy = Policy::from_toml(toml).unwrap();
        assert_eq!(policy.memory.max_mb, 64);
        assert_eq!(policy.fuel.per_step, 500_000);
        assert_eq!(policy.timeout.step_ms, 50);
        assert_eq!(policy.wasi.filesystem_read.len(), 1);
        assert!(!policy.wasi.network);
    }

    #[test]
    fn test_policy_roundtrip_toml() {
        let original = PolicyBuilder::new()
            .max_memory_mb(128)
            .fuel_per_step(1_000_000)
            .timeout_step_ms(100)
            .build();

        let toml = original.to_toml().unwrap();
        let parsed = Policy::from_toml(&toml).unwrap();

        assert_eq!(original.memory.max_mb, parsed.memory.max_mb);
        assert_eq!(original.fuel.per_step, parsed.fuel.per_step);
        assert_eq!(original.timeout.step_ms, parsed.timeout.step_ms);
    }

    #[test]
    fn test_capability_config() {
        let mut caps = CapabilityConfig::new();
        assert!(!caps.threading);
        assert!(caps.simd);

        caps.set("custom_feature", true);
        assert!(caps.allows("custom_feature"));
        assert!(!caps.allows("unknown"));
    }
}
