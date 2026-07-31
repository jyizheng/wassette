// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL Policy - Schema, parser, and enforcement for resource budgets and capabilities.
//!
//! This crate provides a comprehensive policy system for controlling WebAssembly
//! environment execution, including:
//!
//! - **Resource Budgets**: Memory limits, fuel/instruction budgets, timeouts
//! - **Capability Control**: WASI filesystem permissions, network access
//! - **Telemetry**: Tracking of budget overruns, denied capabilities, trap reasons
//!
//! # Quick Start
//!
//! ```rust
//! use wasmrl_policy::{Policy, PolicyBuilder, MemoryLimit, FuelBudget};
//!
//! // Create a policy with specific limits
//! let policy = PolicyBuilder::new()
//!     .max_memory_mb(64)
//!     .fuel_per_step(1_000_000)
//!     .timeout_step_ms(100)
//!     .timeout_reset_ms(500)
//!     .allow_filesystem_read(&["/data"])
//!     .deny_network()
//!     .build();
//!
//! // Or load from TOML
//! let policy = Policy::from_toml(r#"
//!     [memory]
//!     max_mb = 64
//!
//!     [fuel]
//!     per_step = 1000000
//!
//!     [timeout]
//!     step_ms = 100
//!     reset_ms = 500
//!
//!     [wasi]
//!     filesystem_read = ["/data"]
//!     network = false
//! "#)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Policy Schema
//!
//! Policies can be defined in TOML or JSON format:
//!
//! ```toml
//! # Memory limits
//! [memory]
//! max_mb = 64              # Maximum memory in megabytes
//! initial_mb = 16          # Initial memory allocation
//!
//! # Fuel/instruction budgets
//! [fuel]
//! per_step = 1000000       # Fuel per step call
//! per_reset = 5000000      # Fuel per reset call
//! per_batch = 10000000     # Fuel per batch call
//!
//! # Timeouts
//! [timeout]
//! step_ms = 100            # Timeout for step (milliseconds)
//! reset_ms = 500           # Timeout for reset
//! batch_ms = 1000          # Timeout for batch operations
//!
//! # WASI capabilities
//! [wasi]
//! filesystem_read = ["/data", "/models"]
//! filesystem_write = ["/tmp"]
//! network = false          # Deny network by default
//! env_vars = ["HOME", "PATH"]
//! ```

#![warn(missing_docs)]

mod enforcement;
mod error;
mod schema;
mod telemetry;

pub use enforcement::{BudgetEnforcer, EnforcementAction, EnforcementResult};
pub use error::{PolicyError, PolicyResult};
pub use schema::{
    CapabilityConfig, FuelBudget, MemoryLimit, Policy, PolicyBuilder, TimeoutConfig, WasiConfig,
};
pub use telemetry::{
    BudgetOverrun, BudgetType, CapabilityDenial, PolicyEvent, PolicyTelemetry, TelemetryCollector,
    TrapInfo,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_builder_basic() {
        let policy = PolicyBuilder::new()
            .max_memory_mb(64)
            .fuel_per_step(1_000_000)
            .build();

        assert_eq!(policy.memory.max_mb, 64);
        assert_eq!(policy.fuel.per_step, 1_000_000);
    }

    #[test]
    fn test_policy_default() {
        let policy = Policy::default();
        assert!(policy.memory.max_mb > 0);
        assert!(policy.fuel.per_step > 0);
    }

    #[test]
    fn test_policy_from_toml() {
        let toml = r#"
            [memory]
            max_mb = 128

            [fuel]
            per_step = 2000000

            [timeout]
            step_ms = 50
        "#;

        let policy = Policy::from_toml(toml).unwrap();
        assert_eq!(policy.memory.max_mb, 128);
        assert_eq!(policy.fuel.per_step, 2_000_000);
        assert_eq!(policy.timeout.step_ms, 50);
    }

    #[test]
    fn test_policy_from_json() {
        let json = r#"{
            "memory": { "max_mb": 256 },
            "fuel": { "per_step": 500000 }
        }"#;

        let policy = Policy::from_json(json).unwrap();
        assert_eq!(policy.memory.max_mb, 256);
        assert_eq!(policy.fuel.per_step, 500_000);
    }

    #[test]
    fn test_policy_to_toml() {
        let policy = PolicyBuilder::new()
            .max_memory_mb(64)
            .fuel_per_step(1_000_000)
            .build();

        let toml = policy.to_toml().unwrap();
        assert!(toml.contains("max_mb = 64"));
    }
}
