// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL Runtime - in-process execution layer for WebAssembly environments.
//!
//! This crate provides the core runtime infrastructure for executing WebAssembly
//! environment components with batched stepping, instance pooling, and resource budgets.
//!
//! # Overview
//!
//! The runtime consists of several key components:
//!
//! - [`WasmEnvFactory`]: Loads components and spawns environment instances
//! - [`EnvRuntime`]: Executes environment operations (reset, step, snapshot)
//! - [`InstancePool`]: Manages instance lifecycle and recycling
//! - [`RuntimeMetrics`]: Collects latency statistics (p50, p99)
//!
//! # Quick Start
//!
//! ```ignore
//! use wasmrl_runtime::{WasmEnvFactory, EnvRuntime, ComponentRef, PolicyConfig};
//! use std::sync::Arc;
//!
//! // Create factory from component file
//! let factory = WasmEnvFactory::new(
//!     ComponentRef::from_file("counter_env.wasm"),
//!     PolicyConfig::default(),
//! )?;
//!
//! // Create runtime
//! let mut runtime = EnvRuntime::new(Arc::new(factory));
//!
//! // Spawn instances
//! let handles = runtime.factory().spawn(4)?;
//!
//! // Initialize and reset
//! for handle in &handles {
//!     runtime.init(*handle, EnvConfig::empty())?;
//!     runtime.reset(*handle, 42)?;
//! }
//!
//! // Step batch
//! let actions = vec![Tensor::zeros(DType::Int32, vec![1]); 4];
//! let results = runtime.step_many(&handles, &actions)?;
//! ```
//!
//! # Batch Execution
//!
//! For high-throughput RL workloads, use batch operations:
//!
//! ```ignore
//! use wasmrl_runtime::BatchExecutor;
//!
//! // Create batch executor with 256 environments
//! let mut executor = BatchExecutor::new(runtime, 256)?;
//!
//! // Initialize all environments
//! executor.init_all(EnvConfig::empty())?;
//!
//! // Reset with different seeds
//! let seeds: Vec<u64> = (0..256).collect();
//! let observations = executor.reset_all(&seeds)?;
//!
//! // Step all environments
//! let results = executor.step_all(&actions)?;
//! ```
//!
//! # Metrics
//!
//! The runtime tracks latency statistics:
//!
//! ```ignore
//! let metrics = runtime.metrics();
//! println!("Step p50: {:?}", metrics.step_latency.p50());
//! println!("Step p99: {:?}", metrics.step_latency.p99());
//! println!("Throughput: {:.2} steps/sec", metrics.throughput_steps_per_sec());
//! ```

#![warn(missing_docs)]

mod config;
mod engine;
mod error;
mod executor;
mod factory;
mod fast_reset;
mod instance;
mod metrics;
mod pool;
mod replay;
mod snapshot_cache;

// Re-export main types
pub use config::{PolicyConfig, RuntimeConfig};
pub use engine::{EngineContext, EnvState};
pub use error::{RuntimeError, RuntimeResult};
pub use executor::{BatchExecutor, EnvRuntime};
pub use factory::{ComponentRef, FactoryBuilder, WasmEnvFactory};
pub use fast_reset::{FastResetConfig, FastResetManager, FastResetMetrics, ResetResult};
pub use instance::{InstanceHandle, InstanceInfo, InstanceStatus};
pub use metrics::{LatencyStats, MetricsSummary, RuntimeMetrics, Timer};
pub use pool::{InstancePool, PoolStats, SharedPool};
pub use replay::{Checkpoint, RecordedAction, ReplayConfig, ReplayData, ReplayManager, ReplayRecorder};
pub use snapshot_cache::{CachedSnapshot, SharedSnapshotCache, SnapshotCache, SnapshotKey};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_new() {
        let config = RuntimeConfig::new();
        assert_eq!(config.max_instances, 256);
        assert_eq!(config.max_memory_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_instances, 256);
    }

    #[test]
    fn test_instance_handle_display() {
        let handle = InstanceHandle::new();
        assert!(format!("{}", handle).starts_with("Instance("));
    }

    #[test]
    fn test_instance_status() {
        assert_eq!(InstanceStatus::Ready, InstanceStatus::Ready);
        assert_ne!(InstanceStatus::Ready, InstanceStatus::Running);
        assert_ne!(InstanceStatus::Running, InstanceStatus::ErrorFatal);
    }

    #[test]
    fn test_pool_basic() {
        let mut pool = InstancePool::new(10);
        let handle = pool.allocate().unwrap();
        assert!(pool.get_info(handle).is_some());
        pool.release(handle).unwrap();
    }

    #[test]
    fn test_metrics_basic() {
        let mut metrics = RuntimeMetrics::new();
        metrics.record_step(std::time::Duration::from_micros(100));
        assert_eq!(metrics.steps_completed, 1);
    }

    #[test]
    fn test_latency_percentiles() {
        let mut stats = LatencyStats::new(100);
        for i in 1..=100 {
            stats.record(std::time::Duration::from_micros(i));
        }
        assert!(stats.p50().as_micros() > 0);
        assert!(stats.p99().as_micros() > 0);
    }

    #[test]
    fn test_component_ref_variants() {
        let _ = ComponentRef::from_bytes(vec![0u8; 10]);
        let _ = ComponentRef::from_file("/path/to/file.wasm");
        let _ = ComponentRef::from_oci("ghcr.io/example/env:latest");
    }

    #[test]
    fn test_factory_builder() {
        let builder = FactoryBuilder::new()
            .max_instances(128)
            .max_memory_mb(256)
            .fuel_per_step(1_000_000);

        assert_eq!(builder.config.max_instances, 128);
    }

    #[test]
    fn test_runtime_error_display() {
        let err = RuntimeError::instance_not_found(42);
        assert_eq!(err.to_string(), "Instance not found: 42");
    }

    #[test]
    fn test_pool_stats() {
        let pool = InstancePool::new(10);
        let stats = pool.stats();
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.active, 0);
    }
}

