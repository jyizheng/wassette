// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Environment factory for creating and managing WasmRL environment instances.
//!
//! The factory provides:
//! - Component loading and validation
//! - Instance spawning with pre-warming support
//! - Configuration management

use std::sync::Arc;

use anyhow::Result;
use wasmtime::component::Component;

use crate::config::{PolicyConfig, RuntimeConfig};
use crate::engine::EngineContext;
use crate::error::RuntimeResult;
use crate::instance::InstanceHandle;
use crate::pool::SharedPool;

/// Reference to a component (either bytes or file path).
#[derive(Debug, Clone)]
pub enum ComponentRef {
    /// Component as raw bytes.
    Bytes(Arc<[u8]>),
    /// Component as file path.
    File(String),
    /// Component as OCI reference (future).
    Oci(String),
}

impl ComponentRef {
    /// Create a component reference from bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(bytes.into().into())
    }

    /// Create a component reference from a file path.
    pub fn from_file(path: impl Into<String>) -> Self {
        Self::File(path.into())
    }

    /// Create a component reference from an OCI reference.
    pub fn from_oci(reference: impl Into<String>) -> Self {
        Self::Oci(reference.into())
    }
}

/// Factory for creating environment instances.
pub struct WasmEnvFactory {
    /// Engine context for Wasmtime operations.
    engine: EngineContext,
    /// Loaded component.
    component: Component,
    /// Runtime configuration.
    config: RuntimeConfig,
    /// Policy configuration.
    policy: PolicyConfig,
    /// Instance pool.
    pool: SharedPool,
    /// Component reference for identification.
    component_ref: ComponentRef,
}

impl std::fmt::Debug for WasmEnvFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmEnvFactory")
            .field("engine", &self.engine)
            .field("component", &"<wasmtime::component::Component>")
            .field("config", &self.config)
            .field("policy", &self.policy)
            .field("pool", &self.pool)
            .field("component_ref", &self.component_ref)
            .finish()
    }
}

impl WasmEnvFactory {
    /// Create a new environment factory.
    ///
    /// # Arguments
    /// * `component_ref` - Reference to the component to load
    /// * `policy` - Policy configuration for the environments
    ///
    /// # Example
    /// ```ignore
    /// use wasmrl_runtime::{WasmEnvFactory, ComponentRef, PolicyConfig};
    ///
    /// let factory = WasmEnvFactory::new(
    ///     ComponentRef::from_file("counter_env.wasm"),
    ///     PolicyConfig::default(),
    /// )?;
    /// ```
    pub fn new(component_ref: ComponentRef, policy: PolicyConfig) -> Result<Self> {
        Self::with_config(component_ref, policy, RuntimeConfig::default())
    }

    /// Create a factory with custom runtime configuration.
    pub fn with_config(
        component_ref: ComponentRef,
        policy: PolicyConfig,
        mut config: RuntimeConfig,
    ) -> Result<Self> {
        // Apply policy to config
        policy.apply_to(&mut config);

        // Create engine context
        let engine = EngineContext::new(&config)?;

        // Load component
        let component = match &component_ref {
            ComponentRef::Bytes(bytes) => engine.load_component(bytes)?,
            ComponentRef::File(path) => engine.load_component_file(path)?,
            ComponentRef::Oci(_reference) => {
                // TODO: Implement OCI loading
                anyhow::bail!("OCI component loading not yet implemented")
            }
        };

        // Create instance pool
        let pool = SharedPool::new(config.max_instances);

        Ok(Self {
            engine,
            component,
            config,
            policy,
            pool,
            component_ref,
        })
    }

    /// Spawn new environment instances.
    ///
    /// # Arguments
    /// * `n` - Number of instances to spawn
    ///
    /// # Returns
    /// Vector of instance handles, pre-warmed and ready to use.
    pub fn spawn(&self, n: usize) -> RuntimeResult<Vec<InstanceHandle>> {
        self.pool.allocate_many(n)
    }

    /// Spawn a single environment instance.
    pub fn spawn_one(&self) -> RuntimeResult<InstanceHandle> {
        self.pool.allocate()
    }

    /// Release an instance back to the pool.
    pub fn release(&self, handle: InstanceHandle) -> RuntimeResult<()> {
        self.pool.release(handle)
    }

    /// Get the engine context.
    pub fn engine(&self) -> &EngineContext {
        &self.engine
    }

    /// Get the loaded component.
    pub fn component(&self) -> &Component {
        &self.component
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get the policy configuration.
    pub fn policy(&self) -> &PolicyConfig {
        &self.policy
    }

    /// Get the instance pool.
    pub fn pool(&self) -> &SharedPool {
        &self.pool
    }

    /// Get the component reference.
    pub fn component_ref(&self) -> &ComponentRef {
        &self.component_ref
    }

    /// Get available instance count.
    pub fn available(&self) -> usize {
        self.pool.available()
    }

    /// Check if pool is at capacity.
    pub fn is_full(&self) -> bool {
        self.pool.is_full()
    }
}

/// Builder for creating WasmEnvFactory with fluent API.
#[derive(Debug, Default)]
pub struct FactoryBuilder {
    policy: PolicyConfig,
    config: RuntimeConfig,
}

impl FactoryBuilder {
    /// Create a new factory builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the policy configuration.
    #[must_use]
    pub fn policy(mut self, policy: PolicyConfig) -> Self {
        self.policy = policy;
        self
    }

    /// Set maximum instances.
    #[must_use]
    pub fn max_instances(mut self, max: usize) -> Self {
        self.config = self.config.with_max_instances(max);
        self
    }

    /// Set maximum memory per instance.
    #[must_use]
    pub fn max_memory_mb(mut self, mb: u64) -> Self {
        self.config = self.config.with_max_memory_mb(mb);
        self
    }

    /// Set fuel per step.
    #[must_use]
    pub fn fuel_per_step(mut self, fuel: u64) -> Self {
        self.config = self.config.with_fuel_per_step(fuel);
        self
    }

    /// Enable pre-warming.
    #[must_use]
    pub fn prewarm(mut self, count: usize) -> Self {
        self.config = self.config.with_prewarming(count);
        self
    }

    /// Get the current runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get the current policy configuration.
    pub fn policy_config(&self) -> &PolicyConfig {
        &self.policy
    }

    /// Build the factory from a component file.
    pub fn build_from_file(self, path: &str) -> Result<WasmEnvFactory> {
        WasmEnvFactory::with_config(ComponentRef::from_file(path), self.policy, self.config)
    }

    /// Build the factory from component bytes.
    pub fn build_from_bytes(self, bytes: impl Into<Vec<u8>>) -> Result<WasmEnvFactory> {
        WasmEnvFactory::with_config(ComponentRef::from_bytes(bytes), self.policy, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_ref_from_bytes() {
        let bytes = vec![0u8; 100];
        let ref_ = ComponentRef::from_bytes(bytes.clone());
        if let ComponentRef::Bytes(b) = ref_ {
            assert_eq!(b.len(), 100);
        } else {
            panic!("Expected Bytes variant");
        }
    }

    #[test]
    fn test_component_ref_from_file() {
        let ref_ = ComponentRef::from_file("/path/to/component.wasm");
        if let ComponentRef::File(p) = ref_ {
            assert_eq!(p, "/path/to/component.wasm");
        } else {
            panic!("Expected File variant");
        }
    }

    #[test]
    fn test_component_ref_from_oci() {
        let ref_ = ComponentRef::from_oci("ghcr.io/example/env:latest");
        if let ComponentRef::Oci(r) = ref_ {
            assert_eq!(r, "ghcr.io/example/env:latest");
        } else {
            panic!("Expected Oci variant");
        }
    }

    #[test]
    fn test_factory_builder_defaults() {
        let builder = FactoryBuilder::new();
        assert_eq!(builder.config().max_instances, 256);
    }

    #[test]
    fn test_factory_builder_chain() {
        let builder = FactoryBuilder::new()
            .max_instances(128)
            .max_memory_mb(256)
            .fuel_per_step(1_000_000)
            .prewarm(16);

        assert_eq!(builder.config().max_instances, 128);
        assert_eq!(builder.config().max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(builder.config().fuel_per_step, 1_000_000);
        assert!(builder.config().prewarm_instances);
        assert_eq!(builder.config().prewarm_count, 16);
    }

    // Note: Full factory tests require actual Wasm components
    // These are tested in integration tests
}
