// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Wasmtime engine and component context for WasmRL runtime.

use std::sync::Arc;

use anyhow::Result;
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use crate::config::RuntimeConfig;

/// State held in Wasmtime Store for WasmRL environments.
pub struct EnvState {
    /// Fuel consumed during execution.
    pub fuel_consumed: u64,
    /// Whether instance has trapped.
    pub trapped: bool,
    /// Last error message if any.
    pub last_error: Option<String>,
    /// WASI context exposed to component-model WASI imports.
    wasi: WasiCtx,
    /// Component resource table for WASI-owned resources.
    table: ResourceTable,
}

impl EnvState {
    /// Create a new empty state.
    pub fn new() -> Self {
        Self {
            fuel_consumed: 0,
            trapped: false,
            last_error: None,
            wasi: WasiCtxBuilder::new().build(),
            table: ResourceTable::new(),
        }
    }

    /// Record that a trap occurred.
    pub fn record_trap(&mut self, message: &str) {
        self.trapped = true;
        self.last_error = Some(message.to_string());
    }

    /// Clear error state.
    pub fn clear_error(&mut self) {
        self.trapped = false;
        self.last_error = None;
    }
}

impl Default for EnvState {
    fn default() -> Self {
        Self::new()
    }
}

impl WasiView for EnvState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl std::fmt::Debug for EnvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvState")
            .field("fuel_consumed", &self.fuel_consumed)
            .field("trapped", &self.trapped)
            .field("last_error", &self.last_error)
            .field("wasi", &"<WasiCtx>")
            .field("table", &"<ResourceTable>")
            .finish()
    }
}

/// Shared Wasmtime engine context.
#[derive(Clone)]
pub struct EngineContext {
    /// The Wasmtime engine.
    engine: Arc<Engine>,
    /// Component linker.
    linker: Arc<Linker<EnvState>>,
}

impl EngineContext {
    /// Create a new engine context with the given runtime configuration.
    pub fn new(runtime_config: &RuntimeConfig) -> Result<Self> {
        let mut config = Config::new();

        // Enable component model
        config.wasm_component_model(true);

        // Configure fuel metering if enabled
        if runtime_config.fuel_enabled() {
            config.consume_fuel(true);
        }

        // Configure epoch interruption if enabled
        if runtime_config.enable_epoch_interruption {
            config.epoch_interruption(true);
        }

        // Memory configuration
        config.max_wasm_stack(512 * 1024); // 512 KB stack
        config.memory_reservation_for_growth(1024 * 1024 * 1024); // 1 GB reserve

        let engine = Arc::new(Engine::new(&config)?);

        // Create linker with WASI support
        let mut linker = Linker::new(engine.as_ref());
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;

        Ok(Self {
            engine,
            linker: Arc::new(linker),
        })
    }

    /// Get a reference to the engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get a reference to the linker.
    pub fn linker(&self) -> &Linker<EnvState> {
        &self.linker
    }

    /// Load a component from bytes.
    pub fn load_component(&self, bytes: &[u8]) -> Result<Component> {
        Component::from_binary(self.engine.as_ref(), bytes)
    }

    /// Load a component from a file.
    pub fn load_component_file(&self, path: &str) -> Result<Component> {
        Component::from_file(self.engine.as_ref(), path)
    }

    /// Create a new store with the given state.
    pub fn create_store(&self, state: EnvState) -> Store<EnvState> {
        Store::new(self.engine.as_ref(), state)
    }

    /// Create a new store with fuel limit.
    pub fn create_store_with_fuel(&self, state: EnvState, fuel: u64) -> Result<Store<EnvState>> {
        let mut store = Store::new(self.engine.as_ref(), state);
        store.set_fuel(fuel)?;
        Ok(store)
    }
}

impl std::fmt::Debug for EngineContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineContext")
            .field("engine", &"<wasmtime::Engine>")
            .field("linker", &"<wasmtime::Linker>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_state_new() {
        let state = EnvState::new();
        assert_eq!(state.fuel_consumed, 0);
        assert!(!state.trapped);
        assert!(state.last_error.is_none());
    }

    #[test]
    fn test_env_state_record_trap() {
        let mut state = EnvState::new();
        state.record_trap("out of bounds memory access");

        assert!(state.trapped);
        assert_eq!(
            state.last_error.as_deref(),
            Some("out of bounds memory access")
        );
    }

    #[test]
    fn test_env_state_clear_error() {
        let mut state = EnvState::new();
        state.record_trap("error");
        state.clear_error();

        assert!(!state.trapped);
        assert!(state.last_error.is_none());
    }

    #[test]
    fn test_engine_context_create() {
        let config = RuntimeConfig::new();
        let ctx = EngineContext::new(&config);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_engine_context_with_fuel() {
        let config = RuntimeConfig::new().with_fuel_per_step(1_000_000);
        let ctx = EngineContext::new(&config);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_engine_context_with_epoch() {
        let config = RuntimeConfig::new().with_epoch_interruption(100);
        let ctx = EngineContext::new(&config);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_engine_context_create_store() {
        let config = RuntimeConfig::new();
        let ctx = EngineContext::new(&config).unwrap();
        let state = EnvState::new();
        let _store = ctx.create_store(state);
    }
}
