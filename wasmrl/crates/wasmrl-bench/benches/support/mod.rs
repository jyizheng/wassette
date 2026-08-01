// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Shared fixtures for runtime benchmarks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use wasmrl_runtime::{
    ComponentRef, EnvRuntime, InstanceHandle, PolicyConfig, RuntimeConfig, WasmEnvFactory,
};
use wasmrl_wit::EnvConfig;

pub fn component_path(file_name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/wasm32-wasip2/release")
        .join(file_name);
    path.is_file().then_some(path)
}

pub fn counter_config() -> EnvConfig {
    EnvConfig::new(r#"{"initial_value":0,"target":1000000000,"max_steps":4294967295}"#)
}

pub fn create_factory(path: &Path, max_instances: usize) -> Arc<WasmEnvFactory> {
    Arc::new(
        WasmEnvFactory::with_config(
            ComponentRef::from_file(path.to_string_lossy()),
            PolicyConfig::default(),
            RuntimeConfig::new().with_max_instances(max_instances.max(1)),
        )
        .expect("failed to create benchmark factory"),
    )
}

pub struct RuntimeSet {
    pub runtime: EnvRuntime,
    pub handles: Vec<InstanceHandle>,
}

impl RuntimeSet {
    pub fn new(factory: Arc<WasmEnvFactory>, count: usize, config: &EnvConfig) -> Self {
        let handles = factory.spawn(count).expect("failed to spawn environments");
        let mut runtime = EnvRuntime::new(factory);
        for (index, handle) in handles.iter().enumerate() {
            runtime
                .init(*handle, config.clone())
                .expect("failed to initialize environment");
            runtime
                .reset(*handle, index as u64)
                .expect("failed to reset environment");
        }
        Self { runtime, handles }
    }
}

impl Drop for RuntimeSet {
    fn drop(&mut self) {
        for handle in self.handles.drain(..) {
            let _ = self.runtime.close(handle);
        }
    }
}
