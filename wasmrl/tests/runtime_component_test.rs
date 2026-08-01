// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! End-to-end tests using a real wasm32-wasip2 WasmRL component.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use wasmrl_mcp_bridge::{EnvMcpBridge, McpBridgeConfig};
use wasmrl_py::{PyEnvConfig, PyWasmEnv};
use wasmrl_runtime::{ComponentRef, EnvRuntime, PolicyConfig, RuntimeError, WasmEnvFactory};
use wasmrl_wit::{DType, EnvConfig, Tensor};

fn counter_component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/wasm32-wasip2/release/counter_env.wasm")
}

fn malicious_loop_component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-wasip2/release/malicious_loop_env.wasm")
}

fn malicious_memory_component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-wasip2/release/malicious_memory_env.wasm")
}

fn increment_action() -> Tensor {
    Tensor::new(DType::Int32, vec![1], 1_i32.to_le_bytes().to_vec())
}

fn scalar_f32(tensor: &Tensor) -> f32 {
    assert_eq!(tensor.dtype, DType::Float32);
    assert_eq!(tensor.shape, vec![1]);
    f32::from_le_bytes(tensor.data.as_slice().try_into().unwrap())
}

#[test]
fn counter_component_lifecycle_and_snapshot_round_trip() {
    let component_path = counter_component_path();
    assert!(
        component_path.is_file(),
        "build the component first with: cargo build -p counter-env --target wasm32-wasip2 --release"
    );

    let factory = WasmEnvFactory::new(
        ComponentRef::from_file(component_path.to_string_lossy()),
        PolicyConfig::default(),
    )
    .expect("counter component should load and expose the WasmRL world");
    let factory = Arc::new(factory);
    let handle = factory.spawn_one().unwrap();
    let mut runtime = EnvRuntime::new(factory);

    runtime
        .init(
            handle,
            EnvConfig::new(r#"{"initial_value":0,"target":3,"max_steps":10}"#),
        )
        .unwrap();

    let observation_space = runtime.observation_space(handle).unwrap();
    assert_eq!(observation_space.dtype, DType::Float32);
    assert_eq!(observation_space.shape, vec![1]);

    let action_space = runtime.action_space(handle).unwrap();
    assert_eq!(action_space.dtype, DType::Int32);
    assert_eq!(i32::from_le_bytes(action_space.data.try_into().unwrap()), 3);

    let observation = runtime.reset(handle, 42).unwrap();
    assert_eq!(scalar_f32(&observation), 0.0);

    let first = runtime.step(handle, &increment_action()).unwrap();
    assert_eq!(scalar_f32(&first.observation), 1.0);
    assert_eq!(first.reward, -0.01);
    assert!(!first.done());

    let snapshot = runtime.snapshot(handle).unwrap();

    let second = runtime.step(handle, &increment_action()).unwrap();
    assert_eq!(scalar_f32(&second.observation), 2.0);
    assert!(!second.done());

    runtime.restore(handle, &snapshot).unwrap();
    let replayed_second = runtime.step(handle, &increment_action()).unwrap();
    assert_eq!(scalar_f32(&replayed_second.observation), 2.0);
    assert_eq!(replayed_second.reward, second.reward);
    assert_eq!(replayed_second.terminated, second.terminated);
    assert_eq!(replayed_second.truncated, second.truncated);

    let terminal = runtime.step(handle, &increment_action()).unwrap();
    assert_eq!(scalar_f32(&terminal.observation), 3.0);
    assert_eq!(terminal.reward, 1.0);
    assert!(terminal.terminated);
    assert!(!terminal.truncated);

    runtime.close(handle).unwrap();
    assert_eq!(runtime.active_count(), 0);
}

#[test]
fn counter_component_runs_through_mcp_bridge() {
    let component_path = counter_component_path();
    let config = McpBridgeConfig::new(component_path.to_string_lossy()).with_env_name("counter");
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    let created = bridge.call_tool(
        "counter_create",
        serde_json::json!({
            "seed": 7,
            "config": {"initial_value": 0, "target": 2, "max_steps": 10}
        }),
    );
    assert!(created.is_ok, "{:?}", created.error);
    let session_id = created.data.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    let reset = bridge.call_tool(
        "counter_reset",
        serde_json::json!({"session_id": session_id}),
    );
    assert!(reset.is_ok, "{:?}", reset.error);
    assert_eq!(reset.data.unwrap()["observation"], serde_json::json!([0.0]));

    let first = bridge.call_tool(
        "counter_step",
        serde_json::json!({"session_id": session_id, "action": 1}),
    );
    assert!(first.is_ok, "{:?}", first.error);
    let first = first.data.unwrap();
    assert_eq!(first["observation"], serde_json::json!([1.0]));
    assert_eq!(first["reward"], serde_json::json!(-0.01));
    assert_eq!(first["terminated"], serde_json::json!(false));

    let terminal = bridge.call_tool(
        "counter_step",
        serde_json::json!({"session_id": session_id, "action": 1}),
    );
    assert!(terminal.is_ok, "{:?}", terminal.error);
    let terminal = terminal.data.unwrap();
    assert_eq!(terminal["observation"], serde_json::json!([2.0]));
    assert_eq!(terminal["reward"], serde_json::json!(1.0));
    assert_eq!(terminal["terminated"], serde_json::json!(true));

    let info = bridge.call_tool("counter_info", serde_json::json!({}));
    assert!(info.is_ok, "{:?}", info.error);
    let info = info.data.unwrap();
    assert_eq!(info["action_space"]["n"], serde_json::json!(3));
    assert_eq!(info["observation_space"]["shape"], serde_json::json!([1]));

    let closed = bridge.call_tool(
        "counter_close",
        serde_json::json!({"session_id": session_id}),
    );
    assert!(closed.is_ok, "{:?}", closed.error);
}

#[test]
fn counter_component_runs_through_python_wrapper() {
    Python::initialize();
    Python::attach(|py| {
        let component_bytes = std::fs::read(counter_component_path()).unwrap();
        let mut config = PyEnvConfig::default();
        config.config_json = r#"{"initial_value":0,"target":1,"max_steps":10}"#.to_string();
        let env = PyWasmEnv::new(py, component_bytes, Some(&config)).unwrap();
        let env = Py::new(py, env).unwrap();
        let env = env.bind(py);

        let action_count: i64 = env
            .getattr("action_space")
            .unwrap()
            .getattr("n")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(action_count, 3);

        let observation_shape: Vec<usize> = env
            .getattr("observation_space")
            .unwrap()
            .getattr("shape")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(observation_shape, vec![1]);
        env.call_method0("close").unwrap();
    });
}

fn malicious_loop_runtime(policy: PolicyConfig) -> (EnvRuntime, wasmrl_runtime::InstanceHandle) {
    let factory = WasmEnvFactory::new(
        ComponentRef::from_file(malicious_loop_component_path().to_string_lossy()),
        policy,
    )
    .unwrap();
    let factory = Arc::new(factory);
    let handle = factory.spawn_one().unwrap();
    let mut runtime = EnvRuntime::new(factory);
    runtime
        .init(handle, EnvConfig::new(r#"{"loop_on":"step"}"#))
        .unwrap();
    runtime.reset(handle, 0).unwrap();
    (runtime, handle)
}

#[test]
fn fuel_interrupts_a_non_terminating_component() {
    let policy = PolicyConfig {
        fuel_per_step: Some(25_000),
        fuel_per_reset: Some(1_000_000),
        ..Default::default()
    };
    let (mut runtime, handle) = malicious_loop_runtime(policy);
    let action = Tensor::new(DType::Int32, vec![1], 0_i32.to_le_bytes().to_vec());

    let error = runtime.step(handle, &action).unwrap_err();
    assert!(
        matches!(error, RuntimeError::FuelExhausted { .. }),
        "{error}"
    );
}

#[test]
fn timeout_interrupts_a_non_terminating_component() {
    let policy = PolicyConfig {
        timeout_ms_step: Some(20),
        timeout_ms_reset: Some(500),
        ..Default::default()
    };
    let (mut runtime, handle) = malicious_loop_runtime(policy);
    let action = Tensor::new(DType::Int32, vec![1], 0_i32.to_le_bytes().to_vec());
    let started = std::time::Instant::now();

    let error = runtime.step(handle, &action).unwrap_err();
    assert!(matches!(error, RuntimeError::Timeout { .. }), "{error}");
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn memory_limit_stops_guest_memory_growth() {
    let policy = PolicyConfig {
        max_memory_mb: Some(32),
        timeout_ms_step: Some(1_000),
        timeout_ms_reset: Some(1_000),
        ..Default::default()
    };
    let factory = WasmEnvFactory::new(
        ComponentRef::from_file(malicious_memory_component_path().to_string_lossy()),
        policy,
    )
    .unwrap();
    let factory = Arc::new(factory);
    let handle = factory.spawn_one().unwrap();
    let mut runtime = EnvRuntime::new(factory);
    runtime
        .init(
            handle,
            EnvConfig::new(r#"{"alloc_mb_per_step":64,"keep_allocations":true}"#),
        )
        .unwrap();
    runtime.reset(handle, 0).unwrap();
    let action = Tensor::new(DType::Int32, vec![1], 0_i32.to_le_bytes().to_vec());
    let started = std::time::Instant::now();

    let error = runtime.step(handle, &action).unwrap_err();
    assert!(
        matches!(
            error,
            RuntimeError::MemoryLimitExceeded { .. } | RuntimeError::InstanceTrapped { .. }
        ),
        "{error}"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
}
