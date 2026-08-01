// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! M8 MCP Bridge Integration Tests
//!
//! Tests for the wasmrl-mcp-bridge crate, verifying:
//! - MCP tool exposure for environments
//! - Session management
//! - Overhead metrics collection
//! - Mode switching in benchmarks

use std::time::Duration;

use wasmrl_mcp_bridge::{
    BridgeError, EnvMcpBridge, McpBridgeConfig, McpTool, OverheadMetrics, SessionConfig, SessionId,
    SessionManager, SessionState, TimingBreakdown, ToolResult,
};

#[cfg(feature = "integration")]
fn counter_bridge_config() -> McpBridgeConfig {
    let component_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-wasip2/release/counter_env.wasm");
    assert!(component_path.is_file());
    McpBridgeConfig::new(component_path.to_string_lossy()).with_env_name("env")
}

// =============================================================================
// Configuration Tests
// =============================================================================

#[test]
fn test_mcp_bridge_config_creation() {
    let config = McpBridgeConfig::new("/path/to/counter_env.wasm")
        .with_max_sessions(32)
        .with_timeout_ms(60_000)
        .with_metrics(true)
        .with_env_name("counter");

    assert_eq!(config.component_path, "/path/to/counter_env.wasm");
    assert_eq!(config.max_sessions, 32);
    assert_eq!(config.timeout_ms, 60_000);
    assert!(config.collect_metrics);
    assert_eq!(config.get_env_name(), "counter");
}

#[test]
fn test_mcp_bridge_config_env_name_inference() {
    let config = McpBridgeConfig::new("/workspaces/wasmrl/envs/counter_env.wasm");
    assert_eq!(config.get_env_name(), "counter_env");

    let config2 = McpBridgeConfig::new("simple.wasm");
    assert_eq!(config2.get_env_name(), "simple");
}

#[test]
fn test_session_config_builder() {
    let config = SessionConfig::new()
        .with_seed(42)
        .with_auto_reset(true)
        .with_max_steps(1000)
        .with_recording(true)
        .with_env_config(serde_json::json!({"grid_size": 10}));

    assert_eq!(config.seed, Some(42));
    assert!(config.auto_reset);
    assert_eq!(config.max_steps, Some(1000));
    assert!(config.record_trajectory);
    assert!(config.env_config.is_some());
}

// =============================================================================
// Session Management Tests
// =============================================================================

#[test]
fn test_session_id_uniqueness() {
    let ids: Vec<SessionId> = (0..100).map(|_| SessionId::new()).collect();

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(ids[i], ids[j], "Session IDs should be unique");
        }
    }
}

#[test]
fn test_session_manager_lifecycle() {
    let mut manager = SessionManager::new(10);

    // Create sessions
    let id1 = manager.create_session(SessionConfig::new()).unwrap();
    let id2 = manager.create_session(SessionConfig::new()).unwrap();

    assert_eq!(manager.active_count(), 2);

    // Access session
    let session = manager.get(&id1).unwrap();
    assert_eq!(session.state, SessionState::Created);

    // Close session
    manager.close_session(&id1).unwrap();
    let session = manager.get(&id1).unwrap();
    assert_eq!(session.state, SessionState::Closed);

    // Still have id2 active
    assert!(manager.get(&id2).is_ok());
}

#[test]
fn test_session_manager_max_sessions() {
    let mut manager = SessionManager::new(3);

    manager.create_session(SessionConfig::new()).unwrap();
    manager.create_session(SessionConfig::new()).unwrap();
    manager.create_session(SessionConfig::new()).unwrap();

    // Fourth should fail
    let result = manager.create_session(SessionConfig::new());
    assert!(matches!(
        result,
        Err(BridgeError::MaxSessionsExceeded { .. })
    ));
}

#[test]
fn test_session_state_transitions() {
    assert!(SessionState::Created.can_reset());
    assert!(!SessionState::Created.can_step());

    assert!(SessionState::Ready.can_reset());
    assert!(SessionState::Ready.can_step());

    assert!(SessionState::Terminated.can_reset());
    assert!(!SessionState::Terminated.can_step());

    assert!(SessionState::Closed.is_active() == false);
}

// =============================================================================
// MCP Bridge Tests
// =============================================================================

#[test]
fn test_bridge_tool_definitions() {
    let config = McpBridgeConfig::new("counter.wasm");
    let bridge = EnvMcpBridge::new(config).unwrap();

    let tools = bridge.get_tools();
    assert!(!tools.is_empty());

    // Check expected tools exist
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"counter_create"));
    assert!(tool_names.contains(&"counter_reset"));
    assert!(tool_names.contains(&"counter_step"));
    assert!(tool_names.contains(&"counter_close"));
    assert!(tool_names.contains(&"counter_info"));
    assert!(tool_names.contains(&"counter_list"));
    assert!(tool_names.contains(&"counter_metrics"));
}

#[test]
#[cfg(feature = "integration")]
fn test_bridge_create_and_reset() {
    let config = counter_bridge_config();
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    // Create session
    let result = bridge.call_tool("env_create", serde_json::json!({"seed": 123}));
    assert!(result.is_ok);

    let session_id = result.data.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Reset session
    let result = bridge.call_tool(
        "env_reset",
        serde_json::json!({"session_id": session_id, "seed": 456}),
    );
    assert!(result.is_ok);

    let data = result.data.unwrap();
    assert!(data.get("observation").is_some());
}

#[test]
#[cfg(feature = "integration")]
fn test_bridge_step_workflow() {
    let config = counter_bridge_config();
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    // Create and reset
    let create = bridge.call_tool("env_create", serde_json::json!({}));
    let session_id = create.data.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    bridge.call_tool("env_reset", serde_json::json!({"session_id": &session_id}));

    // Step multiple times
    for _ in 0..10 {
        let result = bridge.call_tool(
            "env_step",
            serde_json::json!({"session_id": &session_id, "action": 0}),
        );
        assert!(result.is_ok);

        let data = result.data.unwrap();
        assert!(data.get("observation").is_some());
        assert!(data.get("reward").is_some());
        assert!(data.get("terminated").is_some());
    }
}

#[test]
fn test_bridge_session_not_found() {
    let config = McpBridgeConfig::new("env.wasm");
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    let result = bridge.call_tool(
        "env_reset",
        serde_json::json!({"session_id": "nonexistent-session"}),
    );

    assert!(!result.is_ok);
    assert!(result.error.unwrap().contains("not found"));
}

#[test]
fn test_bridge_unknown_tool() {
    let config = McpBridgeConfig::new("env.wasm");
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    let result = bridge.call_tool("unknown_tool", serde_json::json!({}));

    assert!(!result.is_ok);
    assert!(result.error.unwrap().contains("Unknown tool"));
}

#[test]
#[cfg(feature = "integration")]
fn test_bridge_list_sessions() {
    let config = counter_bridge_config();
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    // Create several sessions
    for _ in 0..5 {
        bridge.call_tool("env_create", serde_json::json!({}));
    }

    let result = bridge.call_tool("env_list", serde_json::json!({}));
    assert!(result.is_ok);

    let data = result.data.unwrap();
    assert_eq!(data["count"], 5);
}

// =============================================================================
// Overhead Metrics Tests
// =============================================================================

#[test]
fn test_timing_breakdown() {
    let breakdown = TimingBreakdown::new(100, 50, 850);

    assert_eq!(breakdown.rpc_serialization_us, 100);
    assert_eq!(breakdown.runtime_overhead_us, 50);
    assert_eq!(breakdown.env_compute_us, 850);
    assert_eq!(breakdown.total_us, 1000);

    // Overhead is (100 + 50) / 1000 = 15%
    assert!((breakdown.overhead_ratio() - 0.15).abs() < 0.001);
    assert!((breakdown.efficiency() - 0.85).abs() < 0.001);
}

#[test]
fn test_overhead_metrics_recording() {
    let mut metrics = OverheadMetrics::new();

    // Record several calls
    for _ in 0..10 {
        metrics.record_call(
            Duration::from_micros(100),
            Duration::from_micros(50),
            Duration::from_micros(850),
        );
    }

    assert_eq!(metrics.total_calls, 10);
    assert_eq!(metrics.total_env_time().as_micros(), 8500);

    // Average overhead should be 15%
    assert!((metrics.avg_overhead_ratio() - 0.15).abs() < 0.01);
}

#[test]
fn test_overhead_summary_report() {
    let mut metrics = OverheadMetrics::new();
    metrics.record_call(
        Duration::from_micros(100),
        Duration::from_micros(50),
        Duration::from_micros(850),
    );

    let summary = metrics.summary();
    let report = summary.report();

    assert!(report.contains("Overhead Report"));
    assert!(report.contains("Total Calls: 1"));
    assert!(report.contains("RPC/Serialization"));
}

#[test]
#[cfg(feature = "integration")]
fn test_bridge_metrics_collection() {
    let config = counter_bridge_config().with_metrics(true);
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    // Perform operations
    let create = bridge.call_tool("env_create", serde_json::json!({}));
    let session_id = create.data.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    bridge.call_tool("env_reset", serde_json::json!({"session_id": &session_id}));
    bridge.call_tool(
        "env_step",
        serde_json::json!({"session_id": &session_id, "action": 0}),
    );

    // Get metrics
    let result = bridge.call_tool("env_metrics", serde_json::json!({}));
    assert!(result.is_ok);

    let metrics = result.data.unwrap();
    assert!(metrics["total_calls"].as_u64().unwrap() >= 3);
}

#[test]
fn test_bridge_timing_in_results() {
    let config = McpBridgeConfig::new("env.wasm").with_metrics(true);
    let mut bridge = EnvMcpBridge::new(config).unwrap();

    let result = bridge.call_tool("env_list", serde_json::json!({}));
    assert!(result.is_ok);

    // With metrics enabled, timing should be included
    assert!(result.timing.is_some());
    let timing = result.timing.unwrap();
    assert!(timing.total_us > 0);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_bridge_error_types() {
    assert!(BridgeError::session_not_found("x")
        .to_string()
        .contains("not found"));
    assert!(BridgeError::max_sessions_exceeded(10)
        .to_string()
        .contains("10"));
    assert!(BridgeError::timeout(1000).to_string().contains("1000"));
    assert!(BridgeError::unknown_tool("x")
        .to_string()
        .contains("Unknown"));
}

#[test]
fn test_bridge_error_recoverability() {
    assert!(BridgeError::invalid_action("bad").is_recoverable());
    assert!(BridgeError::timeout(100).is_recoverable());

    assert!(BridgeError::component_load("fail").is_fatal());
    assert!(BridgeError::policy_violation("denied").is_fatal());
}

#[test]
fn test_bridge_error_codes() {
    assert_eq!(BridgeError::session_not_found("x").error_code(), -32001);
    assert_eq!(BridgeError::max_sessions_exceeded(10).error_code(), -32002);
    assert_eq!(BridgeError::unknown_tool("x").error_code(), -32601);
}

// =============================================================================
// Tool Result Tests
// =============================================================================

#[test]
fn test_tool_result_success() {
    let result = ToolResult::success(serde_json::json!({"key": "value"}));
    assert!(result.is_ok);
    assert!(result.data.is_some());
    assert!(result.error.is_none());
}

#[test]
fn test_tool_result_error() {
    let result = ToolResult::error("Something went wrong");
    assert!(!result.is_ok);
    assert!(result.data.is_none());
    assert_eq!(result.error, Some("Something went wrong".to_string()));
}

#[test]
fn test_tool_result_with_timing() {
    let timing = TimingBreakdown::new(10, 5, 85);
    let result = ToolResult::success_with_timing(serde_json::json!({}), timing);

    assert!(result.is_ok);
    assert!(result.timing.is_some());
}

// =============================================================================
// Benchmark Mode Tests
// =============================================================================

#[test]
fn test_bench_mode_switch() {
    use wasmrl_bench::{BenchConfig, BenchMode};

    let inproc_config = BenchConfig::new()
        .with_mode(BenchMode::WasmInproc)
        .with_num_envs(256);

    let mcp_config = BenchConfig::new()
        .with_mode(BenchMode::McpTool)
        .with_num_envs(16); // Fewer envs for MCP due to overhead

    assert_eq!(inproc_config.mode, BenchMode::WasmInproc);
    assert_eq!(mcp_config.mode, BenchMode::McpTool);

    assert!(inproc_config.mode.is_inproc());
    assert!(mcp_config.mode.has_rpc_overhead());
}

#[test]
fn test_bench_mode_comparison() {
    use std::time::Duration;

    use wasmrl_bench::{BenchMode, ModeComparison, TimingResult};

    let inproc_result = TimingResult {
        mean: Duration::from_micros(100),
        std_dev: Duration::from_micros(10),
        min: Duration::from_micros(80),
        max: Duration::from_micros(120),
        p50: Duration::from_micros(100),
        p99: Duration::from_micros(115),
        samples: 1000,
    };

    let mcp_result = TimingResult {
        mean: Duration::from_micros(1000), // 10x slower
        std_dev: Duration::from_micros(100),
        min: Duration::from_micros(800),
        max: Duration::from_micros(1200),
        p50: Duration::from_micros(1000),
        p99: Duration::from_micros(1150),
        samples: 1000,
    };

    let comparison = ModeComparison::new(
        BenchMode::WasmInproc,
        BenchMode::McpTool,
        &inproc_result,
        &mcp_result,
    );

    // MCP is 10x slower, so speedup = 0.1
    assert!((comparison.speedup - 0.1).abs() < 0.01);

    // Overhead is 900%
    assert!((comparison.overhead_percent - 900.0).abs() < 1.0);

    let report = comparison.report();
    assert!(report.contains("wasm_inproc"));
    assert!(report.contains("mcp_tool"));
}
