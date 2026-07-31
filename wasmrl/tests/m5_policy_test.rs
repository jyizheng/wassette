// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! M5 Policy Integration Tests
//!
//! This test suite validates the policy schema, parsing, enforcement, and telemetry.

use std::time::Duration;

use wasmrl_policy::{
    BudgetEnforcer, BudgetOverrun, BudgetType, CapabilityConfig, CapabilityDenial,
    EnforcementAction, EnforcementResult, FuelBudget, MemoryLimit, Policy, PolicyBuilder,
    PolicyError, PolicyTelemetry, TelemetryCollector, TimeoutConfig, TrapInfo, WasiConfig,
};

// ============================================================================
// Policy Schema Integration Tests
// ============================================================================

#[test]
fn test_policy_complete_toml_parsing() {
    let toml = r#"
        [memory]
        max_mb = 128
        initial_mb = 16
        enabled = true

        [fuel]
        per_step = 500000
        per_reset = 2000000
        per_batch = 5000000
        enabled = true

        [timeout]
        step_ms = 50
        reset_ms = 200
        batch_ms = 500
        enabled = true

        [wasi]
        filesystem_read = ["/data", "/models"]
        filesystem_write = ["/tmp", "/output"]
        network = false
        clock = true
        random = true

        [capabilities]
        threading = false
        simd = true
        component_model = true
    "#;

    let policy = Policy::from_toml(toml).unwrap();

    // Memory checks
    assert_eq!(policy.memory.max_mb, 128);
    assert_eq!(policy.memory.initial_mb, 16);
    assert!(policy.memory.enabled);

    // Fuel checks
    assert_eq!(policy.fuel.per_step, 500_000);
    assert_eq!(policy.fuel.per_reset, 2_000_000);
    assert_eq!(policy.fuel.per_batch, 5_000_000);

    // Timeout checks
    assert_eq!(policy.timeout.step_ms, 50);
    assert_eq!(policy.timeout.reset_ms, 200);
    assert_eq!(policy.timeout.batch_ms, 500);

    // WASI checks
    assert_eq!(policy.wasi.filesystem_read.len(), 2);
    assert_eq!(policy.wasi.filesystem_write.len(), 2);
    assert!(!policy.wasi.network);
    assert!(policy.wasi.clock);

    // Capability checks
    assert!(!policy.capabilities.threading);
    assert!(policy.capabilities.simd);
}

#[test]
fn test_policy_json_parsing() {
    let json = r#"{
        "memory": {
            "max_mb": 256,
            "initial_mb": 32,
            "enabled": true
        },
        "fuel": {
            "per_step": 1000000,
            "per_reset": 5000000,
            "enabled": true
        },
        "timeout": {
            "step_ms": 100,
            "reset_ms": 500,
            "enabled": true
        },
        "wasi": {
            "filesystem_read": ["/data"],
            "network": false
        }
    }"#;

    let policy = Policy::from_json(json).unwrap();
    assert_eq!(policy.memory.max_mb, 256);
    assert_eq!(policy.fuel.per_step, 1_000_000);
    assert!(!policy.wasi.network);
}

#[test]
fn test_policy_roundtrip_toml() {
    let original = PolicyBuilder::new()
        .max_memory_mb(64)
        .fuel_per_step(750_000)
        .timeout_step_ms(75)
        .allow_filesystem_read(&["/data"])
        .deny_network()
        .build();

    let toml = original.to_toml().unwrap();
    let restored = Policy::from_toml(&toml).unwrap();

    assert_eq!(original.memory.max_mb, restored.memory.max_mb);
    assert_eq!(original.fuel.per_step, restored.fuel.per_step);
    assert_eq!(original.timeout.step_ms, restored.timeout.step_ms);
}

#[test]
fn test_policy_roundtrip_json() {
    let original = PolicyBuilder::new()
        .max_memory_mb(128)
        .fuel_per_step(500_000)
        .build();

    let json = original.to_json().unwrap();
    let restored = Policy::from_json(&json).unwrap();

    assert_eq!(original.memory.max_mb, restored.memory.max_mb);
    assert_eq!(original.fuel.per_step, restored.fuel.per_step);
}

#[test]
fn test_policy_validation_errors() {
    // Zero memory should fail
    let mut policy = Policy::default();
    policy.memory.max_mb = 0;
    assert!(policy.validate().is_err());

    // Initial > max should fail
    let mut policy = Policy::default();
    policy.memory.initial_mb = 512;
    policy.memory.max_mb = 256;
    assert!(policy.validate().is_err());

    // Zero fuel with fuel enabled should fail
    let mut policy = Policy::default();
    policy.fuel.per_step = 0;
    policy.fuel.enabled = true;
    assert!(policy.validate().is_err());

    // Valid policy should pass
    let policy = Policy::default();
    assert!(policy.validate().is_ok());
}

// ============================================================================
// Policy Builder Integration Tests
// ============================================================================

#[test]
fn test_policy_builder_complete() {
    let policy = PolicyBuilder::new()
        .max_memory_mb(128)
        .initial_memory_mb(32)
        .fuel_per_step(500_000)
        .fuel_per_reset(2_000_000)
        .fuel_per_batch(5_000_000)
        .timeout_step_ms(50)
        .timeout_reset_ms(200)
        .timeout_batch_ms(500)
        .allow_filesystem_read(&["/data", "/models"])
        .allow_filesystem_write(&["/tmp"])
        .deny_network()
        .env_vars(&["HOME", "PATH"])
        .build();

    assert_eq!(policy.memory.max_mb, 128);
    assert_eq!(policy.memory.initial_mb, 32);
    assert_eq!(policy.fuel.per_step, 500_000);
    assert_eq!(policy.timeout.step_ms, 50);
    assert_eq!(policy.wasi.filesystem_read.len(), 2);
    assert_eq!(policy.wasi.filesystem_write.len(), 1);
    assert!(!policy.wasi.network);
}

#[test]
fn test_policy_builder_validated() {
    // Valid policy should work
    let result = PolicyBuilder::new().max_memory_mb(64).build_validated();
    assert!(result.is_ok());

    // Invalid should fail
    let result = PolicyBuilder::new().max_memory_mb(0).build_validated();
    assert!(result.is_err());
}

// ============================================================================
// Budget Enforcement Integration Tests
// ============================================================================

#[test]
fn test_enforcer_memory_enforcement() {
    let policy = PolicyBuilder::new().max_memory_mb(64).build();
    let enforcer = BudgetEnforcer::new(policy);

    // Within limit
    let result = enforcer.check_memory(32 * 1024 * 1024);
    assert!(result.is_allowed());

    // At limit
    let result = enforcer.check_memory(64 * 1024 * 1024);
    assert!(result.is_allowed());

    // Over limit
    let result = enforcer.check_memory(128 * 1024 * 1024);
    assert!(result.is_denied());
    if let EnforcementResult::Denied(action) = result {
        let error = action.to_error();
        assert!(error.is_budget_error());
    }
}

#[test]
fn test_enforcer_fuel_tracking() {
    let policy = PolicyBuilder::new().fuel_per_step(1_000_000).build();
    let enforcer = BudgetEnforcer::new(policy);

    // Record fuel usage
    enforcer.record_fuel(500_000);
    assert_eq!(enforcer.fuel_consumed(), 500_000);

    // Within budget
    let result = enforcer.check_fuel("step");
    assert!(result.is_allowed());

    // Exceed budget
    enforcer.record_fuel(600_000);
    let result = enforcer.check_fuel("step");
    assert!(matches!(result, EnforcementResult::Exceeded(_)));
}

#[test]
fn test_enforcer_capability_checks() {
    let policy = PolicyBuilder::new()
        .allow_filesystem_read(&["/data", "/models"])
        .allow_filesystem_write(&["/tmp"])
        .deny_network()
        .build();
    let enforcer = BudgetEnforcer::new(policy);

    // Read checks
    assert!(enforcer.check_read("/data/file.txt").is_allowed());
    assert!(enforcer.check_read("/models/model.bin").is_allowed());
    assert!(enforcer.check_read("/etc/passwd").is_denied());

    // Write checks
    assert!(enforcer.check_write("/tmp/output.txt").is_allowed());
    assert!(enforcer.check_write("/data/file.txt").is_denied());

    // Network check
    assert!(enforcer.check_network().is_denied());
}

#[test]
fn test_enforcer_network_allowed() {
    let policy = PolicyBuilder::new().allow_network().build();
    let enforcer = BudgetEnforcer::new(policy);

    assert!(enforcer.check_network().is_allowed());
}

#[test]
fn test_enforcer_timed_operations() {
    let policy = PolicyBuilder::new().timeout_step_ms(100).build();
    let enforcer = BudgetEnforcer::new(policy);

    // Start a timed operation
    let guard = enforcer.timed_guard("step");
    assert!(!guard.is_timed_out());
    assert!(guard.remaining() > Duration::from_millis(50));

    // Finalize (should be allowed since not enough time passed)
    let result = guard.finalize();
    assert!(result.is_allowed());
}

#[test]
fn test_enforcer_step_tracking() {
    let policy = Policy::default();
    let enforcer = BudgetEnforcer::new(policy);

    enforcer.record_step();
    enforcer.record_step();
    enforcer.record_step();

    assert_eq!(enforcer.steps_executed(), 3);
}

#[test]
fn test_enforcer_reset_counters() {
    let policy = Policy::default();
    let enforcer = BudgetEnforcer::new(policy);

    enforcer.record_fuel(500_000);
    enforcer.reset_counters();

    assert_eq!(enforcer.fuel_consumed(), 0);
}

// ============================================================================
// Telemetry Integration Tests
// ============================================================================

#[test]
fn test_telemetry_step_recording() {
    let collector = TelemetryCollector::new();

    for i in 0..100 {
        collector.record_step(Duration::from_micros(100 + i * 10), 50_000);
    }

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.steps_total, 100);
    assert_eq!(snapshot.fuel_consumed_total, 5_000_000);
    assert_eq!(snapshot.step_timing.count, 100);
    assert!(snapshot.step_timing.min_us >= 100);
    assert!(snapshot.step_timing.max_us <= 1100);
}

#[test]
fn test_telemetry_reset_recording() {
    let collector = TelemetryCollector::new();

    collector.record_reset(Duration::from_millis(10), 100_000);
    collector.record_reset(Duration::from_millis(15), 150_000);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.resets_total, 2);
    assert_eq!(snapshot.fuel_consumed_total, 250_000);
    assert_eq!(snapshot.reset_timing.count, 2);
}

#[test]
fn test_telemetry_budget_overruns() {
    let collector = TelemetryCollector::new();

    collector.record_budget_overrun(BudgetType::Fuel, 1_000_000, 1_500_000, "step");
    collector.record_budget_overrun(
        BudgetType::Memory,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        "alloc",
    );
    collector.record_budget_overrun(BudgetType::Timeout, 100, 150, "step");

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.budget_overruns.len(), 3);
    assert_eq!(snapshot.budget_overruns.get("fuel"), Some(&1));
    assert_eq!(snapshot.budget_overruns.get("memory"), Some(&1));
    assert_eq!(snapshot.budget_overruns.get("timeout"), Some(&1));
}

#[test]
fn test_telemetry_capability_denials() {
    let collector = TelemetryCollector::new();

    collector.record_capability_denial("network", "Denied by policy", None);
    collector.record_capability_denial("filesystem_read", "Path not allowed", Some("/etc/passwd"));
    collector.record_capability_denial("network", "Denied by policy", None);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.capability_denials.get("network"), Some(&2));
    assert_eq!(snapshot.capability_denials.get("filesystem_read"), Some(&1));
}

#[test]
fn test_telemetry_trap_recording() {
    let collector = TelemetryCollector::new();

    collector.record_trap("unreachable", "unreachable code", Some("step"));
    collector.record_trap("oom", "out of memory", None);
    collector.record_trap("unreachable", "unreachable code", None);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.traps.get("unreachable"), Some(&2));
    assert_eq!(snapshot.traps.get("oom"), Some(&1));
}

#[test]
fn test_telemetry_rates() {
    let collector = TelemetryCollector::new();

    // 10 steps, 2 overruns, 1 trap
    for _ in 0..10 {
        collector.record_step(Duration::from_micros(100), 50_000);
    }
    collector.record_budget_overrun(BudgetType::Fuel, 1_000_000, 1_500_000, "step");
    collector.record_budget_overrun(BudgetType::Timeout, 100, 150, "step");
    collector.record_trap("unreachable", "trap", None);

    let snapshot = collector.snapshot();
    // Overrun rate: 2 / 10 = 0.2
    assert!((snapshot.overrun_rate() - 0.2).abs() < 0.001);
    // Trap rate: 1 / 10 = 0.1
    assert!((snapshot.trap_rate() - 0.1).abs() < 0.001);
}

#[test]
fn test_telemetry_serialization() {
    let collector = TelemetryCollector::new();
    collector.record_step(Duration::from_micros(100), 50_000);
    collector.record_budget_overrun(BudgetType::Fuel, 1_000_000, 1_500_000, "step");

    let snapshot = collector.snapshot();
    let json = serde_json::to_string_pretty(&snapshot).unwrap();

    assert!(json.contains("steps_total"));
    assert!(json.contains("budget_overruns"));

    // Round-trip
    let parsed: PolicyTelemetry = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.steps_total, snapshot.steps_total);
}

#[test]
fn test_telemetry_events() {
    let collector = TelemetryCollector::new();

    collector.record_step(Duration::from_micros(100), 50_000);
    collector.record_trap("oom", "out of memory", None);

    let events = collector.events();
    assert_eq!(events.len(), 2);
}

#[test]
fn test_telemetry_reset() {
    let collector = TelemetryCollector::new();

    collector.record_step(Duration::from_micros(100), 50_000);
    collector.record_trap("oom", "out of memory", None);

    collector.reset();

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.steps_total, 0);
    assert!(snapshot.traps.is_empty());
}

// ============================================================================
// Security Suite Simulation Tests
// ============================================================================

#[test]
fn test_security_fuel_exhaustion_simulation() {
    // Simulate a malicious loop that exhausts fuel budget
    let policy = PolicyBuilder::new().fuel_per_step(1_000_000).build();
    let enforcer = BudgetEnforcer::new(policy);
    let collector = TelemetryCollector::new();

    // Simulate step execution consuming fuel
    let simulated_fuel = 1_500_000; // Exceeds budget
    enforcer.record_fuel(simulated_fuel);

    let result = enforcer.check_fuel("step");
    if let EnforcementResult::Exceeded(action) = result {
        if let EnforcementAction::FuelExhausted {
            consumed, budget, ..
        } = action
        {
            collector.record_budget_overrun(BudgetType::Fuel, budget, consumed, "step");
        }
    }

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.budget_overruns.get("fuel"), Some(&1));
}

#[test]
fn test_security_memory_bomb_simulation() {
    // Simulate a malicious memory allocation
    let policy = PolicyBuilder::new().max_memory_mb(64).build();
    let enforcer = BudgetEnforcer::new(policy);
    let collector = TelemetryCollector::new();

    // Simulate memory allocation exceeding limit
    let allocation = 128 * 1024 * 1024; // 128 MB
    let result = enforcer.check_memory(allocation);

    if let EnforcementResult::Denied(action) = result {
        if let EnforcementAction::MemoryExceeded { current, limit } = action {
            collector.record_budget_overrun(BudgetType::Memory, limit, current, "alloc");
        }
    }

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.budget_overruns.get("memory"), Some(&1));
}

#[test]
fn test_security_unauthorized_io_simulation() {
    // Simulate unauthorized filesystem access
    let policy = PolicyBuilder::new()
        .allow_filesystem_read(&["/data"])
        .deny_network()
        .build();
    let enforcer = BudgetEnforcer::new(policy);
    let collector = TelemetryCollector::new();

    // Attempt to read unauthorized path
    let result = enforcer.check_read("/etc/passwd");
    if let EnforcementResult::Denied(action) = result {
        if let EnforcementAction::CapabilityDenied { capability, reason } = action {
            collector.record_capability_denial(&capability, &reason, Some("/etc/passwd"));
        }
    }

    // Attempt network access
    let result = enforcer.check_network();
    if let EnforcementResult::Denied(action) = result {
        if let EnforcementAction::CapabilityDenied { capability, reason } = action {
            collector.record_capability_denial(&capability, &reason, None);
        }
    }

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.capability_denials.get("filesystem_read"), Some(&1));
    assert_eq!(snapshot.capability_denials.get("network"), Some(&1));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_policy_error_types() {
    // Budget exceeded
    let err = PolicyError::budget_exceeded("fuel", 1_000_000, 1_500_000);
    assert!(err.is_budget_error());
    assert!(!err.is_capability_error());

    // Capability denied
    let err = PolicyError::capability_denied("network");
    assert!(!err.is_budget_error());
    assert!(err.is_capability_error());

    // Timeout exceeded
    let err = PolicyError::timeout_exceeded("step", 150, 100);
    assert!(err.is_budget_error());

    // Memory exceeded
    let err = PolicyError::memory_exceeded(128, 64);
    assert!(err.is_budget_error());
}

#[test]
fn test_enforcement_action_to_error_conversion() {
    let actions = vec![
        EnforcementAction::FuelExhausted {
            operation: "step".to_string(),
            consumed: 1_500_000,
            budget: 1_000_000,
        },
        EnforcementAction::Timeout {
            operation: "step".to_string(),
            elapsed: Duration::from_millis(150),
            limit: Duration::from_millis(100),
        },
        EnforcementAction::MemoryExceeded {
            current: 128 * 1024 * 1024,
            limit: 64 * 1024 * 1024,
        },
        EnforcementAction::CapabilityDenied {
            capability: "network".to_string(),
            reason: "Denied".to_string(),
        },
        EnforcementAction::Trap {
            message: "unreachable".to_string(),
        },
    ];

    for action in actions {
        let error = action.to_error();
        // All should produce valid errors
        let _ = error.to_string();
    }
}

// ============================================================================
// Performance Overhead Simulation
// ============================================================================

#[test]
fn test_enforcement_overhead_simulation() {
    // Simulate overhead measurement
    let policy_disabled = PolicyBuilder::new()
        .disable_fuel()
        .disable_timeouts()
        .build();
    let enforcer_disabled = BudgetEnforcer::new(policy_disabled);

    let policy_enabled = Policy::default();
    let enforcer_enabled = BudgetEnforcer::new(policy_enabled);

    // Simulate 1000 steps with enforcement checks
    let iterations = 1000;

    // With budgets disabled
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = enforcer_disabled.check_step();
        let _ = enforcer_disabled.check_memory(32 * 1024 * 1024);
    }
    let time_disabled = start.elapsed();

    // With budgets enabled
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = enforcer_enabled.check_step();
        let _ = enforcer_enabled.check_memory(32 * 1024 * 1024);
        enforcer_enabled.record_fuel(1000);
        enforcer_enabled.record_step();
    }
    let time_enabled = start.elapsed();

    // Print overhead for visibility (not a hard assertion)
    println!(
        "Enforcement overhead: disabled={:?}, enabled={:?}",
        time_disabled, time_enabled
    );

    // Both should complete in reasonable time
    assert!(time_disabled < Duration::from_millis(100));
    assert!(time_enabled < Duration::from_millis(100));
}
