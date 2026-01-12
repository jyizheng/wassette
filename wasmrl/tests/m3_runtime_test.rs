// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Integration tests for M3 - Data Plane Runtime v0.

use std::sync::Arc;
use std::time::Duration;
use wasmrl_runtime::{
    BatchExecutor, ComponentRef, EnvRuntime, FactoryBuilder, InstanceHandle, InstancePool,
    InstanceStatus, LatencyStats, PolicyConfig, PoolStats, RuntimeConfig, RuntimeError,
    RuntimeMetrics, SharedPool, Timer, WasmEnvFactory,
};
use wasmrl_wit::{BatchStepResult, DType, EnvConfig, SnapshotData, StepResult, Tensor};

// ============================================================================
// Runtime Configuration Tests
// ============================================================================

#[test]
fn test_runtime_config_builder_pattern() {
    let config = RuntimeConfig::new()
        .with_max_instances(128)
        .with_max_memory_mb(256)
        .with_fuel_per_step(1_000_000)
        .with_fuel_per_reset(500_000)
        .with_step_timeout(Duration::from_secs(1))
        .with_reset_timeout(Duration::from_millis(500))
        .with_epoch_interruption(100)
        .with_prewarming(16);

    assert_eq!(config.max_instances, 128);
    assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
    assert_eq!(config.fuel_per_step, 1_000_000);
    assert_eq!(config.fuel_per_reset, 500_000);
    assert!(config.fuel_enabled());
    assert_eq!(config.step_timeout, Some(Duration::from_secs(1)));
    assert_eq!(config.reset_timeout, Some(Duration::from_millis(500)));
    assert!(config.enable_epoch_interruption);
    assert_eq!(config.epoch_deadline, 100);
    assert!(config.prewarm_instances);
    assert_eq!(config.prewarm_count, 16);
}

#[test]
fn test_policy_config_application() {
    let policy = PolicyConfig {
        max_memory_mb: Some(64),
        fuel_per_step: Some(100_000),
        timeout_ms_step: Some(50),
        timeout_ms_reset: Some(100),
        ..Default::default()
    };

    let mut config = RuntimeConfig::new();
    policy.apply_to(&mut config);

    assert_eq!(config.max_memory_bytes, 64 * 1024 * 1024);
    assert_eq!(config.fuel_per_step, 100_000);
    assert_eq!(config.step_timeout, Some(Duration::from_millis(50)));
    assert_eq!(config.reset_timeout, Some(Duration::from_millis(100)));
}

// ============================================================================
// Instance Pool Tests
// ============================================================================

#[test]
fn test_instance_pool_allocation_lifecycle() {
    let mut pool = InstancePool::new(10);

    // Allocate instances
    let h1 = pool.allocate().unwrap();
    let h2 = pool.allocate().unwrap();
    let h3 = pool.allocate().unwrap();

    assert_ne!(h1.id, h2.id);
    assert_ne!(h2.id, h3.id);

    // Check stats
    let stats = pool.stats();
    assert_eq!(stats.active, 3);
    assert_eq!(stats.ready, 0);
    assert_eq!(stats.total_created, 3);

    // Release one
    pool.release(h2).unwrap();

    let stats = pool.stats();
    assert_eq!(stats.active, 3);
    assert_eq!(stats.ready, 1);

    // Allocate should reuse released instance
    let h4 = pool.allocate().unwrap();
    assert_eq!(h4.id, h2.id);
}

#[test]
fn test_instance_pool_capacity_enforcement() {
    let mut pool = InstancePool::new(3);

    let _h1 = pool.allocate().unwrap();
    let _h2 = pool.allocate().unwrap();
    let _h3 = pool.allocate().unwrap();

    // Should fail - at capacity
    let result = pool.allocate();
    assert!(matches!(result, Err(RuntimeError::PoolExhausted(3))));
}

#[test]
fn test_instance_pool_error_recycling() {
    let mut pool = InstancePool::new(5);

    let h1 = pool.allocate().unwrap();

    // Mark as fatal error
    pool.mark_error(h1, true).unwrap();

    let info = pool.get_info(h1).unwrap();
    assert_eq!(info.status, InstanceStatus::ErrorFatal);

    // Recycle
    let recycled = pool.recycle(h1).unwrap();
    assert_eq!(recycled.handle, h1);

    // Stats should reflect recycling
    let stats = pool.stats();
    assert_eq!(stats.recycled, 1);
    assert_eq!(stats.active, 0);
}

#[test]
fn test_shared_pool_thread_safety() {
    use std::thread;

    let pool = SharedPool::new(100);
    let mut handles = vec![];

    // Spawn 4 threads each doing 25 allocate/release cycles
    for _ in 0..4 {
        let pool_clone = pool.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..25 {
                let h = pool_clone.allocate().unwrap();
                pool_clone.record_step(h).unwrap();
                pool_clone.release(h).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let stats = pool.stats();
    assert!(stats.total_created <= 100);
}

// ============================================================================
// Metrics Tests
// ============================================================================

#[test]
fn test_latency_stats_percentiles() {
    let mut stats = LatencyStats::new(1000);

    // Add samples: 1, 2, ..., 1000 microseconds
    for i in 1..=1000 {
        stats.record(Duration::from_micros(i));
    }

    assert_eq!(stats.count, 1000);

    // p50 should be around 500
    let p50 = stats.p50();
    assert!(
        p50.as_micros() >= 490 && p50.as_micros() <= 510,
        "p50 = {} us",
        p50.as_micros()
    );

    // p99 should be around 990
    let p99 = stats.p99();
    assert!(
        p99.as_micros() >= 985 && p99.as_micros() <= 1000,
        "p99 = {} us",
        p99.as_micros()
    );

    // Mean should be around 500
    let mean = stats.mean();
    assert!(
        mean.as_micros() >= 490 && mean.as_micros() <= 510,
        "mean = {} us",
        mean.as_micros()
    );
}

#[test]
fn test_runtime_metrics_comprehensive() {
    let mut metrics = RuntimeMetrics::new();

    // Record steps
    for _ in 0..100 {
        metrics.record_step(Duration::from_micros(100));
    }

    // Record resets
    for _ in 0..10 {
        metrics.record_reset(Duration::from_micros(500));
    }

    // Record batch operations
    metrics.record_batch_step(Duration::from_millis(10), 256);

    // Record errors
    metrics.record_trap();
    metrics.record_timeout();
    metrics.record_fuel_exhausted();
    metrics.record_instance_recycled();

    assert_eq!(metrics.steps_completed, 100 + 256);
    assert_eq!(metrics.resets_completed, 10);
    assert_eq!(metrics.traps_count, 1);
    assert_eq!(metrics.timeouts_count, 1);
    assert_eq!(metrics.fuel_exhausted_count, 1);
    assert_eq!(metrics.instances_recycled, 1);

    // Check summary
    let summary = metrics.summary();
    assert!(summary.throughput_steps_per_sec > 0.0);
    assert!(summary.step_p50_us > 0);
}

#[test]
fn test_timer_accuracy() {
    let timer = Timer::start();
    std::thread::sleep(Duration::from_millis(50));
    let elapsed = timer.stop();

    // Should be at least 50ms
    assert!(
        elapsed >= Duration::from_millis(50),
        "elapsed = {:?}",
        elapsed
    );
    // Should be less than 100ms (allowing for some overhead)
    assert!(
        elapsed < Duration::from_millis(100),
        "elapsed = {:?}",
        elapsed
    );
}

// ============================================================================
// Factory Builder Tests
// ============================================================================

#[test]
fn test_factory_builder_configuration() {
    let builder = FactoryBuilder::new()
        .max_instances(64)
        .max_memory_mb(128)
        .fuel_per_step(500_000)
        .prewarm(8)
        .policy(PolicyConfig {
            max_memory_mb: Some(64),
            network_enabled: false,
            ..Default::default()
        });

    // Check builder state
    assert_eq!(builder.config.max_instances, 64);
    assert_eq!(builder.config.max_memory_bytes, 128 * 1024 * 1024);
    assert_eq!(builder.config.fuel_per_step, 500_000);
    assert!(builder.config.prewarm_instances);
    assert_eq!(builder.config.prewarm_count, 8);
    assert!(!builder.policy.network_enabled);
}

// ============================================================================
// Component Reference Tests
// ============================================================================

#[test]
fn test_component_ref_variants() {
    // Bytes variant
    let bytes = vec![0x00, 0x61, 0x73, 0x6d]; // Wasm magic bytes
    let ref_bytes = ComponentRef::from_bytes(bytes);
    assert!(matches!(ref_bytes, ComponentRef::Bytes(_)));

    // File variant
    let ref_file = ComponentRef::from_file("/path/to/component.wasm");
    if let ComponentRef::File(path) = ref_file {
        assert_eq!(path, "/path/to/component.wasm");
    } else {
        panic!("Expected File variant");
    }

    // OCI variant
    let ref_oci = ComponentRef::from_oci("ghcr.io/example/env:v1.0.0");
    if let ComponentRef::Oci(reference) = ref_oci {
        assert_eq!(reference, "ghcr.io/example/env:v1.0.0");
    } else {
        panic!("Expected Oci variant");
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_runtime_error_variants() {
    // Component load error
    let err = RuntimeError::component_load("Invalid component format");
    assert!(err.to_string().contains("Invalid component format"));

    // Instance not found
    let err = RuntimeError::instance_not_found(42);
    assert!(err.to_string().contains("42"));

    // Pool exhausted
    let err = RuntimeError::pool_exhausted(256);
    assert!(err.to_string().contains("256"));

    // Instance trapped
    let err = RuntimeError::instance_trapped(1, "out of bounds");
    assert!(err.to_string().contains("Instance 1"));
    assert!(err.to_string().contains("out of bounds"));

    // Timeout
    let err = RuntimeError::timeout("step", 1500, 1000);
    assert!(err.to_string().contains("1500ms"));
    assert!(err.to_string().contains("1000ms"));

    // Fuel exhausted
    let err = RuntimeError::fuel_exhausted("step_batch");
    assert!(err.to_string().contains("step_batch"));

    // Batch size mismatch
    let err = RuntimeError::BatchSizeMismatch {
        expected: 10,
        actual: 5,
    };
    assert!(err.to_string().contains("10"));
    assert!(err.to_string().contains("5"));

    // Memory limit exceeded
    let err = RuntimeError::MemoryLimitExceeded {
        attempted_mb: 1024,
        limit_mb: 512,
    };
    assert!(err.to_string().contains("1024"));
    assert!(err.to_string().contains("512"));
}

// ============================================================================
// Instance Status Tests
// ============================================================================

#[test]
fn test_instance_status_transitions() {
    // Available states
    assert!(InstanceStatus::Ready.is_available());
    assert!(InstanceStatus::Paused.is_available());
    assert!(!InstanceStatus::Running.is_available());
    assert!(!InstanceStatus::ErrorFatal.is_available());

    // Error states
    assert!(InstanceStatus::ErrorRecoverable.is_error());
    assert!(InstanceStatus::ErrorFatal.is_error());
    assert!(!InstanceStatus::Ready.is_error());
    assert!(!InstanceStatus::Running.is_error());

    // Recyclable states
    assert!(InstanceStatus::ErrorFatal.can_recycle());
    assert!(InstanceStatus::ErrorRecoverable.can_recycle());
    assert!(InstanceStatus::Terminated.can_recycle());
    assert!(!InstanceStatus::Ready.can_recycle());
    assert!(!InstanceStatus::Running.can_recycle());
}

// ============================================================================
// Batch Step Result Tests
// ============================================================================

#[test]
fn test_batch_step_result_construction() {
    let mut result = BatchStepResult::with_capacity(4);

    for i in 0..4 {
        result
            .observations
            .push(Tensor::zeros(DType::Float32, vec![84, 84, 4]));
        result.rewards.push(i as f64 * 0.1);
        result.terminated.push(i == 3);
        result.truncated.push(false);
        result.infos.push(None);
    }

    assert_eq!(result.len(), 4);
    assert!(!result.is_empty());
    assert!(result.is_valid());
    assert_eq!(result.terminated[3], true);
}

// ============================================================================
// Snapshot Tests
// ============================================================================

#[test]
fn test_snapshot_data_versioning() {
    let data = vec![1, 2, 3, 4];
    let snapshot = SnapshotData::new(data.clone());

    assert_eq!(snapshot.version, SnapshotData::CURRENT_VERSION);
    assert!(snapshot.is_compatible());
    assert_eq!(snapshot.data, data);

    // Test incompatible version
    let old_snapshot = SnapshotData {
        version: 0,
        data: vec![],
    };
    assert!(!old_snapshot.is_compatible());
}

// ============================================================================
// Integration: Pool + Metrics
// ============================================================================

#[test]
fn test_pool_with_metrics_tracking() {
    let mut pool = InstancePool::new(10);
    let mut metrics = RuntimeMetrics::new();

    // Simulate workload
    for i in 0..5 {
        let handle = pool.allocate().unwrap();
        pool.record_step(handle).unwrap();
        pool.record_reset(handle).unwrap();
        metrics.record_step(Duration::from_micros(100 + i * 10));
        metrics.record_reset(Duration::from_micros(500 + i * 20));
        pool.release(handle).unwrap();
    }

    // One error
    let bad_handle = pool.allocate().unwrap();
    pool.mark_error(bad_handle, true).unwrap();
    metrics.record_trap();

    let stats = pool.stats();
    assert_eq!(stats.errored, 1);

    let summary = metrics.summary();
    assert_eq!(summary.steps_completed, 5);
    assert_eq!(summary.resets_completed, 5);
    assert_eq!(summary.traps_count, 1);
}

// ============================================================================
// Performance Baseline Tests
// ============================================================================

#[test]
fn test_pool_allocation_performance() {
    let mut pool = InstancePool::new(1000);

    let start = std::time::Instant::now();

    // Allocate and release 1000 times
    for _ in 0..1000 {
        let handle = pool.allocate().unwrap();
        pool.release(handle).unwrap();
    }

    let elapsed = start.elapsed();

    // Should complete in under 100ms
    assert!(
        elapsed < Duration::from_millis(100),
        "Pool operations took {:?}",
        elapsed
    );
}

#[test]
fn test_metrics_recording_performance() {
    let mut metrics = RuntimeMetrics::new();

    let start = std::time::Instant::now();

    // Record 10000 samples
    for i in 0..10000 {
        metrics.record_step(Duration::from_micros(100 + (i % 100)));
    }

    let elapsed = start.elapsed();

    // Should complete in under 50ms
    assert!(
        elapsed < Duration::from_millis(50),
        "Metrics recording took {:?}",
        elapsed
    );

    // Percentiles should still be accurate
    let summary = metrics.summary();
    assert!(summary.step_p50_us > 0);
}
