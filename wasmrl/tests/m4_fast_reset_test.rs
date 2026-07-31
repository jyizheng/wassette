// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! M4 Fast Reset Integration Tests
//!
//! This test suite validates the snapshot/restore and fast reset functionality.

use std::time::Duration;

use wasmrl_runtime::{
    FastResetConfig, FastResetManager, InstanceHandle, ReplayConfig, ReplayManager,
    SharedSnapshotCache, SnapshotCache, SnapshotKey,
};
use wasmrl_wit::SnapshotData;

// ============================================================================
// Snapshot Cache Integration Tests
// ============================================================================

#[test]
fn test_snapshot_cache_basic_operations() {
    let mut cache = SnapshotCache::with_byte_limit(10, 1024 * 1024);
    let key = SnapshotKey::from_id_seed("test-component", 42);

    // Insert snapshot
    let data = vec![1, 2, 3, 4, 5];
    cache.put(key.clone(), SnapshotData::new(data.clone()));

    // Retrieve snapshot
    let cached = cache.get(&key).unwrap();
    assert_eq!(cached.data, data);
}

#[test]
fn test_snapshot_cache_lru_eviction() {
    // Small cache with 3 entries max
    let mut cache = SnapshotCache::with_byte_limit(3, 1024 * 1024);

    // Insert 4 entries
    for i in 0..4u64 {
        let key = SnapshotKey::from_id_seed("comp", i);
        cache.put(key, SnapshotData::new(vec![i as u8; 10]));
    }

    // First entry should be evicted
    let first_key = SnapshotKey::from_id_seed("comp", 0);
    assert!(cache.get(&first_key).is_none());

    // Last three should still exist
    for i in 1..4u64 {
        let key = SnapshotKey::from_id_seed("comp", i);
        assert!(cache.get(&key).is_some());
    }
}

#[test]
fn test_snapshot_cache_byte_limit() {
    // Small byte limit (50 bytes)
    let mut cache = SnapshotCache::with_byte_limit(100, 50);

    // Insert entries until byte limit forces eviction
    for i in 0..10u64 {
        let key = SnapshotKey::from_id_seed("comp", i);
        cache.put(key, SnapshotData::new(vec![i as u8; 20])); // Each entry ~20 bytes
    }

    // Should have at most ~2-3 entries due to byte limit
    assert!(cache.len() <= 3, "cache len = {}", cache.len());
}

#[test]
fn test_shared_snapshot_cache_thread_safety() {
    use std::thread;

    let cache = SharedSnapshotCache::with_byte_limit(100, 1024 * 1024);
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let cache = cache.clone();
            thread::spawn(move || {
                for i in 0..100 {
                    let key = SnapshotKey::from_id_seed(format!("thread-{}", thread_id), i);
                    cache.put(key.clone(), SnapshotData::new(vec![i as u8; 10]));
                    let _ = cache.get(&key);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have entries from all threads
    assert!(cache.len() > 0);
}

// ============================================================================
// Fast Reset Manager Integration Tests
// ============================================================================

#[test]
fn test_fast_reset_manager_basic() {
    let config = FastResetConfig {
        enabled: true,
        auto_cache: true,
        max_snapshot_size: 1024 * 1024,
        cache_capacity: 10,
        cache_max_bytes: 1024 * 1024,
    };

    let mut manager = FastResetManager::new(config);
    let key = SnapshotKey::from_id_seed("env", 42);

    // Cache a snapshot
    let snapshot = vec![1, 2, 3, 4, 5];
    manager.cache_initial_state(&key, snapshot.clone());

    // Should be able to retrieve it
    let cached = manager.get_cached_state(&key);
    assert!(cached.is_some());
    assert_eq!(cached.unwrap(), snapshot);
}

#[test]
fn test_fast_reset_metrics() {
    let config = FastResetConfig::default();
    let mut manager = FastResetManager::new(config);

    // Simulate some resets
    manager.record_full_reset(Duration::from_millis(100));
    manager.record_fast_reset(Duration::from_millis(10));
    manager.record_fast_reset(Duration::from_millis(15));

    let metrics = manager.metrics();
    assert_eq!(metrics.full_resets, 1);
    assert_eq!(metrics.fast_resets, 2);
    assert!(metrics.speedup_ratio() > 1.0);
}

#[test]
fn test_fast_reset_disabled() {
    let config = FastResetConfig {
        enabled: false,
        auto_cache: false,
        max_snapshot_size: 0,
        cache_capacity: 0,
        cache_max_bytes: 0,
    };

    let mut manager = FastResetManager::new(config);

    // Should not cache when disabled
    let key = SnapshotKey::from_id_seed("env", 42);
    manager.cache_initial_state(&key, vec![1, 2, 3]);

    assert!(manager.get_cached_state(&key).is_none());
}

// ============================================================================
// Replay Manager Integration Tests
// ============================================================================

#[test]
fn test_replay_recorder_basic() {
    use wasmrl_runtime::ReplayRecorder;

    let config = ReplayConfig {
        enabled: true,
        snapshot_interval: 10,
        max_snapshots: 5,
        record_observations: false,
        ..Default::default()
    };

    let mut recorder = ReplayRecorder::new(InstanceHandle { id: 0 }, config);

    // Record initial state
    recorder.record_initial_state(vec![1, 2, 3], 42);

    // Record some actions
    for i in 0..25 {
        let action = vec![i as u8];
        recorder.record_action(action, 0.1, i % 10 == 0, false);

        // Simulate periodic snapshots
        if i % 10 == 0 {
            recorder.record_checkpoint(vec![i as u8; 10]);
        }
    }

    // Should have recorded everything
    assert!(recorder.action_count() >= 25);
    assert!(recorder.checkpoint_count() >= 2);
}

#[test]
fn test_replay_manager_multi_instance() {
    let config = ReplayConfig::default();
    let mut manager = ReplayManager::new(config);

    // Create recorders for multiple instances
    for i in 0..5 {
        manager.create_recorder(i);
    }

    // Record actions on each
    for i in 0..5u64 {
        if let Some(recorder) = manager.get_recorder_mut(i) {
            recorder.record_initial_state(vec![i as u8], i * 10);
            recorder.record_action(vec![1], 0.5, false, false);
        }
    }

    // All should have data
    for i in 0..5u64 {
        let data = manager.get_replay_data(i);
        assert!(data.is_some());
    }
}

#[test]
fn test_replay_data_serialization() {
    use wasmrl_runtime::{ReplayData, ReplayRecorder};

    let config = ReplayConfig::default();
    let mut recorder = ReplayRecorder::new(InstanceHandle { id: 0 }, config);

    recorder.record_initial_state(vec![1, 2, 3], 42);
    recorder.record_action(vec![4], 1.0, true, false);

    let data = recorder.to_replay_data().unwrap();
    let json = serde_json::to_string(&data).unwrap();

    // Should be able to round-trip
    let parsed: ReplayData = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.initial_seed, 42);
    assert_eq!(parsed.actions.len(), 1);
}

// ============================================================================
// Reset Performance Simulation Tests
// ============================================================================

#[test]
fn test_fast_reset_performance_simulation() {
    // Simulate the performance benefit of fast reset
    let config = FastResetConfig::default();
    let mut manager = FastResetManager::new(config);
    let key = SnapshotKey::from_id_seed("counter_env", 12345);

    // First reset is "full" - expensive
    let full_reset_time = Duration::from_millis(50);
    let initial_state = vec![0u8; 1000]; // 1KB state

    // Cache the initial state
    manager.cache_initial_state(&key, initial_state.clone());
    manager.record_full_reset(full_reset_time);

    // Subsequent resets use cached state - fast
    for _ in 0..100 {
        let fast_reset_time = Duration::from_micros(500); // 0.5ms
        let cached = manager.get_cached_state(&key);
        assert!(cached.is_some());
        manager.record_fast_reset(fast_reset_time);
    }

    let metrics = manager.metrics();
    println!(
        "Fast reset speedup: {:.2}x (full: {:?}, fast: {:?})",
        metrics.speedup_ratio(),
        metrics.avg_full_reset_time(),
        metrics.avg_fast_reset_time()
    );

    // Should see significant speedup
    assert!(
        metrics.speedup_ratio() > 10.0,
        "Expected speedup > 10x, got {:.2}x",
        metrics.speedup_ratio()
    );
}

#[test]
fn test_reset_heavy_scenario() {
    // Simulate a reset-heavy workload (short episodes)
    let config = FastResetConfig::default();
    let mut manager = FastResetManager::new(config);

    let episode_length = 10; // Short episodes
    let num_episodes = 100;

    let mut total_full_reset_time = Duration::ZERO;
    let mut total_fast_reset_time = Duration::ZERO;

    for episode in 0..num_episodes {
        let key = SnapshotKey::from_id_seed("reset_heavy_env", episode as u64);

        // First time seeing this seed - full reset
        if manager.get_cached_state(&key).is_none() {
            let full_time = Duration::from_millis(20);
            let state = vec![0u8; 5000]; // 5KB state
            manager.cache_initial_state(&key, state);
            manager.record_full_reset(full_time);
            total_full_reset_time += full_time;
        } else {
            // Fast reset
            let fast_time = Duration::from_micros(200);
            manager.record_fast_reset(fast_time);
            total_fast_reset_time += fast_time;
        }

        // Simulate episode steps (not timed here)
        let _steps = episode_length;
    }

    let metrics = manager.metrics();
    println!(
        "Reset-heavy scenario: {} full resets, {} fast resets",
        metrics.full_resets, metrics.fast_resets
    );
    println!(
        "Total reset time: full={:?}, fast={:?}",
        total_full_reset_time, total_fast_reset_time
    );
}

// ============================================================================
// Cache Hit Rate Tests
// ============================================================================

#[test]
fn test_cache_hit_rate_tracking() {
    let mut cache = SnapshotCache::with_byte_limit(10, 1024 * 1024);

    // Insert some entries
    for i in 0..5u64 {
        let key = SnapshotKey::from_id_seed("env", i);
        cache.put(key, SnapshotData::new(vec![i as u8; 10]));
    }

    // 5 hits
    for i in 0..5u64 {
        let key = SnapshotKey::from_id_seed("env", i);
        cache.get(&key);
    }

    // 3 misses
    for i in 10..13u64 {
        let key = SnapshotKey::from_id_seed("env", i);
        cache.get(&key);
    }

    let hit_rate = cache.hit_rate();
    // 5 hits / 8 total = 62.5%
    assert!(hit_rate > 0.5 && hit_rate < 0.7, "hit_rate = {}", hit_rate);
}

// ============================================================================
// Checkpoint Replay Tests
// ============================================================================

#[test]
fn test_replay_from_checkpoint() {
    use wasmrl_runtime::ReplayRecorder;

    let config = ReplayConfig {
        enabled: true,
        snapshot_interval: 5,
        max_snapshots: 10,
        record_observations: true,
        ..Default::default()
    };

    let mut recorder = ReplayRecorder::new(InstanceHandle { id: 0 }, config);

    // Initial state
    recorder.record_initial_state(vec![0; 100], 42);

    // Record actions with periodic checkpoints
    for step in 0..20 {
        recorder.record_action(vec![step as u8], step as f64 * 0.1, step == 19, false);

        if step % 5 == 0 && step > 0 {
            recorder.record_checkpoint(vec![step as u8; 100]);
        }
    }

    // Find checkpoint closest to step 12
    let checkpoint = recorder.get_checkpoint_before(12);

    assert!(checkpoint.is_some());
    println!(
        "Can replay from checkpoint at step {}",
        checkpoint.unwrap().step
    );
}

// ============================================================================
// Memory Pressure Tests
// ============================================================================

#[test]
fn test_cache_under_memory_pressure() {
    // Very small cache (10KB)
    let mut cache = SnapshotCache::with_byte_limit(1000, 10 * 1024);

    // Insert large entries until eviction
    let mut inserted = 0;
    for i in 0..100u64 {
        let key = SnapshotKey::from_id_seed("env", i);
        cache.put(key, SnapshotData::new(vec![i as u8; 1024])); // 1KB each
        inserted += 1;
    }

    // Should have evicted entries to stay under limit
    let current_bytes = cache.current_bytes();
    assert!(
        current_bytes <= 10 * 1024,
        "current_bytes = {} (limit 10KB)",
        current_bytes
    );
    println!(
        "Inserted {} entries, {} remain in cache ({} bytes)",
        inserted,
        cache.len(),
        current_bytes
    );
}

#[test]
fn test_cache_eviction_order() {
    // Cache with LRU eviction
    let mut cache = SnapshotCache::with_byte_limit(3, 1024 * 1024);

    // Insert A, B, C
    for (i, name) in ["A", "B", "C"].iter().enumerate() {
        let key = SnapshotKey::from_id_seed(*name, 0);
        cache.put(key, SnapshotData::new(vec![i as u8]));
    }

    // Access A (makes it recently used)
    let key_a = SnapshotKey::from_id_seed("A", 0);
    cache.get(&key_a);

    // Insert D - should evict B (least recently used)
    let key_d = SnapshotKey::from_id_seed("D", 0);
    cache.put(key_d.clone(), SnapshotData::new(vec![3]));

    // A, C, D should exist; B should be evicted
    let key_b = SnapshotKey::from_id_seed("B", 0);
    assert!(cache.get(&key_a).is_some(), "A should exist");
    assert!(cache.get(&key_b).is_none(), "B should be evicted");
    assert!(
        cache.get(&SnapshotKey::from_id_seed("C", 0)).is_some(),
        "C should exist"
    );
    assert!(cache.get(&key_d).is_some(), "D should exist");
}
