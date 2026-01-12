// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Determinism tests for WasmRL environments.
//!
//! These tests verify that environments produce identical trajectories
//! when given the same seed and action sequence.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use counter_env::CounterEnv;
use wasmrl_sdk_rust::TensorEncoder;

/// Calculate hash of a trajectory.
fn trajectory_hash(trajectory: &[(Vec<u8>, f64, bool, bool)]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for (obs, reward, terminated, truncated) in trajectory {
        obs.hash(&mut hasher);
        reward.to_bits().hash(&mut hasher);
        terminated.hash(&mut hasher);
        truncated.hash(&mut hasher);
    }
    hasher.finish()
}

/// Run an episode and collect trajectory.
fn run_episode(
    env: &mut CounterEnv,
    seed: u64,
    actions: &[i32],
) -> Vec<(Vec<u8>, f64, bool, bool)> {
    env.reset(seed).unwrap();

    let mut trajectory = Vec::new();
    for &action in actions {
        let action_bytes = TensorEncoder::encode_i32(&[action]);
        let result = env.step(&action_bytes).unwrap();
        trajectory.push((
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
        ));
        if result.terminated || result.truncated {
            break;
        }
    }
    trajectory
}

#[test]
fn test_counter_env_determinism_20_runs() {
    // Acceptance criteria: 20 runs with same seed produce identical trajectory hash
    const NUM_RUNS: usize = 20;
    const SEED: u64 = 42;
    
    // Fixed action sequence
    let actions: Vec<i32> = vec![1, 1, 1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1];
    
    let mut hashes = Vec::with_capacity(NUM_RUNS);
    
    for run in 0..NUM_RUNS {
        let mut env = CounterEnv::new();
        env.init(r#"{"initial_value": 0, "target": 100, "max_steps": 50}"#)
            .unwrap();
        
        let trajectory = run_episode(&mut env, SEED, &actions);
        let hash = trajectory_hash(&trajectory);
        hashes.push(hash);
        
        // Verify all hashes are identical
        if run > 0 {
            assert_eq!(
                hashes[0], hashes[run],
                "Run {} produced different trajectory hash: {} vs {}",
                run, hashes[0], hashes[run]
            );
        }
    }
    
    println!("✓ All {} runs produced identical trajectory hash: {}", NUM_RUNS, hashes[0]);
}

#[test]
fn test_counter_env_different_seeds_different_trajectories() {
    // Different seeds should (usually) produce different trajectories
    let actions: Vec<i32> = vec![1, 1, 1, 0, 1, 2, 1];
    
    let mut hashes = Vec::new();
    for seed in [1, 2, 3, 42, 12345] {
        let mut env = CounterEnv::new();
        env.init("{}").unwrap();
        
        let trajectory = run_episode(&mut env, seed, &actions);
        hashes.push(trajectory_hash(&trajectory));
    }
    
    // At least some should be different (initial state varies with seed)
    let unique_hashes: std::collections::HashSet<_> = hashes.iter().collect();
    assert!(
        unique_hashes.len() > 1,
        "Different seeds should produce different trajectories"
    );
}

#[test]
fn test_counter_env_snapshot_determinism() {
    // Restore from snapshot + replay should produce identical trajectory
    const SEED: u64 = 42;
    let actions: Vec<i32> = vec![1, 1, 1, 0, 1];
    
    let mut env = CounterEnv::new();
    env.init(r#"{"initial_value": 0, "target": 100}"#).unwrap();
    env.reset(SEED).unwrap();
    
    // Run a few steps
    for &action in &actions[..2] {
        let action_bytes = TensorEncoder::encode_i32(&[action]);
        env.step(&action_bytes).unwrap();
    }
    
    // Take snapshot
    let snapshot = env.snapshot().unwrap();
    
    // Run remaining steps and record trajectory
    let remaining_actions = &actions[2..];
    let trajectory1 = run_remaining(&mut env, remaining_actions);
    
    // Restore and replay
    env.restore(&snapshot).unwrap();
    let trajectory2 = run_remaining(&mut env, remaining_actions);
    
    // Trajectories should be identical
    assert_eq!(
        trajectory_hash(&trajectory1),
        trajectory_hash(&trajectory2),
        "Restored trajectory should match original"
    );
}

fn run_remaining(
    env: &mut CounterEnv,
    actions: &[i32],
) -> Vec<(Vec<u8>, f64, bool, bool)> {
    let mut trajectory = Vec::new();
    for &action in actions {
        let action_bytes = TensorEncoder::encode_i32(&[action]);
        let result = env.step(&action_bytes).unwrap();
        trajectory.push((
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
        ));
    }
    trajectory
}

#[test]
fn test_rng_cross_instance_determinism() {
    // Two separate instances with same config and seed should behave identically
    const SEED: u64 = 12345;
    let actions: Vec<i32> = vec![1, 0, 1, 1, 2, 0, 1, 1];
    
    let mut env1 = CounterEnv::new();
    let mut env2 = CounterEnv::new();
    
    env1.init("{}").unwrap();
    env2.init("{}").unwrap();
    
    let trajectory1 = run_episode(&mut env1, SEED, &actions);
    let trajectory2 = run_episode(&mut env2, SEED, &actions);
    
    assert_eq!(
        trajectory_hash(&trajectory1),
        trajectory_hash(&trajectory2),
        "Two instances with same seed must produce identical trajectories"
    );
}
