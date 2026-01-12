// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Integration tests for WasmRL.

use wasmrl_sdk_rust::{DeterministicRng, SnapshotHelper, TensorDecoder, TensorEncoder};
use wasmrl_wit::{
    BatchStepResult, DType, EnvConfig, EnvHandle, SnapshotData, StepResult, Tensor, WIT_PACKAGE,
    WIT_VERSION,
};

#[test]
fn test_all_crates_compile() {
    // This test verifies that all crates can be imported
    assert!(true, "All crates compiled successfully");
}

#[test]
fn test_wasmrl_wit_available() {
    // Verify wasmrl-wit is available as a workspace member
    assert_eq!(WIT_VERSION, "0.1.0", "WIT version should match");
    assert_eq!(WIT_PACKAGE, "wasmrl:env@0.1.0", "WIT package should match");
}

#[test]
fn test_wasmrl_runtime_available() {
    // Verify wasmrl-runtime is available
    let config = wasmrl_runtime::RuntimeConfig::new();
    assert_eq!(config.max_instances, 256);
    assert_eq!(config.max_memory_mb, 512);
}

#[test]
fn test_wasmrl_sdk_available() {
    // Verify wasmrl-sdk-rust is available
    let mut rng = DeterministicRng::new(42);
    let val = rng.next_in_range(100);
    assert!(val < 100);
}

// M1 Integration Tests

#[test]
fn test_tensor_creation_and_validation() {
    // Test tensor creation with various dtypes
    let float_tensor = Tensor::zeros(DType::Float32, vec![4, 4]);
    assert!(float_tensor.is_valid());
    assert_eq!(float_tensor.num_elements(), 16);
    assert_eq!(float_tensor.byte_size(), 64);

    let image_tensor = Tensor::zeros(DType::Uint8, vec![84, 84, 4]);
    assert!(image_tensor.is_valid());
    assert_eq!(image_tensor.num_elements(), 84 * 84 * 4);
}

#[test]
fn test_tensor_encoding_roundtrip() {
    // Test encoding and decoding tensors
    let original_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let encoded = TensorEncoder::encode_f32(&original_data);
    let decoded = TensorDecoder::decode_f32(&encoded).unwrap();
    assert_eq!(original_data, decoded);
}

#[test]
fn test_step_result_semantics() {
    let obs = Tensor::zeros(DType::Float32, vec![4]);

    // Terminal state
    let terminal = StepResult::new(obs.clone(), 1.0, true, false);
    assert!(terminal.done());
    assert!(terminal.terminated);
    assert!(!terminal.truncated);

    // Truncated state
    let truncated = StepResult::new(obs.clone(), 0.0, false, true);
    assert!(truncated.done());
    assert!(!truncated.terminated);
    assert!(truncated.truncated);

    // Ongoing state
    let ongoing = StepResult::new(obs, 0.5, false, false);
    assert!(!ongoing.done());
}

#[test]
fn test_env_handle_uniqueness() {
    let handle1 = EnvHandle::new(1);
    let handle2 = EnvHandle::new(2);
    let handle1_copy = EnvHandle::new(1);

    assert_ne!(handle1, handle2);
    assert_eq!(handle1, handle1_copy);
}

#[test]
fn test_batch_step_result_validation() {
    let mut batch = BatchStepResult::with_capacity(2);

    // Add two complete results
    batch.observations.push(Tensor::zeros(DType::Float32, vec![4]));
    batch.observations.push(Tensor::zeros(DType::Float32, vec![4]));
    batch.rewards.push(1.0);
    batch.rewards.push(2.0);
    batch.terminated.push(false);
    batch.terminated.push(true);
    batch.truncated.push(false);
    batch.truncated.push(false);
    batch.infos.push(None);
    batch.infos.push(Some("done".to_string()));

    assert!(batch.is_valid());
    assert_eq!(batch.len(), 2);
}

#[test]
fn test_snapshot_versioning() {
    let snapshot = SnapshotData::new(vec![1, 2, 3, 4]);
    assert!(snapshot.is_compatible());
    assert_eq!(snapshot.version, SnapshotData::CURRENT_VERSION);
}

#[test]
fn test_deterministic_rng_reproducibility() {
    // Critical test: same seed must produce same sequence
    let mut rng1 = DeterministicRng::new(12345);
    let mut rng2 = DeterministicRng::new(12345);

    let seq1: Vec<u64> = (0..100).map(|_| rng1.next()).collect();
    let seq2: Vec<u64> = (0..100).map(|_| rng2.next()).collect();

    assert_eq!(seq1, seq2, "Same seed must produce identical sequence");
}

#[test]
fn test_snapshot_helper_state_roundtrip() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct EnvState {
        step_count: u32,
        position: Vec<f32>,
        done: bool,
    }

    let state = EnvState {
        step_count: 42,
        position: vec![1.0, 2.0, 3.0],
        done: false,
    };

    let serialized = SnapshotHelper::serialize(&state).unwrap();
    let restored: EnvState = SnapshotHelper::deserialize(&serialized).unwrap();

    assert_eq!(state, restored);
}

#[test]
fn test_env_config_json() {
    let config = EnvConfig::new(r#"{"max_steps": 1000, "reward_scale": 0.1}"#);
    assert!(config.config_json.contains("max_steps"));
    assert!(config.config_json.contains("reward_scale"));
}

// M2 Integration Tests - Counter Environment

#[test]
fn test_counter_env_full_episode() {
    use counter_env::CounterEnv;

    let mut env = CounterEnv::new();
    env.init(r#"{"initial_value": 0, "target": 3, "max_steps": 100}"#)
        .unwrap();

    let obs = env.reset(42).unwrap();
    assert_eq!(obs.len(), 4); // Single f32

    // Increment 3 times to reach target
    for _ in 0..3 {
        let action = TensorEncoder::encode_i32(&[1]); // Increment
        let result = env.step(&action).unwrap();
        if result.terminated {
            assert_eq!(result.reward, 1.0);
            break;
        }
    }
}

#[test]
fn test_counter_env_observation_space() {
    use counter_env::CounterEnv;

    let mut env = CounterEnv::new();
    env.init("{}").unwrap();
    env.reset(42).unwrap();

    let obs_space = env.observation_space();
    assert_eq!(obs_space.len(), 4); // Shape [1] encoded as f32
}

#[test]
fn test_counter_env_action_space() {
    use counter_env::CounterEnv;

    let mut env = CounterEnv::new();
    env.init("{}").unwrap();
    env.reset(42).unwrap();

    let action_space = env.action_space();
    assert_eq!(action_space.len(), 4); // Number of actions as i32
}

// M2 Integration Tests - Security Suite

#[test]
fn test_malicious_loop_env_init() {
    use malicious_loop_env::MaliciousLoopEnv;

    let mut env = MaliciousLoopEnv::new();
    // Use step trigger so init doesn't loop
    let result = env.init(r#"{"loop_on": "step"}"#);
    assert!(result.is_ok());
}

#[test]
fn test_malicious_loop_env_partial_execution() {
    use malicious_loop_env::MaliciousLoopEnv;

    let mut env = MaliciousLoopEnv::new();
    env.init(r#"{"loop_on": "step", "iterations_before_loop": 5}"#)
        .unwrap();
    env.reset(42).unwrap();

    // First 5 steps should succeed
    for i in 0..5 {
        let action = vec![0, 0, 0, 0];
        let result = env.step(&action);
        assert!(result.is_ok(), "Step {} should succeed", i);
    }
}

#[test]
fn test_malicious_memory_env_init() {
    use malicious_memory_env::MaliciousMemoryEnv;

    let mut env = MaliciousMemoryEnv::new();
    let result = env.init(r#"{"alloc_mb_per_step": 1}"#);
    assert!(result.is_ok());
}

#[test]
fn test_malicious_memory_env_controlled_alloc() {
    use malicious_memory_env::MaliciousMemoryEnv;

    let mut env = MaliciousMemoryEnv::new();
    // Use small allocation for testing
    env.init(r#"{"alloc_mb_per_step": 1, "keep_allocations": false}"#)
        .unwrap();
    env.reset(42).unwrap();

    // Should work without keeping allocations
    let action = vec![0, 0, 0, 0];
    let result = env.step(&action);
    assert!(result.is_ok());
}

