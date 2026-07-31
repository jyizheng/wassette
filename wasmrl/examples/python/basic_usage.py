#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Basic usage example for WasmRL Python bindings.

This example demonstrates how to use WasmVecEnv for parallel
environment execution without any RL training.
"""

import numpy as np

# Import wasmrl (built via maturin)
try:
    import wasmrl_py as wasmrl
except ImportError:
    print("wasmrl_py not installed. Build with: cd wasmrl/crates/wasmrl-py && maturin develop")
    exit(1)


def basic_usage():
    """Demonstrate basic WasmVecEnv usage."""
    print("=== Basic WasmRL Usage ===\n")
    
    # List available environments
    envs = wasmrl.list_available_envs("./envs")
    print(f"Available environments: {envs}")
    
    # Create configuration
    config = wasmrl.EnvConfig(
        num_envs=4,
        max_memory_mb=64,
        fuel_per_step=1_000_000,
        timeout_step_ms=100,
        auto_reset=True,
        seed=42,
    )
    print(f"\nConfiguration: {config}")
    
    # Create vectorized environment
    env = wasmrl.WasmVecEnv("./envs/counter_env.wasm", config)
    print(f"\nCreated: {env}")
    print(f"Number of envs: {env.num_envs}")
    print(f"Observation space: {env.single_observation_space}")
    print(f"Action space: {env.single_action_space}")
    
    # Reset all environments
    obs, info = env.reset(seed=42)
    print(f"\nInitial observations shape: {obs.shape}")
    print(f"Initial observations:\n{obs}")
    
    # Take random steps
    total_rewards = np.zeros(env.num_envs)
    for step in range(100):
        # Sample random actions
        actions = env.sample_actions()
        
        # Step all environments
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_rewards += rewards
        
        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: avg_reward={total_rewards.mean():.2f}")
    
    print(f"\nFinal total rewards: {total_rewards}")
    print(f"Episode rewards: {info['episode_rewards']}")
    print(f"Episode lengths: {info['episode_lengths']}")
    
    # Close environment
    env.close()
    print("\nEnvironment closed.")


def snapshot_demo():
    """Demonstrate snapshot/restore functionality."""
    print("\n=== Snapshot/Restore Demo ===\n")
    
    config = wasmrl.EnvConfig(num_envs=2, auto_reset=False)
    env = wasmrl.WasmVecEnv("./envs/counter_env.wasm", config)
    
    # Reset and take a few steps
    obs, _ = env.reset()
    for _ in range(10):
        actions = np.array([1, 1])  # Increment action
        obs, _, _, _, _ = env.step(actions)
    
    print(f"State after 10 steps: {obs}")
    
    # Take snapshots
    snapshots = env.snapshot_all()
    print(f"Took {len(snapshots)} snapshots")
    
    # Take more steps
    for _ in range(10):
        actions = np.array([1, 1])
        obs, _, _, _, _ = env.step(actions)
    
    print(f"State after 10 more steps: {obs}")
    
    # Restore to previous state
    env.restore_all(snapshots)
    print("Restored to snapshot")
    
    # Verify state
    actions = np.array([0, 0])  # No-op action
    obs, _, _, _, _ = env.step(actions)
    print(f"State after restore: {obs}")
    
    env.close()


def single_env_demo():
    """Demonstrate single environment wrapper."""
    print("\n=== Single Environment Demo ===\n")
    
    # Load component bytes
    component_bytes = wasmrl.load_component("./envs/counter_env.wasm")
    print(f"Loaded component: {len(component_bytes)} bytes")
    
    # Create single environment
    config = wasmrl.EnvConfig(max_memory_mb=32, fuel_per_step=500_000)
    env = wasmrl.WasmEnv(component_bytes, config)
    print(f"Created: {env}")
    
    # Reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Env spec: {env.spec()}")
    
    # Run episode
    done = False
    step = 0
    while not done and step < 50:
        action = np.array([1])  # Increment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    
    print(f"Episode finished after {step} steps")
    print(f"Final reward: {env.episode_reward}")
    print(f"Render: {env.render()}")
    
    env.close()


if __name__ == "__main__":
    basic_usage()
    snapshot_demo()
    single_env_demo()
    print("\n✅ All demos completed successfully!")
