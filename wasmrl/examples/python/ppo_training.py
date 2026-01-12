#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PPO Training Example with WasmRL and Stable-Baselines3.

This example demonstrates how to train a PPO agent on a WasmRL
environment using the popular Stable-Baselines3 library.

Prerequisites:
    pip install stable-baselines3
    cd wasmrl/crates/wasmrl-py && maturin develop
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

# Import wasmrl
try:
    import wasmrl_py as wasmrl
except ImportError:
    print("wasmrl_py not installed. Build with: cd wasmrl/crates/wasmrl-py && maturin develop")
    exit(1)

# Try importing stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    HAS_SB3 = False


class WasmRLVecEnvWrapper(VecEnv if HAS_SB3 else object):
    """
    Stable-Baselines3 compatible wrapper for WasmVecEnv.
    
    This wrapper bridges WasmRL's vectorized environment interface
    with SB3's VecEnv interface for seamless integration.
    """
    
    def __init__(
        self,
        component_path: str,
        num_envs: int = 8,
        max_memory_mb: int = 64,
        fuel_per_step: int = 1_000_000,
    ):
        """
        Initialize the wrapper.
        
        Args:
            component_path: Path to the .wasm component file.
            num_envs: Number of parallel environments.
            max_memory_mb: Maximum memory per environment in MB.
            fuel_per_step: Fuel budget per step.
        """
        # Create WasmRL config
        config = wasmrl.EnvConfig(
            num_envs=num_envs,
            max_memory_mb=max_memory_mb,
            fuel_per_step=fuel_per_step,
            auto_reset=True,
        )
        
        # Create the WasmRL vectorized environment
        self.wasmrl_env = wasmrl.WasmVecEnv(component_path, config)
        
        # Get spaces from wasmrl
        obs_space = self.wasmrl_env.single_observation_space
        act_space = self.wasmrl_env.single_action_space
        
        # Convert to gymnasium spaces
        if HAS_SB3:
            import gymnasium as gym
            
            # Create observation space
            if hasattr(obs_space, 'low'):
                # Box space
                observation_space = gym.spaces.Box(
                    low=obs_space.low,
                    high=obs_space.high,
                    shape=tuple(obs_space.shape),
                    dtype=np.float32,
                )
            else:
                # Discrete observation (treated as Box for now)
                observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                )
            
            # Create action space
            if hasattr(act_space, 'n'):
                # Discrete action space
                action_space = gym.spaces.Discrete(act_space.n)
            else:
                # Box action space
                action_space = gym.spaces.Box(
                    low=act_space.low,
                    high=act_space.high,
                    shape=tuple(act_space.shape),
                    dtype=np.float32,
                )
            
            # Initialize VecEnv base class
            super().__init__(num_envs, observation_space, action_space)
        
        self.num_envs = num_envs
        self._obs_space = obs_space
        self._act_space = act_space
    
    def reset(self) -> np.ndarray:
        """Reset all environments."""
        obs, info = self.wasmrl_env.reset()
        return obs
    
    def step_async(self, actions: np.ndarray) -> None:
        """Store actions for later execution."""
        self._actions = actions
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute stored actions and return results."""
        obs, rewards, dones, truncs, info = self.wasmrl_env.step(self._actions)
        
        # SB3 expects combined done flag in older API
        # For newer API, we return both terminated and truncated
        combined_dones = np.logical_or(dones, truncs)
        
        # Convert info to list of dicts for SB3
        infos = []
        for i in range(self.num_envs):
            env_info = {}
            if info.get('final_observation') and info['final_observation'][i] is not None:
                env_info['terminal_observation'] = info['final_observation'][i]
            if 'episode_rewards' in info:
                env_info['episode'] = {
                    'r': info['episode_rewards'][i],
                    'l': info['episode_lengths'][i],
                }
            infos.append(env_info)
        
        return obs, rewards, combined_dones, infos
    
    def close(self) -> None:
        """Close all environments."""
        self.wasmrl_env.close()
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environments (not implemented)."""
        pass
    
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[np.ndarray] = None,
        **method_kwargs
    ) -> Any:
        """Call a method on the environments."""
        if method_name == 'snapshot':
            return self.wasmrl_env.snapshot_all()
        elif method_name == 'restore':
            return self.wasmrl_env.restore_all(*method_args)
        return None
    
    def get_attr(self, attr_name: str, indices: Optional[np.ndarray] = None) -> Any:
        """Get an attribute from the environments."""
        return getattr(self.wasmrl_env, attr_name, None)
    
    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Optional[np.ndarray] = None
    ) -> None:
        """Set an attribute on the environments."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed."""
        if seed is not None:
            self.wasmrl_env.reset(seed=seed)


class RewardLoggerCallback(BaseCallback if HAS_SB3 else object):
    """Callback for logging training progress."""
    
    def __init__(self, verbose: int = 0):
        if HAS_SB3:
            super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode info when available
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                if len(self.episode_rewards) % 100 == 0:
                    print(f"Episodes: {len(self.episode_rewards)}, "
                          f"Avg reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        return True


def train_ppo():
    """Train a PPO agent on a WasmRL environment."""
    if not HAS_SB3:
        print("stable-baselines3 required for PPO training")
        return
    
    print("=== PPO Training with WasmRL ===\n")
    
    # Create wrapped environment
    env = WasmRLVecEnvWrapper(
        component_path="./envs/counter_env.wasm",
        num_envs=8,
        max_memory_mb=64,
        fuel_per_step=1_000_000,
    )
    print(f"Created environment with {env.num_envs} parallel envs")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    print("\nCreated PPO agent")
    
    # Create callback
    callback = RewardLoggerCallback()
    
    # Train
    print("\nStarting training...")
    total_timesteps = 100_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Print results
    print(f"\nTraining completed!")
    print(f"Total episodes: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        print(f"Final avg reward (last 100): {np.mean(callback.episode_rewards[-100:]):.2f}")
    
    # Save model
    model.save("wasmrl_ppo_counter")
    print("Model saved to wasmrl_ppo_counter.zip")
    
    # Evaluate
    print("\nEvaluating...")
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward.sum()
    
    print(f"Evaluation total reward: {total_reward:.2f}")
    
    env.close()


def demo_without_sb3():
    """Demo that works without stable-baselines3."""
    print("=== Manual Training Loop Demo ===\n")
    print("(Install stable-baselines3 for full PPO training)")
    
    # Create environment directly
    config = wasmrl.EnvConfig(
        num_envs=4,
        max_memory_mb=64,
        fuel_per_step=1_000_000,
        auto_reset=True,
    )
    env = wasmrl.WasmVecEnv("./envs/counter_env.wasm", config)
    
    print(f"Created {env.num_envs} parallel environments")
    print(f"Observation space: {env.single_observation_space}")
    print(f"Action space: {env.single_action_space}")
    
    # Simple rollout
    obs, _ = env.reset()
    total_rewards = np.zeros(env.num_envs)
    
    for step in range(500):
        # Random policy
        actions = env.sample_actions()
        obs, rewards, dones, truncs, info = env.step(actions)
        total_rewards += rewards
        
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: Total rewards = {total_rewards}")
    
    print(f"\nFinal total rewards: {total_rewards}")
    env.close()


if __name__ == "__main__":
    if HAS_SB3:
        train_ppo()
    else:
        demo_without_sb3()
    
    print("\n✅ Training example completed!")
