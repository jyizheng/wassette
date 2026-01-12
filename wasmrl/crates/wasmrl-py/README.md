# WasmRL Python Bindings

Python bindings for WasmRL, enabling WebAssembly RL environment execution
with seamless integration into the Python ML ecosystem.

## Features

- **Gymnasium VecEnv Compatible**: Drop-in replacement for SB3's VecEnv
- **Parallel Execution**: Run thousands of environments concurrently
- **Snapshot/Restore**: Save and restore environment states
- **Policy Enforcement**: Built-in resource budgets and security policies
- **Zero-Copy Tensors**: Efficient numpy array conversions

## Installation

### From Source (Development)

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd wasmrl/crates/wasmrl-py
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/wasmrl_py-*.whl
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20
- Rust toolchain (for building)

## Quick Start

```python
import wasmrl_py as wasmrl
import numpy as np

# Create a vectorized environment
config = wasmrl.EnvConfig(
    num_envs=8,
    max_memory_mb=64,
    fuel_per_step=1_000_000,
    auto_reset=True,
)

env = wasmrl.WasmVecEnv("counter_env.wasm", config)

# Standard Gymnasium interface
obs, info = env.reset()
for _ in range(1000):
    actions = env.sample_actions()
    obs, rewards, terminated, truncated, info = env.step(actions)

env.close()
```

## Integration with Stable-Baselines3

```python
from wasmrl_py import WasmVecEnv, EnvConfig
from stable_baselines3 import PPO

# Create environment (compatible with SB3's VecEnv)
config = EnvConfig(num_envs=8, auto_reset=True)
env = WasmVecEnv("counter_env.wasm", config)

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, _, _, _ = env.step(action)
```

## API Reference

### Classes

#### `EnvConfig`
Configuration for WasmRL environments.

```python
config = EnvConfig(
    num_envs=8,           # Number of parallel environments
    max_memory_mb=64,     # Maximum memory per env in MB
    fuel_per_step=1e6,    # Fuel budget per step
    timeout_step_ms=100,  # Timeout per step in ms
    auto_reset=True,      # Auto-reset on episode end
    seed=None,            # Random seed (optional)
)
```

#### `WasmVecEnv`
Vectorized environment for parallel RL training.

```python
env = WasmVecEnv(component_path, config)

# Reset all environments
obs, info = env.reset(seed=42)

# Step all environments
obs, rewards, terminated, truncated, info = env.step(actions)

# Sample random actions
actions = env.sample_actions()

# Snapshot/restore
snapshots = env.snapshot_all()
env.restore_all(snapshots)

# Close
env.close()
```

#### `WasmEnv`
Single environment wrapper.

```python
component_bytes = wasmrl.load_component("env.wasm")
env = WasmEnv(component_bytes, config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### Spaces

#### `Box`
Continuous observation/action space.

```python
space = env.single_observation_space
print(space.low, space.high, space.shape)
sample = space.sample()
assert space.contains(sample)
```

#### `Discrete`
Discrete action space.

```python
space = env.single_action_space
print(space.n, space.start)
action = space.sample()
```

### Functions

#### `make_vec_env`
Convenience function to create vectorized environments.

```python
env = wasmrl.make_vec_env("env.wasm", num_envs=8, config=config)
```

#### `list_available_envs`
List .wasm files in a directory.

```python
envs = wasmrl.list_available_envs("./envs/")
```

#### `load_component`
Load component bytes from a file.

```python
component_bytes = wasmrl.load_component("env.wasm")
```

## Performance Tips

1. **Batch Size**: Use `num_envs` that is a multiple of your batch size
2. **Memory**: Set `max_memory_mb` appropriately for your environment
3. **Fuel**: Increase `fuel_per_step` for complex step computations
4. **Snapshots**: Use sparingly as they have memory overhead

## Development

### Running Tests

```bash
# Rust tests
cargo test -p wasmrl-py

# Python tests (after maturin develop)
pytest tests/
```

### Building Wheels

```bash
# Build for current platform
maturin build --release

# Build manylinux wheels (for distribution)
maturin build --release --manylinux 2_28
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.
