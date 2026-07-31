# WasmRL WIT Interface Specification

## Version

**Package**: `wasmrl:env@0.1.0`  
**Status**: Frozen ABI (v0.1.x backward compatible)

## Overview

The WasmRL WIT interface defines the contract for reinforcement learning environments
running as WebAssembly components. This specification covers:

- Core environment operations (init, reset, step, close)
- Batch operations for vectorized execution
- Snapshot/restore for fast reset and replay

## Tensor Encoding

All observations and actions use the `tensor` record type:

```wit
record tensor {
    dtype: dtype,      // Element data type
    shape: list<u32>,  // Dimensions (e.g., [84, 84, 4])
    data: list<u8>,    // Raw bytes in row-major order
}
```

### Data Types

| DType | Size | Description |
|-------|------|-------------|
| `float32` | 4 bytes | 32-bit IEEE 754 float |
| `float64` | 8 bytes | 64-bit IEEE 754 float |
| `int32` | 4 bytes | 32-bit signed integer |
| `int64` | 8 bytes | 64-bit signed integer |
| `uint8` | 1 byte | 8-bit unsigned integer |
| `boolean` | 1 byte | 0 = false, 1 = true |

### Byte Order

All multi-byte values use **little-endian** byte order for cross-platform compatibility.

### Shape Convention

- Shapes are stored as `list<u32>` (e.g., `[84, 84, 4]`)
- Total elements = product of all dimensions
- Expected bytes = elements × dtype_size
- Empty tensor: shape = `[]`, data = `[]`

## Core Interface: `environment`

### `init(config) -> result<env-handle, string>`

Initialize an environment instance with the given configuration.

**Parameters:**
- `config: env-config` - Configuration containing JSON string

**Returns:**
- `ok(env-handle)` - Handle for subsequent operations
- `err(string)` - Error message if initialization fails

**Semantics:**
- Must be called before any other operation
- Configuration is environment-specific JSON
- Multiple instances can be created with different configs

### `reset(handle, seed) -> result<tensor, string>`

Reset the environment to initial state.

**Parameters:**
- `handle: env-handle` - Environment instance handle
- `seed: u64` - Random seed for reproducibility

**Returns:**
- `ok(tensor)` - Initial observation
- `err(string)` - Error message if reset fails

**Semantics:**
- Same seed must produce identical initial states
- Resets episode step counter
- Previous episode statistics are cleared

### `step(handle, action) -> result<step-result, string>`

Execute one environment step with the given action.

**Parameters:**
- `handle: env-handle` - Environment instance handle
- `action: tensor` - Action to execute

**Returns:**
- `ok(step-result)` - Step result with observation, reward, done flags
- `err(string)` - Error message if step fails

**Semantics:**
- Action shape must match action space
- `terminated=true` means natural episode end (goal/failure)
- `truncated=true` means artificial end (time limit)
- After `terminated` or `truncated`, `reset` must be called

### `close(handle) -> result<_, string>`

Close and cleanup an environment instance.

**Parameters:**
- `handle: env-handle` - Environment instance handle

**Returns:**
- `ok(_)` - Success
- `err(string)` - Error message if close fails

**Semantics:**
- Releases all resources associated with the instance
- Handle becomes invalid after close
- Calling operations on closed handle returns error

## Batch Interface: `batch`

Batch operations are **optional** but recommended for high-throughput training.

### `reset-batch(handles, seeds) -> result<list<tensor>, string>`

Reset multiple environments in batch.

**Parameters:**
- `handles: list<env-handle>` - Environment handles
- `seeds: list<u64>` - Seeds for each environment

**Requirements:**
- `handles.len() == seeds.len()` (error otherwise)

**Returns:**
- `ok(list<tensor>)` - Initial observations, one per environment
- `err(string)` - Error message (batch fails atomically)

### `step-batch(handles, actions) -> result<batch-step-result, string>`

Step multiple environments in batch.

**Parameters:**
- `handles: list<env-handle>` - Environment handles
- `actions: list<tensor>` - Actions for each environment

**Requirements:**
- `handles.len() == actions.len()` (error otherwise)

**Returns:**
- `ok(batch-step-result)` - Batched results
- `err(string)` - Error message (batch fails atomically)

**Error Behavior:**
- If any environment fails, the entire batch fails
- Partial results are not returned
- Environments are left in undefined state after batch failure

## Snapshot Interface: `snapshot`

Snapshot operations are **optional** but recommended for reset-heavy workloads.

### `snapshot(handle) -> result<snapshot-data, string>`

Capture environment state as a snapshot.

**Parameters:**
- `handle: env-handle` - Environment instance handle

**Returns:**
- `ok(snapshot-data)` - Serialized state with version
- `err(string)` - Error message if snapshot fails

**Semantics:**
- Captures complete environment state
- Snapshot is opaque binary data
- Version field enables forward compatibility

### `restore(handle, snapshot) -> result<_, string>`

Restore environment to a previously captured state.

**Parameters:**
- `handle: env-handle` - Environment instance handle
- `snapshot: snapshot-data` - Previously captured snapshot

**Returns:**
- `ok(_)` - Success
- `err(string)` - Error message if restore fails

**Requirements:**
- Snapshot must be from same environment type
- Version must be compatible

**Semantics:**
- After restore, environment behaves as if at snapshot point
- Subsequent operations produce identical results (determinism)

## Determinism Requirements

For reproducibility, environments MUST satisfy:

1. **Seed Determinism**: `reset(seed=X)` always produces identical initial state
2. **Action Determinism**: Same action sequence from same initial state produces same trajectory
3. **Snapshot Determinism**: `restore(snapshot)` then replay produces identical trajectory

## Error Handling

All functions return `result<T, string>` where the error is a human-readable message.

Error conventions:
- Include context (e.g., "step failed: invalid action shape [3] expected [4]")
- Do not include stack traces (security)
- Use consistent error prefixes (init/reset/step/close/snapshot/restore)

## Implementation Notes

### For Environment Authors

1. Use `wasmrl-sdk-rust` for tensor encoding/decoding
2. Use `DeterministicRng` for reproducible randomness
3. Use `SnapshotHelper` for state serialization
4. Test determinism with trajectory hash verification

### For Runtime Implementers

1. Validate tensor shapes before passing to environments
2. Enforce resource budgets (fuel, memory, time)
3. Recycle instances after traps
4. Collect metrics per-step (timing, memory)

## Compatibility

- **v0.1.x**: Backward compatible within minor version
- **v0.2.0**: May introduce breaking changes
- **Snapshot versions**: Checked at restore time

## Example Usage

```rust
// Initialize environment
let config = EnvConfig::new(r#"{"max_steps": 1000}"#);
let handle = env.init(&config)?;

// Run episode
let obs = env.reset(handle, seed)?;
loop {
    let action = policy.act(&obs);
    let result = env.step(handle, &action)?;
    if result.done() {
        break;
    }
}

// Cleanup
env.close(handle)?;
```
