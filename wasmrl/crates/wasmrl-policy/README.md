# WasmRL Policy

Policy schema, parser, and enforcement for WasmRL environments.

## Overview

This crate provides a comprehensive policy system for controlling WebAssembly
environment execution with:

- **Resource Budgets**: Memory limits, fuel/instruction budgets, timeouts
- **Capability Control**: WASI filesystem permissions, network access
- **Telemetry**: Tracking of budget overruns, denied capabilities, trap reasons

## Quick Start

### Creating Policies

```rust
use wasmrl_policy::{Policy, PolicyBuilder};

// Using the builder
let policy = PolicyBuilder::new()
    .max_memory_mb(64)
    .fuel_per_step(1_000_000)
    .timeout_step_ms(100)
    .allow_filesystem_read(&["/data"])
    .deny_network()
    .build();

// From TOML
let policy = Policy::from_toml(r#"
    [memory]
    max_mb = 64

    [fuel]
    per_step = 1000000

    [timeout]
    step_ms = 100
"#)?;
```

### Enforcing Budgets

```rust
use wasmrl_policy::{BudgetEnforcer, Policy};

let policy = Policy::default();
let enforcer = BudgetEnforcer::new(policy);

// Check memory usage
let result = enforcer.check_memory(current_bytes);
if result.is_denied() {
    // Handle memory limit exceeded
}

// Check capabilities
if enforcer.check_network().is_denied() {
    // Network access denied
}

// Track fuel consumption
enforcer.record_fuel(consumed);
let result = enforcer.check_fuel("step");
```

### Collecting Telemetry

```rust
use wasmrl_policy::TelemetryCollector;
use std::time::Duration;

let collector = TelemetryCollector::new();

// Record operations
collector.record_step(Duration::from_micros(100), fuel_consumed);
collector.record_reset(Duration::from_millis(10), fuel_consumed);

// Record violations
collector.record_budget_overrun(BudgetType::Fuel, limit, actual, "step");
collector.record_capability_denial("network", "Denied by policy", None);
collector.record_trap("unreachable", "trap message", Some("function_name"));

// Get statistics
let snapshot = collector.snapshot();
println!("Steps: {}", snapshot.steps_total);
println!("Overrun rate: {:.2}%", snapshot.overrun_rate() * 100.0);
println!("Trap rate: {:.2}%", snapshot.trap_rate() * 100.0);
```

## Policy Schema

### TOML Format

```toml
# Memory limits
[memory]
max_mb = 64              # Maximum memory in megabytes
initial_mb = 16          # Initial memory allocation
enabled = true           # Enable memory limiting

# Fuel/instruction budgets
[fuel]
per_step = 1000000       # Fuel per step call
per_reset = 5000000      # Fuel per reset call
per_batch = 10000000     # Fuel per batch call
per_init = 10000000      # Fuel for initialization
per_snapshot = 2000000   # Fuel for snapshot
per_restore = 2000000    # Fuel for restore
enabled = true           # Enable fuel metering

# Timeouts
[timeout]
step_ms = 100            # Timeout for step (milliseconds)
reset_ms = 500           # Timeout for reset
batch_ms = 1000          # Timeout for batch operations
init_ms = 5000           # Timeout for initialization
snapshot_ms = 200        # Timeout for snapshot/restore
enabled = true           # Enable timeouts

# WASI capabilities
[wasi]
filesystem_read = ["/data", "/models"]  # Read-only paths
filesystem_write = ["/tmp"]             # Writable paths
network = false                         # Network access (deny by default)
env_vars = ["HOME", "PATH"]             # Allowed env vars
clock = true                            # Allow clock/time
random = true                           # Allow random generation
inherit_stdio = false                   # Inherit stdio from host

# Additional capabilities
[capabilities]
threading = false        # Allow threading/atomics
simd = true              # Allow SIMD instructions
component_model = true   # Allow component model
multi_memory = false     # Allow multi-memory
bulk_memory = true       # Allow bulk memory ops
```

### JSON Format

```json
{
  "memory": {
    "max_mb": 64,
    "initial_mb": 16,
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
}
```

## Module Structure

- **schema**: Policy data structures and parsing
- **enforcement**: Budget and capability enforcement
- **telemetry**: Event tracking and statistics
- **error**: Error types and handling

## Security Design

The policy system follows a **deny-by-default** principle:

1. **Network**: Disabled unless explicitly allowed
2. **Filesystem**: No access unless paths are whitelisted
3. **Memory**: Capped at configured maximum
4. **CPU**: Limited by fuel budget per operation
5. **Time**: Bounded by timeouts

## License

MIT License - see LICENSE file in the repository root.
