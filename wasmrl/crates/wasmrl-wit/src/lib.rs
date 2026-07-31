// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL WIT interface definitions.
//!
//! This crate defines the WebAssembly Interface Types (WIT) for WasmRL environments.
//! It provides the core contract that all environment components must implement.
//!
//! # WIT Interface Version
//!
//! The current ABI version is `wasmrl:env@0.1.0`. This version is frozen and
//! will maintain backward compatibility within the 0.1.x series.
//!
//! # Tensor Encoding
//!
//! Tensors are encoded as: `dtype + shape + bytes` where:
//! - `dtype`: Element data type (float32, float64, int32, int64, uint8, bool)
//! - `shape`: Dimensions as `list<u32>` (e.g., `[84, 84, 4]` for stacked frames)
//! - `data`: Raw bytes in row-major (C) order
//!
//! # Core Functions
//!
//! - `init(config) -> env`: Initialize environment with JSON configuration
//! - `reset(seed) -> obs`: Reset to initial state, return observation
//! - `step(action) -> step_out`: Execute action, return (obs, reward, done, info)
//!
//! # Batch Functions (Optional)
//!
//! - `reset_batch(seeds[]) -> obs[]`: Reset multiple environments
//! - `step_batch(actions[]) -> step_out[]`: Step multiple environments
//!
//! # Snapshot Functions (Optional)
//!
//! - `snapshot() -> bytes`: Capture environment state
//! - `restore(bytes)`: Restore to captured state

#![warn(missing_docs)]

use std::fmt;

use serde::{Deserialize, Serialize};

/// WIT interface version for tracking ABI compatibility.
pub const WIT_VERSION: &str = "0.1.0";

/// WIT package identifier.
pub const WIT_PACKAGE: &str = "wasmrl:env@0.1.0";

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DType {
    /// 32-bit floating point (4 bytes per element)
    Float32 = 0,
    /// 64-bit floating point (8 bytes per element)
    Float64 = 1,
    /// 32-bit signed integer (4 bytes per element)
    Int32 = 2,
    /// 64-bit signed integer (8 bytes per element)
    Int64 = 3,
    /// 8-bit unsigned integer (1 byte per element)
    Uint8 = 4,
    /// Boolean (1 byte per element, 0=false, 1=true)
    Boolean = 5,
}

impl DType {
    /// Returns the size in bytes of one element of this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Uint8 => 1,
            DType::Boolean => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::Uint8 => write!(f, "uint8"),
            DType::Boolean => write!(f, "bool"),
        }
    }
}

/// Tensor representation for observations and actions.
///
/// # Encoding Format
///
/// Tensors are encoded as:
/// - `dtype`: Element data type
/// - `shape`: Dimensions as `Vec<u32>`
/// - `data`: Raw bytes in row-major (C) order
///
/// # Example
///
/// ```
/// use wasmrl_wit::{Tensor, DType};
///
/// // Create a 2x3 float32 tensor
/// let tensor = Tensor::new(DType::Float32, vec![2, 3], vec![0u8; 24]);
/// assert_eq!(tensor.num_elements(), 6);
/// assert_eq!(tensor.byte_size(), 24);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    /// Data type of tensor elements.
    pub dtype: DType,
    /// Shape of the tensor (dimensions).
    pub shape: Vec<u32>,
    /// Raw bytes in row-major (C) order.
    pub data: Vec<u8>,
}

impl Tensor {
    /// Create a new tensor with the given dtype, shape, and data.
    pub fn new(dtype: DType, shape: Vec<u32>, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }

    /// Create an empty tensor with the given dtype and shape.
    pub fn zeros(dtype: DType, shape: Vec<u32>) -> Self {
        let num_elements: usize = shape.iter().map(|&d| d as usize).product();
        let byte_size = num_elements * dtype.element_size();
        Self {
            dtype,
            shape,
            data: vec![0u8; byte_size],
        }
    }

    /// Calculate total number of elements in tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Calculate expected byte size based on dtype and shape.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }

    /// Validate that data length matches expected byte size.
    pub fn is_valid(&self) -> bool {
        self.data.len() == self.byte_size()
    }

    /// Get the number of dimensions (rank) of this tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Result of a single environment step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Next observation after taking the action.
    pub observation: Tensor,
    /// Reward received for the transition.
    pub reward: f64,
    /// Whether the episode has terminated (goal reached or failure).
    pub terminated: bool,
    /// Whether the episode was truncated (time limit, etc.).
    pub truncated: bool,
    /// Optional info dictionary as JSON string.
    pub info: Option<String>,
}

impl StepResult {
    /// Create a new step result.
    pub fn new(observation: Tensor, reward: f64, terminated: bool, truncated: bool) -> Self {
        Self {
            observation,
            reward,
            terminated,
            truncated,
            info: None,
        }
    }

    /// Check if episode is done (terminated or truncated).
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Environment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    /// Configuration as JSON string for flexibility.
    pub config_json: String,
}

impl EnvConfig {
    /// Create a new environment configuration from JSON string.
    pub fn new(config_json: impl Into<String>) -> Self {
        Self {
            config_json: config_json.into(),
        }
    }

    /// Create an empty configuration.
    pub fn empty() -> Self {
        Self {
            config_json: "{}".to_string(),
        }
    }
}

/// Environment handle returned after initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnvHandle {
    /// Unique identifier for this environment instance.
    pub id: u64,
}

impl EnvHandle {
    /// Create a new environment handle with the given ID.
    pub fn new(id: u64) -> Self {
        Self { id }
    }
}

impl fmt::Display for EnvHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Env({})", self.id)
    }
}

/// Snapshot data for environment state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotData {
    /// Version tag for compatibility checking.
    pub version: u32,
    /// Serialized environment state.
    pub data: Vec<u8>,
}

impl SnapshotData {
    /// Current snapshot format version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new snapshot with current version.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            data,
        }
    }

    /// Check if this snapshot version is compatible.
    pub fn is_compatible(&self) -> bool {
        self.version == Self::CURRENT_VERSION
    }
}

/// Batched step result for vectorized environments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStepResult {
    /// Array of observations (one per environment).
    pub observations: Vec<Tensor>,
    /// Array of rewards (one per environment).
    pub rewards: Vec<f64>,
    /// Array of terminated flags.
    pub terminated: Vec<bool>,
    /// Array of truncated flags.
    pub truncated: Vec<bool>,
    /// Array of optional info strings.
    pub infos: Vec<Option<String>>,
}

impl BatchStepResult {
    /// Create a new batch step result with the given size.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            observations: Vec::with_capacity(n),
            rewards: Vec::with_capacity(n),
            terminated: Vec::with_capacity(n),
            truncated: Vec::with_capacity(n),
            infos: Vec::with_capacity(n),
        }
    }

    /// Get the batch size (number of environments).
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }

    /// Validate that all arrays have the same length.
    pub fn is_valid(&self) -> bool {
        let n = self.rewards.len();
        self.observations.len() == n
            && self.terminated.len() == n
            && self.truncated.len() == n
            && self.infos.len() == n
    }
}

/// Environment interface marker for trait implementations.
///
/// This trait defines the core contract that all WasmRL environments must implement.
/// The functions correspond directly to the WIT interface definitions.
pub trait WasmRLEnvironment {
    /// Initialize environment with given configuration.
    fn init(&mut self, config: &EnvConfig) -> anyhow::Result<EnvHandle>;

    /// Reset environment and return initial observation.
    fn reset(&mut self, handle: EnvHandle, seed: u64) -> anyhow::Result<Tensor>;

    /// Execute one step with given action.
    fn step(&mut self, handle: EnvHandle, action: &Tensor) -> anyhow::Result<StepResult>;

    /// Close and cleanup the environment instance.
    fn close(&mut self, handle: EnvHandle) -> anyhow::Result<()>;
}

/// Optional batch operations trait.
pub trait WasmRLBatch: WasmRLEnvironment {
    /// Reset multiple environments in batch.
    fn reset_batch(&mut self, handles: &[EnvHandle], seeds: &[u64]) -> anyhow::Result<Vec<Tensor>>;

    /// Step multiple environments in batch.
    fn step_batch(
        &mut self,
        handles: &[EnvHandle],
        actions: &[Tensor],
    ) -> anyhow::Result<BatchStepResult>;
}

/// Optional snapshot operations trait.
pub trait WasmRLSnapshot: WasmRLEnvironment {
    /// Capture current environment state as a snapshot.
    fn snapshot(&self, handle: EnvHandle) -> anyhow::Result<SnapshotData>;

    /// Restore environment to a previously captured state.
    fn restore(&mut self, handle: EnvHandle, snapshot: &SnapshotData) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wit_version() {
        assert_eq!(WIT_VERSION, "0.1.0");
    }

    #[test]
    fn test_wit_package() {
        assert_eq!(WIT_PACKAGE, "wasmrl:env@0.1.0");
    }

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(DType::Float32.element_size(), 4);
        assert_eq!(DType::Float64.element_size(), 8);
        assert_eq!(DType::Int32.element_size(), 4);
        assert_eq!(DType::Int64.element_size(), 8);
        assert_eq!(DType::Uint8.element_size(), 1);
        assert_eq!(DType::Boolean.element_size(), 1);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::Float32), "float32");
        assert_eq!(format!("{}", DType::Float64), "float64");
        assert_eq!(format!("{}", DType::Uint8), "uint8");
    }

    #[test]
    fn test_tensor_new() {
        let tensor = Tensor::new(DType::Float32, vec![2, 3], vec![0u8; 24]);
        assert_eq!(tensor.dtype, DType::Float32);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.byte_size(), 24);
        assert!(tensor.is_valid());
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(DType::Float64, vec![4, 4]);
        assert_eq!(tensor.num_elements(), 16);
        assert_eq!(tensor.byte_size(), 128);
        assert_eq!(tensor.data.len(), 128);
        assert!(tensor.is_valid());
    }

    #[test]
    fn test_tensor_ndim() {
        let tensor1d = Tensor::zeros(DType::Uint8, vec![100]);
        let tensor3d = Tensor::zeros(DType::Float32, vec![84, 84, 4]);
        assert_eq!(tensor1d.ndim(), 1);
        assert_eq!(tensor3d.ndim(), 3);
    }

    #[test]
    fn test_tensor_invalid() {
        let tensor = Tensor::new(DType::Float32, vec![2, 3], vec![0u8; 10]); // Wrong size
        assert!(!tensor.is_valid());
    }

    #[test]
    fn test_step_result_done() {
        let obs = Tensor::zeros(DType::Float32, vec![4]);

        let result1 = StepResult::new(obs.clone(), 1.0, true, false);
        assert!(result1.done());

        let result2 = StepResult::new(obs.clone(), 1.0, false, true);
        assert!(result2.done());

        let result3 = StepResult::new(obs, 1.0, false, false);
        assert!(!result3.done());
    }

    #[test]
    fn test_env_config() {
        let config = EnvConfig::new(r#"{"max_steps": 1000}"#);
        assert!(config.config_json.contains("max_steps"));

        let empty = EnvConfig::empty();
        assert_eq!(empty.config_json, "{}");
    }

    #[test]
    fn test_env_handle() {
        let handle = EnvHandle::new(42);
        assert_eq!(handle.id, 42);
        assert_eq!(format!("{}", handle), "Env(42)");
    }

    #[test]
    fn test_snapshot_data() {
        let snapshot = SnapshotData::new(vec![1, 2, 3, 4]);
        assert_eq!(snapshot.version, SnapshotData::CURRENT_VERSION);
        assert!(snapshot.is_compatible());
        assert_eq!(snapshot.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_batch_step_result() {
        let mut result = BatchStepResult::with_capacity(3);
        result
            .observations
            .push(Tensor::zeros(DType::Float32, vec![4]));
        result.rewards.push(1.0);
        result.terminated.push(false);
        result.truncated.push(false);
        result.infos.push(None);

        assert_eq!(result.len(), 1);
        assert!(!result.is_empty());
        assert!(result.is_valid());
    }

    #[test]
    fn test_batch_step_result_invalid() {
        let mut result = BatchStepResult::with_capacity(2);
        result.rewards.push(1.0);
        result.rewards.push(2.0);
        // Missing other arrays
        assert!(!result.is_valid());
    }
}
