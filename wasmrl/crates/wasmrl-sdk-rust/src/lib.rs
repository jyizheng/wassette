// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! WasmRL Rust SDK for building environment components.
//!
//! This SDK provides utilities for Rust environment authors including:
//! - Deterministic PRNG for reproducible environments
//! - Tensor encoding/decoding utilities
//! - Error handling conventions
//! - Snapshot/restore helpers
//!
//! # Example: Creating a Simple Environment
//!
//! ```ignore
//! use wasmrl_sdk_rust::{DeterministicRng, TensorEncoder};
//!
//! struct MyEnv {
//!     rng: DeterministicRng,
//!     state: Vec<f32>,
//! }
//!
//! impl MyEnv {
//!     fn reset(&mut self, seed: u64) -> Vec<u8> {
//!         self.rng = DeterministicRng::new(seed);
//!         self.state = vec![0.0; 4];
//!         TensorEncoder::encode_f32(&self.state)
//!     }
//! }
//! ```

#![warn(missing_docs)]

/// A deterministic pseudo-random number generator for environment use.
///
/// Uses the PCG (Permuted Congruential Generator) algorithm which provides:
/// - Deterministic sequences from a given seed
/// - Cross-platform reproducibility
/// - Good statistical properties
///
/// # Example
///
/// ```
/// use wasmrl_sdk_rust::DeterministicRng;
///
/// let mut rng1 = DeterministicRng::new(42);
/// let mut rng2 = DeterministicRng::new(42);
///
/// // Same seed produces same sequence
/// assert_eq!(rng1.next(), rng2.next());
/// ```
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
    inc: u64,
}

impl DeterministicRng {
    /// PCG multiplier constant.
    const MULTIPLIER: u64 = 6364136223846793005;

    /// Create a new PRNG with the given seed.
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        // Warm up the generator
        rng.next();
        rng.state = rng.state.wrapping_add(seed);
        rng.next();
        rng
    }

    /// Generate the next random u64.
    pub fn next(&mut self) -> u64 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);

        // PCG output function (XSH-RR)
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        ((xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))) as u64
    }

    /// Generate the next random u64.
    ///
    /// This is an explicit alias for [`Self::next`] so environment code can use
    /// integer-specific naming without depending on implementation details.
    pub fn next_u64(&mut self) -> u64 {
        self.next()
    }

    /// Return the internal RNG state for deterministic snapshots.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Generate a random u64 in range [0, max).
    pub fn next_in_range(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.next() % max
    }

    /// Generate a random f64 in range [0.0, 1.0).
    pub fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }

    /// Generate a random f32 in range [0.0, 1.0).
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Generate a random i32 in range [low, high).
    pub fn next_i32_range(&mut self, low: i32, high: i32) -> i32 {
        if low >= high {
            return low;
        }
        let range = (high - low) as u64;
        low + (self.next_in_range(range) as i32)
    }

    /// Shuffle a slice in place using Fisher-Yates algorithm.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = self.next_in_range((i + 1) as u64) as usize;
            slice.swap(i, j);
        }
    }
}

/// Tensor encoder for converting Rust types to raw bytes.
///
/// All encoding uses little-endian byte order for cross-platform compatibility.
#[derive(Debug)]
pub struct TensorEncoder;

impl TensorEncoder {
    /// Encode a slice of f32 values to bytes.
    pub fn encode_f32(data: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Encode a slice of f64 values to bytes.
    pub fn encode_f64(data: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Encode a slice of i32 values to bytes.
    pub fn encode_i32(data: &[i32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Encode a slice of i64 values to bytes.
    pub fn encode_i64(data: &[i64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Encode a slice of u8 values (no conversion needed).
    pub fn encode_u8(data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    /// Encode a slice of bool values to bytes (0 = false, 1 = true).
    pub fn encode_bool(data: &[bool]) -> Vec<u8> {
        data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect()
    }
}

/// Tensor decoder for converting raw bytes to Rust types.
///
/// All decoding uses little-endian byte order for cross-platform compatibility.
#[derive(Debug)]
pub struct TensorDecoder;

impl TensorDecoder {
    /// Decode bytes to a Vec of f32 values.
    pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
        if bytes.len() % 4 != 0 {
            return Err(TensorError::InvalidLength {
                expected: "multiple of 4",
                actual: bytes.len(),
            });
        }
        let mut result = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            result.push(f32::from_le_bytes(arr));
        }
        Ok(result)
    }

    /// Decode bytes to a Vec of f64 values.
    pub fn decode_f64(bytes: &[u8]) -> Result<Vec<f64>, TensorError> {
        if bytes.len() % 8 != 0 {
            return Err(TensorError::InvalidLength {
                expected: "multiple of 8",
                actual: bytes.len(),
            });
        }
        let mut result = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            result.push(f64::from_le_bytes(arr));
        }
        Ok(result)
    }

    /// Decode bytes to a Vec of i32 values.
    pub fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>, TensorError> {
        if bytes.len() % 4 != 0 {
            return Err(TensorError::InvalidLength {
                expected: "multiple of 4",
                actual: bytes.len(),
            });
        }
        let mut result = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            result.push(i32::from_le_bytes(arr));
        }
        Ok(result)
    }

    /// Decode bytes to a Vec of i64 values.
    pub fn decode_i64(bytes: &[u8]) -> Result<Vec<i64>, TensorError> {
        if bytes.len() % 8 != 0 {
            return Err(TensorError::InvalidLength {
                expected: "multiple of 8",
                actual: bytes.len(),
            });
        }
        let mut result = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            result.push(i64::from_le_bytes(arr));
        }
        Ok(result)
    }

    /// Decode bytes to a Vec of bool values.
    pub fn decode_bool(bytes: &[u8]) -> Vec<bool> {
        bytes.iter().map(|&b| b != 0).collect()
    }
}

/// Error type for tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Invalid byte length for the expected type.
    InvalidLength {
        /// Expected length description.
        expected: &'static str,
        /// Actual length received.
        actual: usize,
    },
    /// Shape does not match data length.
    ShapeMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InvalidLength { expected, actual } => {
                write!(f, "invalid length: expected {}, got {}", expected, actual)
            }
            TensorError::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "shape mismatch: expected {} elements, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Snapshot helper for serializing environment state.
///
/// Provides versioned serialization for forward compatibility.
#[derive(Debug)]
pub struct SnapshotHelper;

impl SnapshotHelper {
    /// Current snapshot format version.
    pub const VERSION: u32 = 1;

    /// Create a snapshot header with version information.
    pub fn create_header() -> Vec<u8> {
        let mut header = Vec::with_capacity(8);
        header.extend_from_slice(&Self::VERSION.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes()); // Reserved for future use
        header
    }

    /// Parse snapshot header and return version.
    pub fn parse_header(data: &[u8]) -> Result<u32, &'static str> {
        if data.len() < 8 {
            return Err("snapshot data too short");
        }
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        Ok(version)
    }

    /// Serialize state with header.
    pub fn serialize<T: serde::Serialize>(state: &T) -> Result<Vec<u8>, String> {
        let mut data = Self::create_header();
        let state_bytes =
            serde_json::to_vec(state).map_err(|e| format!("serialization failed: {}", e))?;
        data.extend_from_slice(&(state_bytes.len() as u32).to_le_bytes());
        data.extend(state_bytes);
        Ok(data)
    }

    /// Deserialize state from snapshot data.
    pub fn deserialize<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, String> {
        let version = Self::parse_header(data).map_err(|e| e.to_string())?;
        if version != Self::VERSION {
            return Err(format!(
                "incompatible snapshot version: {} (expected {})",
                version,
                Self::VERSION
            ));
        }
        if data.len() < 12 {
            return Err("snapshot data too short".to_string());
        }
        let state_len = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if data.len() < 12 + state_len {
            return Err("snapshot data truncated".to_string());
        }
        serde_json::from_slice(&data[12..12 + state_len])
            .map_err(|e| format!("deserialization failed: {}", e))
    }
}

/// Tensor metadata for encoding/decoding operations.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Data type of the tensor.
    pub dtype: String,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
}

impl TensorMetadata {
    /// Create new tensor metadata.
    pub fn new(dtype: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            dtype: dtype.into(),
            shape,
        }
    }

    /// Calculate total number of elements in tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng_seed() {
        let rng1 = DeterministicRng::new(42);
        let rng2 = DeterministicRng::new(42);
        assert_eq!(rng1.state, rng2.state);
        assert_eq!(rng1.inc, rng2.inc);
    }

    #[test]
    fn test_deterministic_rng_sequence() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_deterministic_rng_explicit_u64_and_state() {
        let mut rng = DeterministicRng::new(42);
        let initial_state = rng.state();
        let mut same_rng = rng.clone();

        assert_eq!(rng.next_u64(), same_rng.next());
        assert_ne!(rng.state(), initial_state);
    }

    #[test]
    fn test_deterministic_rng_range() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let val = rng.next_in_range(10);
            assert!(val < 10);
        }
    }

    #[test]
    fn test_deterministic_rng_f64() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let val = rng.next_f64();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_deterministic_rng_i32_range() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let val = rng.next_i32_range(-10, 10);
            assert!((-10..10).contains(&val));
        }
    }

    #[test]
    fn test_deterministic_rng_shuffle() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        let mut arr1 = vec![1, 2, 3, 4, 5];
        let mut arr2 = vec![1, 2, 3, 4, 5];

        rng1.shuffle(&mut arr1);
        rng2.shuffle(&mut arr2);

        assert_eq!(arr1, arr2);
    }

    #[test]
    fn test_tensor_encoder_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let bytes = TensorEncoder::encode_f32(&data);
        assert_eq!(bytes.len(), 12);

        let decoded = TensorDecoder::decode_f32(&bytes).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_tensor_encoder_f64() {
        let data = vec![1.0f64, 2.0, 3.0];
        let bytes = TensorEncoder::encode_f64(&data);
        assert_eq!(bytes.len(), 24);

        let decoded = TensorDecoder::decode_f64(&bytes).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_tensor_encoder_i32() {
        let data = vec![1i32, -2, 3];
        let bytes = TensorEncoder::encode_i32(&data);
        assert_eq!(bytes.len(), 12);

        let decoded = TensorDecoder::decode_i32(&bytes).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_tensor_encoder_bool() {
        let data = vec![true, false, true, true];
        let bytes = TensorEncoder::encode_bool(&data);
        assert_eq!(bytes, vec![1, 0, 1, 1]);

        let decoded = TensorDecoder::decode_bool(&bytes);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_tensor_decoder_invalid_length() {
        let bytes = vec![1, 2, 3]; // Not multiple of 4
        let result = TensorDecoder::decode_f32(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_metadata_new() {
        let meta = TensorMetadata::new("float32", vec![4, 3, 3]);
        assert_eq!(meta.dtype, "float32");
        assert_eq!(meta.shape, vec![4, 3, 3]);
    }

    #[test]
    fn test_tensor_metadata_num_elements() {
        let meta = TensorMetadata::new("float32", vec![4, 3, 3]);
        assert_eq!(meta.num_elements(), 36);
    }

    #[test]
    fn test_snapshot_helper_roundtrip() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct TestState {
            value: i32,
            data: Vec<f32>,
        }

        let state = TestState {
            value: 42,
            data: vec![1.0, 2.0, 3.0],
        };

        let serialized = SnapshotHelper::serialize(&state).unwrap();
        let deserialized: TestState = SnapshotHelper::deserialize(&serialized).unwrap();

        assert_eq!(state, deserialized);
    }

    #[test]
    fn test_snapshot_helper_version() {
        let header = SnapshotHelper::create_header();
        let version = SnapshotHelper::parse_header(&header).unwrap();
        assert_eq!(version, SnapshotHelper::VERSION);
    }
}
