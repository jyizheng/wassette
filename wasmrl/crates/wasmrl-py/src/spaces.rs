// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Gymnasium-compatible space definitions.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Base trait for spaces (not directly exposed to Python).
pub trait Space {
    /// Get the shape of the space.
    fn shape(&self) -> Vec<usize>;
    /// Get the dtype as string.
    fn dtype(&self) -> &str;
    /// Check if a value is within the space.
    fn contains(&self, value: &[f64]) -> bool;
    /// Sample a random value from the space.
    fn sample(&self) -> Vec<f64>;
}

/// Python-exposed abstract space type.
#[pyclass(name = "Space", subclass)]
#[derive(Debug, Clone)]
pub struct PySpace {
    /// Space shape.
    #[pyo3(get)]
    pub shape: Vec<usize>,
    /// Data type.
    #[pyo3(get)]
    pub dtype: String,
}

#[pymethods]
impl PySpace {
    /// Check if value is in space.
    pub fn contains(&self, _value: &Bound<'_, pyo3::types::PyAny>) -> bool {
        // Base implementation - subclasses override
        true
    }

    fn __repr__(&self) -> String {
        format!("Space(shape={:?}, dtype={})", self.shape, self.dtype)
    }
}

/// Box space for continuous values.
#[pyclass(name = "Box", extends = PySpace)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyBox {
    /// Lower bounds.
    #[pyo3(get)]
    pub low: Vec<f64>,
    /// Upper bounds.
    #[pyo3(get)]
    pub high: Vec<f64>,
}

#[pymethods]
impl PyBox {
    /// Create a new Box space.
    #[new]
    #[pyo3(signature = (low, high, shape=None, dtype="float32"))]
    pub fn new(
        low: Vec<f64>,
        high: Vec<f64>,
        shape: Option<Vec<usize>>,
        dtype: &str,
    ) -> (Self, PySpace) {
        let inferred_shape = shape.unwrap_or_else(|| vec![low.len()]);
        (
            PyBox { low, high },
            PySpace {
                shape: inferred_shape,
                dtype: dtype.to_string(),
            },
        )
    }

    /// Create a Box with uniform bounds.
    #[staticmethod]
    #[pyo3(signature = (low, high, shape, dtype="float32"))]
    pub fn uniform(low: f64, high: f64, shape: Vec<usize>, dtype: &str) -> (Self, PySpace) {
        let size: usize = shape.iter().product();
        (
            PyBox {
                low: vec![low; size],
                high: vec![high; size],
            },
            PySpace {
                shape,
                dtype: dtype.to_string(),
            },
        )
    }

    /// Check if value is within bounds.
    pub fn contains(&self, value: Vec<f64>) -> bool {
        if value.len() != self.low.len() {
            return false;
        }
        value
            .iter()
            .zip(self.low.iter().zip(self.high.iter()))
            .all(|(v, (l, h))| v >= l && v <= h)
    }

    /// Sample from the space.
    pub fn sample(&self) -> Vec<f64> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let mut rng_state = seed;
        self.low
            .iter()
            .zip(self.high.iter())
            .map(|(l, h)| {
                // Simple LCG random
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand = (rng_state >> 33) as f64 / (u32::MAX as f64);
                l + rand * (h - l)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Box(low={:.2?}, high={:.2?})",
            &self.low[..self.low.len().min(4)],
            &self.high[..self.high.len().min(4)]
        )
    }
}

/// Discrete space for integer actions.
#[pyclass(name = "Discrete", extends = PySpace)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyDiscrete {
    /// Number of discrete values (0 to n-1).
    #[pyo3(get)]
    pub n: i64,
    /// Starting value.
    #[pyo3(get)]
    pub start: i64,
}

#[pymethods]
impl PyDiscrete {
    /// Create a new Discrete space.
    #[new]
    #[pyo3(signature = (n, start=0))]
    pub fn new(n: i64, start: i64) -> (Self, PySpace) {
        (
            PyDiscrete { n, start },
            PySpace {
                shape: vec![],
                dtype: "int64".to_string(),
            },
        )
    }

    /// Check if value is valid.
    pub fn contains(&self, value: i64) -> bool {
        value >= self.start && value < self.start + self.n
    }

    /// Sample from the space.
    pub fn sample(&self) -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.start + (seed % self.n as u64) as i64
    }

    fn __repr__(&self) -> String {
        if self.start == 0 {
            format!("Discrete({})", self.n)
        } else {
            format!("Discrete({}, start={})", self.n, self.start)
        }
    }
}

/// Multi-discrete space for multiple independent discrete values.
#[pyclass(name = "MultiDiscrete")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyMultiDiscrete {
    /// Number of discrete values for each dimension.
    #[pyo3(get)]
    pub nvec: Vec<i64>,
}

#[pymethods]
impl PyMultiDiscrete {
    /// Create a new MultiDiscrete space.
    #[new]
    pub fn new(nvec: Vec<i64>) -> Self {
        PyMultiDiscrete { nvec }
    }

    /// Get the shape.
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        vec![self.nvec.len()]
    }

    /// Get the dtype.
    #[getter]
    pub fn dtype(&self) -> &str {
        "int64"
    }

    /// Check if value is valid.
    pub fn contains(&self, value: Vec<i64>) -> bool {
        if value.len() != self.nvec.len() {
            return false;
        }
        value
            .iter()
            .zip(self.nvec.iter())
            .all(|(v, n)| *v >= 0 && *v < *n)
    }

    /// Sample from the space.
    pub fn sample(&self) -> Vec<i64> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let mut seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        self.nvec
            .iter()
            .map(|n| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed % *n as u64) as i64
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("MultiDiscrete({:?})", self.nvec)
    }
}

/// Create observation space from shape and bounds.
pub fn make_box_space(shape: Vec<usize>, low: f64, high: f64) -> (PyBox, PySpace) {
    PyBox::uniform(low, high, shape, "float32")
}

/// Create discrete action space.
pub fn make_discrete_space(n: i64) -> (PyDiscrete, PySpace) {
    PyDiscrete::new(n, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_creation() {
        let (box_space, base) = PyBox::new(vec![0.0, 0.0], vec![1.0, 1.0], None, "float32");
        assert_eq!(base.shape, vec![2]);
        assert_eq!(box_space.low.len(), 2);
    }

    #[test]
    fn test_box_uniform() {
        let (box_space, base) = PyBox::uniform(-1.0, 1.0, vec![4], "float32");
        assert_eq!(base.shape, vec![4]);
        assert_eq!(box_space.low, vec![-1.0; 4]);
        assert_eq!(box_space.high, vec![1.0; 4]);
    }

    #[test]
    fn test_box_contains() {
        let (box_space, _) = PyBox::uniform(0.0, 1.0, vec![2], "float32");
        assert!(box_space.contains(vec![0.5, 0.5]));
        assert!(!box_space.contains(vec![1.5, 0.5]));
        assert!(!box_space.contains(vec![0.5])); // Wrong size
    }

    #[test]
    fn test_box_sample() {
        let (box_space, _) = PyBox::uniform(0.0, 1.0, vec![3], "float32");
        let sample = box_space.sample();
        assert_eq!(sample.len(), 3);
        for v in sample {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_discrete_creation() {
        let (disc, base) = PyDiscrete::new(5, 0);
        assert_eq!(disc.n, 5);
        assert_eq!(disc.start, 0);
        assert!(base.shape.is_empty());
    }

    #[test]
    fn test_discrete_contains() {
        let (disc, _) = PyDiscrete::new(5, 0);
        assert!(disc.contains(0));
        assert!(disc.contains(4));
        assert!(!disc.contains(5));
        assert!(!disc.contains(-1));
    }

    #[test]
    fn test_discrete_with_start() {
        let (disc, _) = PyDiscrete::new(5, 10);
        assert!(disc.contains(10));
        assert!(disc.contains(14));
        assert!(!disc.contains(9));
        assert!(!disc.contains(15));
    }

    #[test]
    fn test_discrete_sample() {
        let (disc, _) = PyDiscrete::new(10, 0);
        let sample = disc.sample();
        assert!(sample >= 0 && sample < 10);
    }

    #[test]
    fn test_multi_discrete() {
        let md = PyMultiDiscrete::new(vec![3, 4, 5]);
        assert_eq!(md.shape(), vec![3]);
        assert!(md.contains(vec![0, 0, 0]));
        assert!(md.contains(vec![2, 3, 4]));
        assert!(!md.contains(vec![3, 0, 0])); // First dim out of range
    }

    #[test]
    fn test_multi_discrete_sample() {
        let md = PyMultiDiscrete::new(vec![3, 4, 5]);
        let sample = md.sample();
        assert_eq!(sample.len(), 3);
        assert!(sample[0] >= 0 && sample[0] < 3);
        assert!(sample[1] >= 0 && sample[1] < 4);
        assert!(sample[2] >= 0 && sample[2] < 5);
    }

    #[test]
    fn test_box_repr() {
        let (box_space, _) = PyBox::uniform(0.0, 1.0, vec![2], "float32");
        let repr = box_space.__repr__();
        assert!(repr.contains("Box"));
    }

    #[test]
    fn test_discrete_repr() {
        let (disc, _) = PyDiscrete::new(5, 0);
        assert_eq!(disc.__repr__(), "Discrete(5)");
    }
}
