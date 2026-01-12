// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Tensor conversion between Python/NumPy and WasmRL.

use crate::error::{PyWasmRLError, PyWasmRLResult};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use wasmrl_wit::{DType, Tensor};

/// Python tensor wrapper for conversions.
#[derive(Debug, Clone)]
pub struct PyTensor {
    /// Inner WasmRL tensor.
    pub inner: Tensor,
}

impl PyTensor {
    /// Create from WasmRL Tensor.
    pub fn new(tensor: Tensor) -> Self {
        Self { inner: tensor }
    }

    /// Create from raw parts.
    pub fn from_parts(dtype: DType, shape: Vec<u32>, data: Vec<u8>) -> Self {
        Self {
            inner: Tensor::new(dtype, shape, data),
        }
    }

    /// Convert i32 array to tensor.
    pub fn from_i32_array(data: &[i32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self::from_parts(DType::Int32, vec![data.len() as u32], bytes)
    }

    /// Convert i64 array to tensor.
    pub fn from_i64_array(data: &[i64]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self::from_parts(DType::Int64, vec![data.len() as u32], bytes)
    }

    /// Convert f32 array to tensor.
    pub fn from_f32_array(data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self::from_parts(DType::Float32, vec![data.len() as u32], bytes)
    }

    /// Convert f64 array to tensor.
    pub fn from_f64_array(data: &[f64]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self::from_parts(DType::Float64, vec![data.len() as u32], bytes)
    }

    /// Convert to f32 vec.
    pub fn to_f32_vec(&self) -> PyWasmRLResult<Vec<f32>> {
        if self.inner.dtype != DType::Float32 {
            return Err(PyWasmRLError::TypeError(format!(
                "Expected Float32, got {:?}",
                self.inner.dtype
            )));
        }

        let data: Vec<f32> = self
            .inner
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(data)
    }

    /// Convert to i32 vec.
    pub fn to_i32_vec(&self) -> PyWasmRLResult<Vec<i32>> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyWasmRLError::TypeError(format!(
                "Expected Int32, got {:?}",
                self.inner.dtype
            )));
        }

        let data: Vec<i32> = self
            .inner
            .data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(data)
    }

    /// Convert to i64 vec.
    pub fn to_i64_vec(&self) -> PyWasmRLResult<Vec<i64>> {
        if self.inner.dtype != DType::Int64 {
            return Err(PyWasmRLError::TypeError(format!(
                "Expected Int64, got {:?}",
                self.inner.dtype
            )));
        }

        let data: Vec<i64> = self
            .inner
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        Ok(data)
    }

    /// Get shape as Vec<usize>.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.iter().map(|&x| x as usize).collect()
    }

    /// Get total number of elements.
    pub fn numel(&self) -> usize {
        self.inner.shape.iter().map(|&x| x as usize).product()
    }
}

/// Convert numpy array to action tensor for a single environment.
pub fn numpy_to_action_tensor(
    py: Python<'_>,
    action: &Bound<'_, pyo3::types::PyAny>,
) -> PyResult<Tensor> {
    // Try as scalar int first
    if let Ok(val) = action.extract::<i32>() {
        return Ok(Tensor::new(
            DType::Int32,
            vec![1],
            val.to_le_bytes().to_vec(),
        ));
    }
    if let Ok(val) = action.extract::<i64>() {
        return Ok(Tensor::new(
            DType::Int64,
            vec![1],
            val.to_le_bytes().to_vec(),
        ));
    }

    // Try as numpy array
    if let Ok(arr) = action.extract::<PyReadonlyArray1<i32>>() {
        let data: Vec<u8> = arr.as_slice()?.iter().flat_map(|x| x.to_le_bytes()).collect();
        return Ok(Tensor::new(DType::Int32, vec![arr.len() as u32], data));
    }
    if let Ok(arr) = action.extract::<PyReadonlyArray1<i64>>() {
        let data: Vec<u8> = arr.as_slice()?.iter().flat_map(|x| x.to_le_bytes()).collect();
        return Ok(Tensor::new(DType::Int64, vec![arr.len() as u32], data));
    }
    if let Ok(arr) = action.extract::<PyReadonlyArray1<f32>>() {
        let data: Vec<u8> = arr.as_slice()?.iter().flat_map(|x| x.to_le_bytes()).collect();
        return Ok(Tensor::new(DType::Float32, vec![arr.len() as u32], data));
    }
    if let Ok(arr) = action.extract::<PyReadonlyArray1<f64>>() {
        let data: Vec<u8> = arr.as_slice()?.iter().flat_map(|x| x.to_le_bytes()).collect();
        return Ok(Tensor::new(DType::Float64, vec![arr.len() as u32], data));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Action must be int, numpy array of int32/int64/float32/float64",
    ))
}

/// Convert batched numpy actions to tensor vec.
pub fn numpy_batch_to_action_tensors(
    py: Python<'_>,
    actions: &Bound<'_, pyo3::types::PyAny>,
    num_envs: usize,
) -> PyResult<Vec<Tensor>> {
    // Try as 1D array (one action per env)
    if let Ok(arr) = actions.extract::<PyReadonlyArray1<i32>>() {
        let slice = arr.as_slice()?;
        if slice.len() != num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                num_envs,
                slice.len()
            )));
        }
        return Ok(slice
            .iter()
            .map(|&v| Tensor::new(DType::Int32, vec![1], v.to_le_bytes().to_vec()))
            .collect());
    }
    if let Ok(arr) = actions.extract::<PyReadonlyArray1<i64>>() {
        let slice = arr.as_slice()?;
        if slice.len() != num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                num_envs,
                slice.len()
            )));
        }
        return Ok(slice
            .iter()
            .map(|&v| Tensor::new(DType::Int64, vec![1], v.to_le_bytes().to_vec()))
            .collect());
    }

    // Try as 2D array (batched continuous actions)
    if let Ok(arr) = actions.extract::<PyReadonlyArray2<f32>>() {
        let shape = arr.shape();
        if shape[0] != num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} envs, got {}",
                num_envs, shape[0]
            )));
        }
        let action_dim = shape[1];
        return Ok((0..num_envs)
            .map(|i| {
                let row: Vec<f32> = (0..action_dim)
                    .map(|j| *arr.get([i, j]).unwrap())
                    .collect();
                let data: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
                Tensor::new(DType::Float32, vec![action_dim as u32], data)
            })
            .collect());
    }

    // Try as Python list
    if let Ok(list) = actions.downcast::<pyo3::types::PyList>() {
        if list.len() != num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                num_envs,
                list.len()
            )));
        }
        let mut tensors = Vec::with_capacity(num_envs);
        for item in list.iter() {
            tensors.push(numpy_to_action_tensor(py, &item)?);
        }
        return Ok(tensors);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Actions must be numpy array or list",
    ))
}

/// Convert observation tensor to numpy array.
pub fn tensor_to_numpy<'py>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyArray1<f32>>> {
    match tensor.dtype {
        DType::Float32 => {
            let data: Vec<f32> = tensor
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(PyArray1::from_vec(py, data))
        }
        DType::Float64 => {
            let data: Vec<f32> = tensor
                .data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect();
            Ok(PyArray1::from_vec(py, data))
        }
        DType::Int32 => {
            let data: Vec<f32> = tensor
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect();
            Ok(PyArray1::from_vec(py, data))
        }
        DType::Int64 => {
            let data: Vec<f32> = tensor
                .data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect();
            Ok(PyArray1::from_vec(py, data))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported dtype: {:?}",
            tensor.dtype
        ))),
    }
}

/// Stack multiple observation tensors into a batched numpy array.
pub fn stack_observations<'py>(
    py: Python<'py>,
    observations: &[Tensor],
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    if observations.is_empty() {
        return Ok(numpy::PyArray2::from_vec2(py, &[]).unwrap());
    }

    let obs_dim: usize = observations[0].shape.iter().map(|&x| x as usize).product();
    let num_envs = observations.len();

    let mut data = Vec::with_capacity(num_envs * obs_dim);
    for obs in observations {
        let obs_data: Vec<f32> = match obs.dtype {
            DType::Float32 => obs
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::Float64 => obs
                .data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            DType::Int32 => obs
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported observation dtype: {:?}",
                    obs.dtype
                )))
            }
        };
        data.extend(obs_data);
    }

    Ok(numpy::PyArray2::from_vec2(
        py,
        &data
            .chunks(obs_dim)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_tensor_from_i32() {
        let tensor = PyTensor::from_i32_array(&[1, 2, 3, 4]);
        assert_eq!(tensor.inner.dtype, DType::Int32);
        assert_eq!(tensor.shape(), vec![4]);
    }

    #[test]
    fn test_py_tensor_from_f32() {
        let tensor = PyTensor::from_f32_array(&[1.0, 2.0, 3.0]);
        assert_eq!(tensor.inner.dtype, DType::Float32);
        assert_eq!(tensor.numel(), 3);
    }

    #[test]
    fn test_py_tensor_to_f32_vec() {
        let tensor = PyTensor::from_f32_array(&[1.5, 2.5, 3.5]);
        let vec = tensor.to_f32_vec().unwrap();
        assert_eq!(vec, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_py_tensor_to_i32_vec() {
        let tensor = PyTensor::from_i32_array(&[10, 20, 30]);
        let vec = tensor.to_i32_vec().unwrap();
        assert_eq!(vec, vec![10, 20, 30]);
    }

    #[test]
    fn test_py_tensor_type_mismatch() {
        let tensor = PyTensor::from_i32_array(&[1, 2, 3]);
        assert!(tensor.to_f32_vec().is_err());
    }

    #[test]
    fn test_py_tensor_i64() {
        let tensor = PyTensor::from_i64_array(&[100i64, 200, 300]);
        let vec = tensor.to_i64_vec().unwrap();
        assert_eq!(vec, vec![100i64, 200, 300]);
    }
}
