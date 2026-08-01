// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Wasmtime host bindings for the WasmRL environment world.

wasmtime::component::bindgen!({
    world: "env",
    path: "../wasmrl-wit/wit",
});

use exports::wasmrl::env::{environment, snapshot};
use wasmrl_wit::{DType, SnapshotData, StepResult, Tensor};

pub(crate) fn lower_tensor(tensor: &Tensor) -> environment::Tensor {
    environment::Tensor {
        dtype: match tensor.dtype {
            DType::Float32 => environment::Dtype::Float32,
            DType::Float64 => environment::Dtype::Float64,
            DType::Int32 => environment::Dtype::Int32,
            DType::Int64 => environment::Dtype::Int64,
            DType::Uint8 => environment::Dtype::Uint8,
            DType::Boolean => environment::Dtype::Boolean,
        },
        shape: tensor.shape.clone(),
        data: tensor.data.clone(),
    }
}

pub(crate) fn lift_tensor(tensor: environment::Tensor) -> Tensor {
    Tensor {
        dtype: match tensor.dtype {
            environment::Dtype::Float32 => DType::Float32,
            environment::Dtype::Float64 => DType::Float64,
            environment::Dtype::Int32 => DType::Int32,
            environment::Dtype::Int64 => DType::Int64,
            environment::Dtype::Uint8 => DType::Uint8,
            environment::Dtype::Boolean => DType::Boolean,
        },
        shape: tensor.shape,
        data: tensor.data,
    }
}

pub(crate) fn lift_step_result(result: environment::StepResult) -> StepResult {
    StepResult {
        observation: lift_tensor(result.observation),
        reward: result.reward,
        terminated: result.terminated,
        truncated: result.truncated,
        info: result.info,
    }
}

pub(crate) fn lower_snapshot(data: &SnapshotData) -> snapshot::SnapshotData {
    snapshot::SnapshotData {
        version: data.version,
        data: data.data.clone(),
    }
}

pub(crate) fn lift_snapshot(data: snapshot::SnapshotData) -> SnapshotData {
    SnapshotData {
        version: data.version,
        data: data.data,
    }
}
