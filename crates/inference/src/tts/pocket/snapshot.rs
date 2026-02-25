// State snapshot/restore for Pocket TTS ONNX state tensors.
//
// Captures flow_main and mimi_decoder state tensors as raw bytes
// so they can be restored for each new utterance without re-running
// voice conditioning.

use {
    crate::error::{InferError, Result},
    onnx::Value,
};

/// Snapshot of all ONNX state tensors as raw bytes
pub(crate) struct StateSnapshot {
    pub flow_states: Vec<TensorSnapshot>,
    //pub mimi_states: Vec<TensorSnapshot>,
}

/// Single tensor snapshot: raw bytes + shape + element type
pub(crate) struct TensorSnapshot {
    data: Vec<u8>,
    shape: Vec<i64>,
    elem_type: onnx::ffi::ONNXTensorElementDataType,
}

/// Extract a slice of Values into a Vec of TensorSnapshots
pub(crate) fn snapshot_values(states: &[Value]) -> Result<Vec<TensorSnapshot>> {
    let mut snapshots = Vec::with_capacity(states.len());
    for state in states {
        let shape = state
            .tensor_shape()
            .map_err(|e| InferError::Runtime(format!("Failed to get tensor shape: {}", e)))?;
        let elem_type = state.tensor_element_type().map_err(|e| {
            InferError::Runtime(format!("Failed to get tensor element type: {}", e))
        })?;
        let data = extract_raw_bytes(state, elem_type)?;
        snapshots.push(TensorSnapshot {
            data,
            shape,
            elem_type,
        });
    }
    Ok(snapshots)
}

/// Restore Values from a slice of TensorSnapshots
pub(crate) fn restore_values(snapshots: &[TensorSnapshot]) -> Result<Vec<Value>> {
    let mut values = Vec::with_capacity(snapshots.len());
    for snap in snapshots {
        let shape_usize: Vec<usize> = snap.shape.iter().map(|&d| d as usize).collect();
        let value = restore_typed_value(&shape_usize, &snap.data, snap.elem_type)?;
        values.push(value);
    }
    Ok(values)
}

/// Extract raw bytes from a Value based on its element type
fn extract_raw_bytes(
    state: &Value,
    elem_type: onnx::ffi::ONNXTensorElementDataType,
) -> Result<Vec<u8>> {
    match elem_type {
        onnx::ffi::ONNXTensorElementDataType::Float => {
            let slice = state
                .extract_tensor::<f32>()
                .map_err(|e| InferError::Runtime(format!("Failed to extract f32 tensor: {}", e)))?;
            Ok(typed_to_bytes(slice))
        }
        onnx::ffi::ONNXTensorElementDataType::Int64 => {
            let slice = state
                .extract_tensor::<i64>()
                .map_err(|e| InferError::Runtime(format!("Failed to extract i64 tensor: {}", e)))?;
            Ok(typed_to_bytes(slice))
        }
        onnx::ffi::ONNXTensorElementDataType::Bool => {
            let slice = state.extract_tensor::<bool>().map_err(|e| {
                InferError::Runtime(format!("Failed to extract bool tensor: {}", e))
            })?;
            Ok(typed_to_bytes(slice))
        }
        _ => Err(InferError::Runtime(format!(
            "Unsupported element type {:?} in snapshot",
            elem_type
        ))),
    }
}

/// Restore a single typed Value from raw bytes
fn restore_typed_value(
    shape: &[usize],
    data: &[u8],
    elem_type: onnx::ffi::ONNXTensorElementDataType,
) -> Result<Value> {
    match elem_type {
        onnx::ffi::ONNXTensorElementDataType::Float => {
            let typed = bytes_to_typed::<f32>(data);
            Ok(Value::from_slice::<f32>(shape, typed)?)
        }
        onnx::ffi::ONNXTensorElementDataType::Int64 => {
            let typed = bytes_to_typed::<i64>(data);
            Ok(Value::from_slice::<i64>(shape, typed)?)
        }
        onnx::ffi::ONNXTensorElementDataType::Bool => {
            let typed = bytes_to_typed::<bool>(data);
            Ok(Value::from_slice::<bool>(shape, typed)?)
        }
        _ => Err(InferError::Runtime(format!(
            "Unsupported element type {:?} in restore",
            elem_type
        ))),
    }
}

/// Convert a typed slice to raw bytes
fn typed_to_bytes<T>(slice: &[T]) -> Vec<u8> {
    if slice.is_empty() {
        return Vec::new();
    }
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
    .to_vec()
}

/// Convert raw bytes back to a typed slice reference
fn bytes_to_typed<T>(data: &[u8]) -> &[T] {
    if data.is_empty() {
        return &[];
    }
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const T,
            data.len() / std::mem::size_of::<T>(),
        )
    }
}
