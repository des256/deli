use crate::Device;
use std::fmt;

#[derive(Debug)]
pub enum InferError {
    BackendError(String),
    ShapeMismatch { expected: String, got: String },
    UnsupportedDevice(Device),
    ModelLoad(String),
    UnsupportedDtype(String),
    InvalidInput {
        name: String,
        expected_names: Vec<String>,
    },
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::BackendError(msg) => write!(f, "backend error: {msg}"),
            InferError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            InferError::UnsupportedDevice(device) => {
                write!(f, "unsupported device: {device}")
            }
            InferError::ModelLoad(msg) => write!(f, "model load error: {msg}"),
            InferError::UnsupportedDtype(dtype) => write!(f, "unsupported dtype: {dtype}"),
            InferError::InvalidInput {
                name,
                expected_names,
            } => {
                write!(
                    f,
                    "invalid input name: '{}'. Expected one of: {:?}",
                    name, expected_names
                )
            }
        }
    }
}

impl std::error::Error for InferError {}
