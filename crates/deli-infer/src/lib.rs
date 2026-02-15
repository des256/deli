pub mod backend;
pub mod device;
pub mod error;
pub mod registry;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use backend::{Backend, ModelSource, Session};
pub use device::Device;
pub use error::InferError;
pub use registry::{create_registry, BackendRegistry};
