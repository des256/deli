pub mod backend;
pub mod device;
pub mod error;
pub mod registry;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub mod pose;

pub use backend::{Backend, ModelSource, Session};
pub use device::Device;
pub use error::InferError;
pub use registry::{create_registry, BackendRegistry};

#[cfg(feature = "onnx")]
pub use pose::{
    iou, postprocess, preprocess, Keypoint, KeypointIndex, LetterboxInfo, PoseDetection,
    YoloPoseEstimator, COCO_KEYPOINT_COUNT,
};
