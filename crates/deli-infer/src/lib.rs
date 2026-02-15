pub mod backend;
pub mod backendregistry;
pub mod backends;
pub mod device;
pub mod error;
pub mod modelsource;
pub mod session;

#[cfg(feature = "onnx")]
pub mod pose;

pub use backend::Backend;
pub use backendregistry::{create_registry, BackendRegistry};
pub use device::Device;
pub use error::InferError;
pub use modelsource::ModelSource;
pub use session::Session;

#[cfg(feature = "onnx")]
pub use pose::{
    iou, postprocess, preprocess, Keypoint, KeypointIndex, LetterboxInfo, PoseDetection,
    YoloPoseEstimator, COCO_KEYPOINT_COUNT,
};
