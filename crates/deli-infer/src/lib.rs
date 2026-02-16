mod error;
mod inference;
mod pose_detector;

pub use error::InferError;
pub use inference::Inference;
pub use pose_detector::{CocoKeypoint, Keypoint, PoseDetection, PoseDetector};
