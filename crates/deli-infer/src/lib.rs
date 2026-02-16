mod detector;
mod error;
mod inference;
mod model;
mod postprocess;
mod types;

pub use detector::PoseDetector;
pub use error::InferError;
pub use inference::Inference;
pub use types::{CocoKeypoint, Keypoint, PoseDetection};
