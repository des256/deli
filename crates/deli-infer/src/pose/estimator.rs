use crate::InferError;
use deli_base::Tensor;

use super::postprocess::postprocess;
use super::preprocess::preprocess;
use super::types::PoseDetection;

/// YOLO pose estimation pipeline
///
/// Integrates preprocessing, ONNX inference, and post-processing into a single
/// `estimate()` call. Handles letterbox resize, model inference, NMS, and coordinate
/// rescaling automatically.
pub struct YoloPoseEstimator {
    session: Box<dyn crate::Session>,
    conf_threshold: f32,
    iou_threshold: f32,
}

impl YoloPoseEstimator {
    /// Create a new YOLO pose estimator
    ///
    /// # Arguments
    /// * `model` - Model source (file path or in-memory bytes)
    /// * `device` - Device to run inference on (CPU, CUDA, TensorRT)
    ///
    /// # Returns
    /// Estimator with default thresholds (conf=0.25, iou=0.45)
    pub fn new(
        model: crate::ModelSource,
        device: crate::Device,
    ) -> Result<Self, InferError> {
        use crate::backends::OnnxBackend;
        use crate::Backend;

        let backend = OnnxBackend;
        let session = backend.load_model(model, device)?;

        Ok(Self {
            session,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        })
    }

    /// Set confidence threshold (builder pattern)
    pub fn with_conf_threshold(mut self, threshold: f32) -> Self {
        self.conf_threshold = threshold;
        self
    }

    /// Set IoU threshold for NMS (builder pattern)
    pub fn with_iou_threshold(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Get current confidence threshold
    pub fn conf_threshold(&self) -> f32 {
        self.conf_threshold
    }

    /// Get current IoU threshold
    pub fn iou_threshold(&self) -> f32 {
        self.iou_threshold
    }

    /// Run pose estimation on an image
    ///
    /// # Arguments
    /// * `image` - Input image as Tensor<f32> with shape [H, W, 3] and values in [0, 255]
    ///
    /// # Returns
    /// Vector of detected poses sorted by confidence descending
    pub fn estimate(&mut self, image: &Tensor<f32>) -> Result<Vec<PoseDetection>, InferError> {
        // Validate input shape
        if image.shape.len() != 3 {
            return Err(InferError::ShapeMismatch {
                expected: format!("[H, W, 3]"),
                got: format!("{:?}", image.shape),
            });
        }
        if image.shape[2] != 3 {
            return Err(InferError::ShapeMismatch {
                expected: format!("3 channels"),
                got: format!("{} channels", image.shape[2]),
            });
        }

        // Preprocess
        let (preprocessed, letterbox) = preprocess(image)?;

        // Run inference
        let input_name = self
            .session
            .input_names()
            .first()
            .ok_or_else(|| InferError::BackendError("model has no inputs".to_string()))?
            .clone();

        let outputs = self.session.run(&[(input_name.as_str(), preprocessed)])?;

        // Extract output tensor
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| InferError::BackendError("model produced no outputs".to_string()))?;

        // Post-process
        let detections = postprocess(output, &letterbox, self.conf_threshold, self.iou_threshold)?;

        Ok(detections)
    }
}
