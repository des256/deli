use crate::model::{Multiples, YoloV8Pose};
use crate::postprocess::postprocess;
use crate::{InferError, PoseDetection};
use candle_core::{DType, Device, Tensor as CanTensor};
use deli_base::Tensor;
use std::path::Path;
use std::sync::Arc;

/// Pose detector for running YOLOv8-Pose inference.
///
/// Preprocessing uses nearest-neighbor interpolation for resizing. For better
/// detection accuracy with upscaled images, consider pre-resizing input frames
/// with bilinear or bicubic interpolation before calling `detect()`.
#[derive(Debug)]
pub struct PoseDetector {
    model: Arc<YoloV8Pose>,
    device: Device,
    conf_threshold: f32,
    nms_threshold: f32,
}

impl PoseDetector {
    pub(crate) fn new(
        model_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, InferError> {
        let multiples = detect_model_size(model_path.as_ref())?;

        let weights = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_path.as_ref()],
                DType::F32,
                &device,
            )?
        };

        let model = YoloV8Pose::load(weights, multiples)?;

        Ok(Self {
            model: Arc::new(model),
            device,
            conf_threshold: 0.25,
            nms_threshold: 0.45,
        })
    }

    /// Set confidence and NMS thresholds
    pub fn with_thresholds(mut self, conf: f32, nms: f32) -> Self {
        self.conf_threshold = conf;
        self.nms_threshold = nms;
        self
    }

    /// Run pose detection on a frame (async).
    ///
    /// The frame should be an HWC `Tensor<f32>` with RGB channels in 0-255 range.
    pub async fn detect(&self, frame: &Tensor<f32>) -> Result<Vec<PoseDetection>, InferError> {
        let (preprocessed, original_hw, actual_model_hw) = self.preprocess(frame)?;

        let model = Arc::clone(&self.model);

        let output = tokio::task::spawn_blocking(move || {
            use candle_nn::Module;
            model.forward(&preprocessed)
        })
        .await
        .map_err(|e| InferError::Runtime(format!("inference task failed: {e}")))?
        ?;

        postprocess(
            &output,
            original_hw,
            actual_model_hw,
            self.conf_threshold,
            self.nms_threshold,
        )
        .map_err(InferError::from)
    }

    /// Preprocess input frame: HWC 0-255 -> NCHW 0-1.
    ///
    /// Returns `(preprocessed_tensor, original_hw, actual_model_hw)`.
    /// Uses nearest-neighbor interpolation for resizing.
    #[allow(clippy::type_complexity)]
    fn preprocess(
        &self,
        frame: &Tensor<f32>,
    ) -> Result<(CanTensor, (usize, usize), (usize, usize)), InferError> {
        if frame.shape.len() != 3 {
            return Err(InferError::Shape(format!(
                "expected HWC tensor, got shape {:?}",
                frame.shape
            )));
        }

        let (h, w, c) = (frame.shape[0], frame.shape[1], frame.shape[2]);
        if h == 0 || w == 0 {
            return Err(InferError::Shape(format!(
                "image dimensions must be non-zero, got {}x{}",
                h, w
            )));
        }
        if c != 3 {
            return Err(InferError::Shape(format!(
                "expected 3 channels (RGB), got {}",
                c
            )));
        }

        let original_hw = (h, w);

        // Compute target size: maintain aspect ratio, max dim 640, divisible by 32
        let max_dim = 640;
        let scale = (max_dim as f32 / h.max(w) as f32).min(1.0);
        let target_h = ((h as f32 * scale) as usize / 32 * 32).max(32);
        let target_w = ((w as f32 * scale) as usize / 32 * 32).max(32);
        let actual_model_hw = (target_h, target_w);

        // Create candle tensor from deli-base data: [h, w, 3]
        let tensor = CanTensor::from_vec(frame.data.clone(), (h, w, c), &self.device)?;

        // Permute HWC -> CHW: [3, h, w]
        let tensor = tensor.permute((2, 0, 1))?;

        // Add batch dim: [3, h, w] -> [1, 3, h, w]
        let tensor = tensor.unsqueeze(0)?;

        // Resize to target if needed (nearest-neighbor, requires 4D NCHW input)
        let tensor = if target_h != h || target_w != w {
            tensor.upsample_nearest2d(target_h, target_w)?
        } else {
            tensor
        };

        // Normalize: 0-255 -> 0-1
        let tensor = (tensor / 255.0f64)?;

        Ok((tensor, original_hw, actual_model_hw))
    }
}

/// Auto-detect model size (N/S/M/L/X) from safetensors file.
///
/// Memory-maps the file to read tensor metadata without copying data into memory.
/// The OS page cache ensures subsequent VarBuilder mmap shares the same pages.
fn detect_model_size(path: &Path) -> Result<Multiples, InferError> {
    use safetensors::SafeTensors;

    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
        InferError::Io(format!("failed to memory-map safetensors file: {e}"))
    })?;

    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
        InferError::Io(format!("failed to deserialize safetensors: {e}"))
    })?;

    // Inspect first conv layer to determine width multiplier
    // Key: "net.b1.0.conv.weight", shape: [out_channels, in_channels, kH, kW]
    let key = "net.b1.0.conv.weight";
    let tensor_view = tensors.tensor(key).map_err(|e| {
        InferError::Shape(format!("key '{}' not found in safetensors: {e}", key))
    })?;

    let shape = tensor_view.shape();
    if shape.is_empty() {
        return Err(InferError::Shape(format!(
            "unexpected shape for {}: {:?}",
            key, shape
        )));
    }

    let out_channels = shape[0];

    // First conv output channels = 64 * width_mult
    let width = out_channels as f64 / 64.0;

    match width {
        w if (w - 0.25).abs() < 0.01 => Ok(Multiples::n()),
        w if (w - 0.50).abs() < 0.01 => Ok(Multiples::s()),
        w if (w - 0.75).abs() < 0.01 => Ok(Multiples::m()),
        w if (w - 1.00).abs() < 0.01 => Ok(Multiples::l()),
        w if (w - 1.25).abs() < 0.01 => Ok(Multiples::x()),
        _ => Err(InferError::Shape(format!(
            "unknown model size: width multiplier {:.2}, out_channels {}",
            width, out_channels
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    use crate::model::Multiples;

    fn test_detector() -> PoseDetector {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = YoloV8Pose::load(vb, Multiples::n()).unwrap();
        PoseDetector {
            model: Arc::new(model),
            device,
            conf_threshold: 0.25,
            nms_threshold: 0.45,
        }
    }

    #[test]
    fn test_preprocess_shape_and_normalization() {
        let detector = test_detector();
        let data = vec![128.0f32; 480 * 640 * 3];
        let frame = Tensor::new(vec![480, 640, 3], data).unwrap();

        let (tensor, original_hw, model_hw) = detector.preprocess(&frame).unwrap();

        assert_eq!(original_hw, (480, 640));
        // scale = 640/640 = 1.0, target = (480/32)*32=480, (640/32)*32=640
        assert_eq!(model_hw, (480, 640));
        assert_eq!(tensor.dims(), &[1, 3, 480, 640]);

        // Values normalized: 128/255 ≈ 0.502
        let flat = tensor.flatten_all().unwrap();
        let val: f32 = flat.to_vec1().unwrap()[0];
        assert!((val - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_small_image_min_32() {
        let detector = test_detector();
        let data = vec![0.0f32; 16 * 16 * 3];
        let frame = Tensor::new(vec![16, 16, 3], data).unwrap();

        let (_tensor, _original_hw, model_hw) = detector.preprocess(&frame).unwrap();

        assert!(model_hw.0 >= 32);
        assert!(model_hw.1 >= 32);
    }

    #[test]
    fn test_preprocess_rejects_wrong_shape() {
        let detector = test_detector();
        let data = vec![0.0f32; 100 * 100];
        let frame = Tensor::new(vec![100, 100], data).unwrap();

        let result = detector.preprocess(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_preprocess_rejects_zero_dimensions() {
        let detector = test_detector();
        let frame = Tensor::new(vec![0, 640, 3], vec![]).unwrap();
        let result = detector.preprocess(&frame);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-zero"));
    }

    #[tokio::test]
    async fn test_detect_full_pipeline() {
        let detector = test_detector();
        let data = vec![128.0f32; 64 * 64 * 3];
        let frame = Tensor::new(vec![64, 64, 3], data).unwrap();

        // With random (zero) weights, inference should complete without panic
        let result = detector.detect(&frame).await;
        assert!(result.is_ok());
    }
}
