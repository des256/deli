use {
    super::{postprocess::postprocess, types::PoseDetections, Multiples, YoloV8Pose},
    crate::InferError,
    base::Tensor,
    candle_core::{DType, Device, Tensor as CanTensor},
    futures_core::Stream,
    futures_sink::Sink,
    image::Image,
    std::{
        collections::VecDeque,
        fmt,
        future::Future,
        path::Path,
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, Waker},
    },
};

/// Pose detector for running YOLOv8-Pose inference.
///
/// Implements `Sink<Image>` to accept input images and
/// `Stream<Item = Result<PoseDetections>>` to produce pose detections.
///
/// Each `Image` sent via the Sink maps 1:1 to a `PoseDetections` yielded
/// from the Stream. Closing the sink signals no more input; the stream
/// ends once all pending images are processed.
///
/// Preprocessing uses nearest-neighbor interpolation for resizing. For better
/// detection accuracy with upscaled images, consider pre-resizing input frames
/// with bilinear or bicubic interpolation before sending.
pub struct PoseDetector {
    model: Arc<YoloV8Pose>,
    device: Device,
    conf_threshold: f32,
    nms_threshold: f32,
    pending: VecDeque<Image>,
    closed: bool,
    inflight: Option<Pin<Box<dyn Future<Output = Result<PoseDetections, InferError>> + Send>>>,
    stream_waker: Option<Waker>,
}

impl fmt::Debug for PoseDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoseDetector")
            .field("device", &self.device)
            .field("conf_threshold", &self.conf_threshold)
            .field("nms_threshold", &self.nms_threshold)
            .field("pending", &self.pending.len())
            .field("closed", &self.closed)
            .field("inflight", &self.inflight.is_some())
            .finish()
    }
}

impl PoseDetector {
    pub(crate) fn new(model_path: impl AsRef<Path>, device: Device) -> Result<Self, InferError> {
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
            pending: VecDeque::new(),
            closed: false,
            inflight: None,
            stream_waker: None,
        })
    }

    /// Set confidence and NMS thresholds
    pub fn with_thresholds(mut self, conf: f32, nms: f32) -> Self {
        self.conf_threshold = conf;
        self.nms_threshold = nms;
        self
    }

    /// Spawn detection of the given image as an inflight future.
    fn start_detection(&mut self, image: Image) {
        let model = Arc::clone(&self.model);
        let device = self.device.clone();
        let conf_threshold = self.conf_threshold;
        let nms_threshold = self.nms_threshold;

        self.inflight = Some(Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                use candle_nn::Module;
                let frame = image_to_f32(image)?;
                let (preprocessed, original_hw, model_hw) = preprocess(&frame, &device)?;
                let output = model.forward(&preprocessed).map_err(InferError::from)?;
                postprocess(
                    &output,
                    original_hw,
                    model_hw,
                    conf_threshold,
                    nms_threshold,
                )
                .map_err(InferError::from)
            })
            .await
            .map_err(|e| InferError::Runtime(format!("inference task failed: {e}")))?
        }));
    }
}

/// Convert an `Image` to `Tensor<f32>` in the 0-255 range expected by `preprocess`.
/// Only accepts Rgb8 format. Other formats must be converted first.
fn image_to_f32(image: Image) -> Result<Tensor<f32>, InferError> {
    if image.format != image::PixelFormat::Rgb8 {
        return Err(InferError::Runtime(format!(
            "image_to_f32 requires Rgb8 format, got {:?}",
            image.format
        )));
    }

    let data: Vec<f32> = image.data.iter().map(|&v| v as f32).collect();
    let shape = vec![image.size.y, image.size.x, 3];
    Tensor::new(shape, data)
        .map_err(|e| InferError::Runtime(format!("failed to create tensor: {e}")))
}

impl Sink<Image> for PoseDetector {
    type Error = InferError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), InferError>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: Image) -> Result<(), InferError> {
        let this = self.get_mut();
        this.pending.push_back(item);
        if let Some(waker) = this.stream_waker.take() {
            waker.wake();
        }
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), InferError>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), InferError>> {
        let this = self.get_mut();
        this.closed = true;
        if let Some(waker) = this.stream_waker.take() {
            waker.wake();
        }
        Poll::Ready(Ok(()))
    }
}

impl Stream for PoseDetector {
    type Item = Result<PoseDetections, InferError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Poll inflight detection
        if let Some(fut) = this.inflight.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    this.inflight = None;
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Dequeue next image and start detection
        if let Some(image) = this.pending.pop_front() {
            this.start_detection(image);
            if let Some(fut) = this.inflight.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(result) => {
                        this.inflight = None;
                        return Poll::Ready(Some(result));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }

        // Stream complete when closed, pending empty, and no inflight
        if this.closed {
            return Poll::Ready(None);
        }

        // Nothing to do yet — park
        this.stream_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

// Ensure PoseDetector is Send
fn _assert_send() {
    fn assert<T: Send>() {}
    assert::<PoseDetector>();
}

/// Preprocess input frame: HWC 0-255 -> NCHW 0-1.
///
/// Returns `(preprocessed_tensor, original_hw, actual_model_hw)`.
/// Uses nearest-neighbor interpolation for resizing.
fn preprocess(
    frame: &Tensor<f32>,
    device: &Device,
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
    let tensor = CanTensor::from_vec(frame.data.clone(), (h, w, c), device)?;

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

/// Auto-detect model size (N/S/M/L/X) from safetensors file.
///
/// Memory-maps the file to read tensor metadata without copying data into memory.
/// The OS page cache ensures subsequent VarBuilder mmap shares the same pages.
fn detect_model_size(path: &Path) -> Result<Multiples, InferError> {
    use safetensors::SafeTensors;

    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| InferError::Io(format!("failed to memory-map safetensors file: {e}")))?;

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| InferError::Io(format!("failed to deserialize safetensors: {e}")))?;

    // Inspect first conv layer to determine width multiplier
    // Key: "net.b1.0.conv.weight", shape: [out_channels, in_channels, kH, kW]
    let key = "net.b1.0.conv.weight";
    let tensor_view = tensors
        .tensor(key)
        .map_err(|e| InferError::Shape(format!("key '{}' not found in safetensors: {e}", key)))?;

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
    use crate::pose_detector::Multiples;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

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
            pending: VecDeque::new(),
            closed: false,
            inflight: None,
            stream_waker: None,
        }
    }

    #[test]
    fn test_preprocess_shape_and_normalization() {
        let device = Device::Cpu;
        let data = vec![128.0f32; 480 * 640 * 3];
        let frame = Tensor::new(vec![480, 640, 3], data).unwrap();

        let (tensor, original_hw, model_hw) = preprocess(&frame, &device).unwrap();

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
        let device = Device::Cpu;
        let data = vec![0.0f32; 16 * 16 * 3];
        let frame = Tensor::new(vec![16, 16, 3], data).unwrap();

        let (_tensor, _original_hw, model_hw) = preprocess(&frame, &device).unwrap();

        assert!(model_hw.0 >= 32);
        assert!(model_hw.1 >= 32);
    }

    #[test]
    fn test_preprocess_rejects_wrong_shape() {
        let device = Device::Cpu;
        let data = vec![0.0f32; 100 * 100];
        let frame = Tensor::new(vec![100, 100], data).unwrap();

        let result = preprocess(&frame, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_preprocess_rejects_zero_dimensions() {
        let device = Device::Cpu;
        let frame = Tensor::new(vec![0, 640, 3], vec![]).unwrap();
        let result = preprocess(&frame, &device);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-zero"));
    }

    #[tokio::test]
    async fn test_detect_full_pipeline() {
        use futures_util::{SinkExt, StreamExt};

        let mut detector = test_detector();
        let data = vec![128u8; 64 * 64 * 3];
        let size = base::Vec2::new(64, 64);
        let frame = Image::new(size, data, image::PixelFormat::Rgb8);

        // Send image via Sink and close
        detector.send(frame).await.unwrap();
        detector.close().await.unwrap();

        // With random (zero) weights, inference should complete without panic
        let result = detector.next().await;
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());
    }
}
