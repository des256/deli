use crate::CameraError;
use deli_base::Tensor;

/// A captured video frame, either decoded RGB pixels or raw JPEG bytes.
#[derive(Debug, Clone)]
pub enum VideoFrame {
    /// RGB pixel data as a `Tensor<u8>` with shape `[height, width, 3]`.
    Rgb(Tensor<u8>),
    /// Raw JPEG-encoded image bytes.
    Jpeg(Vec<u8>),
}

/// Async camera trait for frame capture.
///
/// Implementations provide a `recv` method that asynchronously returns
/// captured frames. The frame format depends on the underlying stream:
/// - MJPEG streams produce `VideoFrame::Jpeg`
/// - RGB or YUYV streams produce `VideoFrame::Rgb`
#[allow(async_fn_in_trait)]
pub trait Camera {
    /// Receive the next frame from the camera.
    async fn recv(&mut self) -> Result<VideoFrame, CameraError>;
}
