use deli_base::Tensor;
use crate::CameraError;

/// Async camera trait for frame capture.
///
/// Implementations provide a `recv` method that asynchronously returns
/// decoded frames as `Tensor<u8>` in HWC layout `[height, width, channels]`.
#[allow(async_fn_in_trait)]
pub trait Camera {
    /// Receive the next frame from the camera.
    ///
    /// Returns a `Tensor<u8>` with shape `[height, width, channels]`.
    /// For RGB images, channels = 3.
    async fn recv(&mut self) -> Result<Tensor<u8>, CameraError>;
}
