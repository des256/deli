use deli_base::Tensor;

/// A captured video frame, either decoded RGB pixels or raw JPEG bytes.
#[derive(Debug, Clone)]
pub enum VideoFrame {
    /// RGB pixel data as a `Tensor<u8>` with shape `[height, width, 3]`.
    Rgb(Tensor<u8>),
    /// Raw JPEG-encoded image bytes.
    Jpeg(Vec<u8>),
}
