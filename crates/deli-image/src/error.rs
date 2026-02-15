use std::fmt;

#[derive(Debug)]
pub enum ImageError {
    Decode(String),
    Tensor(deli_base::TensorError),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::Decode(msg) => write!(f, "decode error: {msg}"),
            ImageError::Tensor(err) => write!(f, "tensor error: {err}"),
        }
    }
}

impl std::error::Error for ImageError {}

impl From<image::ImageError> for ImageError {
    fn from(err: image::ImageError) -> Self {
        ImageError::Decode(err.to_string())
    }
}

impl From<deli_base::TensorError> for ImageError {
    fn from(err: deli_base::TensorError) -> Self {
        ImageError::Tensor(err)
    }
}
