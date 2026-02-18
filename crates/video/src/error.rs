use std::fmt;

#[derive(Debug)]
pub enum CameraError {
    Device(String),
    Stream(String),
    Decode(image::ImageError),
    Channel(String),
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CameraError::Device(msg) => write!(f, "device error: {msg}"),
            CameraError::Stream(msg) => write!(f, "stream error: {msg}"),
            CameraError::Decode(err) => write!(f, "decode error: {err}"),
            CameraError::Channel(msg) => write!(f, "channel error: {msg}"),
        }
    }
}

impl std::error::Error for CameraError {}

impl From<std::io::Error> for CameraError {
    fn from(err: std::io::Error) -> Self {
        CameraError::Device(err.to_string())
    }
}

impl From<image::ImageError> for CameraError {
    fn from(err: image::ImageError) -> Self {
        CameraError::Decode(err)
    }
}
