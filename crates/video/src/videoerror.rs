use std::fmt;

#[derive(Debug)]
pub enum VideoError {
    Device(String),
    Stream(String),
    Decode(image::ImageError),
    Channel(String),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Device(msg) => write!(f, "device error: {msg}"),
            VideoError::Stream(msg) => write!(f, "stream error: {msg}"),
            VideoError::Decode(err) => write!(f, "decode error: {err}"),
            VideoError::Channel(msg) => write!(f, "channel error: {msg}"),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<std::io::Error> for VideoError {
    fn from(err: std::io::Error) -> Self {
        VideoError::Device(err.to_string())
    }
}

impl From<image::ImageError> for VideoError {
    fn from(err: image::ImageError) -> Self {
        VideoError::Decode(err)
    }
}
