use std::fmt;

#[derive(Debug)]
pub enum AudioError {
    Device(String),
    Stream(String),
    Channel(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::Device(msg) => write!(f, "device error: {msg}"),
            AudioError::Stream(msg) => write!(f, "stream error: {msg}"),
            AudioError::Channel(msg) => write!(f, "channel error: {msg}"),
        }
    }
}

impl std::error::Error for AudioError {}

impl From<std::io::Error> for AudioError {
    fn from(err: std::io::Error) -> Self {
        AudioError::Device(err.to_string())
    }
}
