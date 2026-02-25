#[derive(Debug)]
pub enum AudioError {
    Device(String), // device errors (not available, initialization failure, etc.)
    Stream(String), // streaming errors (disconnection, read error, etc.)
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Device(error) => write!(f, "Audio device error: {error}"),
            AudioError::Stream(error) => write!(f, "Audio streaming error: {error}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioSample {
    pub data: Vec<i16>,
    pub sample_rate: usize,
}

mod audioin;
pub use audioin::*;

mod audioout;
pub use audioout::*;
