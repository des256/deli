use {base::Tensor, std::fmt};

#[derive(Debug)]
pub enum AudioError {
    Device(String), // device errors (not available, initialization failure, etc.)
    Stream(String), // streaming errors (disconnection, read error, etc.)
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::Device(error) => write!(f, "Audio device error: {error}"),
            AudioError::Stream(error) => write!(f, "Audio streaming error: {error}"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AudioData {
    Pcm(Tensor<i16>),
}

#[derive(Debug, Clone)]
pub struct AudioSample {
    pub data: AudioData,
    pub sample_rate: usize,
}

mod audioin;
pub use audioin::*;

mod audioout;
pub use audioout::*;
