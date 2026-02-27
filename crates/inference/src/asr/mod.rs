#[derive(Clone, Debug)]
pub struct AsrInput<T: Clone + Send + 'static> {
    pub payload: T,
    pub audio: Vec<i16>,
}

#[derive(Clone, Debug)]
pub struct AsrOutput<T: Clone + Send + 'static> {
    pub payload: T,
    pub text: String,
}

pub enum Transcription {
    Partial { text: String, confidence: f32 },
    Final { text: String, confidence: f32 },
    Cancelled,
}

pub mod parakeet;

//pub mod sherpa;
//pub use sherpa::Sherpa;
