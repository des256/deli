pub struct AsrInput {
    pub audio: Vec<i16>,
}

pub struct AsrOutput {
    pub text: String,
}

pub enum Transcription {
    Partial { text: String, confidence: f32 },
    Final { text: String, confidence: f32 },
    Cancelled,
}

pub mod parakeet;
pub use parakeet::Parakeet;

//pub mod sherpa;
//pub use sherpa::Sherpa;
