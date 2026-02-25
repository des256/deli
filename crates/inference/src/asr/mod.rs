pub enum Transcription {
    Partial { text: String, confidence: f32 },
    Final { text: String, confidence: f32 },
    Cancelled,
}

pub mod asr;
pub use asr::Asr;

pub mod sherpa;
pub use sherpa::Sherpa;

pub mod parakeet;
pub use parakeet::Parakeet;
