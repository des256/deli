pub enum Transcription {
    Partial { text: String, confidence: f32 },
    Final { text: String, confidence: f32 },
    Cancelled,
}

pub mod sherpa;
pub use sherpa::Sherpa;

pub mod whisper;
pub use whisper::Whisper;
