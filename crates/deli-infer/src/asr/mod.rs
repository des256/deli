pub(crate) mod attention;
pub(crate) mod audio;
pub mod config;
pub(crate) mod model;
pub(crate) mod recognizer;
pub(crate) mod token_decoder;
pub mod transcribed;

// Public API
pub use config::Config;
pub use recognizer::Whisper;
pub use transcribed::Transcription;

// Internal — exposed for integration tests only, not part of public API
#[doc(hidden)]
pub use audio::{load_mel_filters, pcm_to_mel};
#[doc(hidden)]
pub use model::Whisper as WhisperModel;
#[doc(hidden)]
pub use token_decoder::{DecodingResult, TokenDecoder, token_id};
