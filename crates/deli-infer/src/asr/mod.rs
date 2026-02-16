pub(crate) mod attention;
pub(crate) mod audio;
pub mod config;
pub(crate) mod model;
pub(crate) mod recognizer;
pub(crate) mod token_decoder;

// Public API
pub use config::Config;
pub use recognizer::SpeechRecognizer;

// Internal — exposed for integration tests only, not part of public API
#[doc(hidden)]
pub use audio::{load_mel_filters, pcm_to_mel};
#[doc(hidden)]
pub use model::Whisper;
#[doc(hidden)]
pub use token_decoder::{token_id, DecodingResult, TokenDecoder};
