// Pocket TTS ONNX inference module
//
// Implements Kyutai's Pocket TTS (100M params, MIT license) using ONNX Runtime.
// Architecture: 5 ONNX models (text_conditioner, flow_lm_main, flow_lm_flow,
// mimi_encoder, mimi_decoder) orchestrated via PocketCore.
//
// Public API follows the same Sink/Stream pattern as Kokoro TTS.

// Module declarations
pub(crate) mod core;
pub(crate) mod pocket;
pub(crate) mod snapshot;

// Re-exports
pub use pocket::PocketTts;

#[cfg(test)]
#[path = "tests/pocket_tts_test.rs"]
mod pocket_tts_test;
