pub mod silero;
pub use silero::SileroVad;

#[cfg(test)]
#[path = "tests/silero_vad_test.rs"]
mod silero_vad_test;
