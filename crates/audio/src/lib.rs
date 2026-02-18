//! Audio capture and playback abstraction for the deli ecosystem.
//!
//! This crate provides a unified interface for async audio I/O with PulseAudio,
//! with backend implementation using the PulseAudio Simple API.

pub mod audio_in;
pub mod audio_out;
pub mod audiosample;
pub mod device;
pub mod error;

pub use audio_in::AudioIn;
pub use audio_out::AudioOut;
pub use audiosample::{AudioData, AudioSample};
pub use device::{AudioDevice, list_devices};
pub use error::AudioError;
