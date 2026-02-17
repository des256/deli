//! Audio capture abstraction for the deli ecosystem.
//!
//! This crate provides a unified interface for async audio capture from PulseAudio,
//! with backend implementation using the PulseAudio Simple API.

pub mod audio_in;
pub mod device;
pub mod error;

pub use audio_in::AudioIn;
pub use device::{list_devices, AudioDevice};
pub use error::AudioError;
