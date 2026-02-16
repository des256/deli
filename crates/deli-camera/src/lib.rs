//! Camera capture abstraction for the deli ecosystem.
//!
//! This crate provides a unified `Camera` trait for async frame capture,
//! with backend implementations for various camera APIs.

pub mod config;
pub mod convert;
pub mod error;
pub mod traits;

#[cfg(feature = "v4l2")]
pub mod v4l2;

#[cfg(feature = "rpicam")]
pub mod rpicam;

pub use config::CameraConfig;
pub use error::CameraError;
pub use traits::Camera;

#[cfg(feature = "v4l2")]
pub use v4l2::V4l2Camera;

#[cfg(feature = "rpicam")]
pub use rpicam::RPiCamera;
