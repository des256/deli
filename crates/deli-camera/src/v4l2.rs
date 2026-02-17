use crate::{Camera, CameraConfig, CameraError, Frame};
use std::thread::{self, JoinHandle};
use tokio::sync::mpsc;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, Format, FourCC};

type FrameResult = Result<Frame, CameraError>;

/// V4L2 camera implementation.
pub struct V4l2Camera {
    config: CameraConfig,
    device: Option<Device>,
    receiver: Option<mpsc::Receiver<FrameResult>>,
    thread_handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for V4l2Camera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4l2Camera")
            .field("config", &self.config)
            .field("device", &"<v4l::Device>")
            .field("receiver", &self.receiver.is_some())
            .field("thread_handle", &self.thread_handle.is_some())
            .finish()
    }
}

impl Camera for V4l2Camera {
    async fn recv(&mut self) -> Result<Frame, CameraError> {
        // Ensure capture thread is running
        self.ensure_started()?;

        // Receive next frame from channel
        let receiver = self
            .receiver
            .as_mut()
            .ok_or_else(|| CameraError::Channel("Receiver not initialized".to_string()))?;

        receiver.recv().await.ok_or_else(|| {
            CameraError::Stream(
                "Capture thread terminated; recreate V4l2Camera to restart".to_string(),
            )
        })?
    }
}

impl Drop for V4l2Camera {
    fn drop(&mut self) {
        // Drop the receiver to signal the thread to stop
        drop(self.receiver.take());

        // Wait for the thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl V4l2Camera {
    /// Create a new V4L2 camera with the given configuration.
    ///
    /// Opens the device at `config.device()`, sets MJPEG format at the
    /// requested resolution, and configures the frame rate.
    ///
    /// # Errors
    ///
    /// Returns `CameraError::Device` if:
    /// - The device cannot be opened
    /// - MJPEG format is not supported
    /// - Format or parameter setting fails
    pub fn new(config: CameraConfig) -> Result<Self, CameraError> {
        // Open V4L2 device
        let device = Device::with_path(config.device())?;

        // Set MJPEG format at requested resolution
        let mut format = Format::new(config.width(), config.height(), FourCC::new(b"MJPG"));
        format = Capture::set_format(&device, &format)?;

        // Verify device accepted MJPEG (it might change to a different format)
        if format.fourcc != FourCC::new(b"MJPG") {
            return Err(CameraError::Device(
                "MJPEG format not supported by device".to_string(),
            ));
        }

        // Set frame rate
        let params = v4l::video::capture::Parameters::with_fps(config.fps());
        v4l::video::Capture::set_params(&device, &params)?;

        Ok(Self {
            config,
            device: Some(device),
            receiver: None,
            thread_handle: None,
        })
    }

    /// Start the capture thread if not already running.
    ///
    /// This is called automatically on the first `recv()` call.
    fn ensure_started(&mut self) -> Result<(), CameraError> {
        if self.receiver.is_some() {
            return Ok(());
        }

        // Take ownership of the device
        let device = self
            .device
            .take()
            .ok_or_else(|| CameraError::Device("Device already consumed".to_string()))?;

        let buffer_count = self.config.buffer_count() as usize;
        let (tx, rx) = mpsc::channel(buffer_count);

        // Spawn capture thread
        let handle = thread::spawn(move || {
            Self::capture_loop(device, tx, buffer_count);
        });

        self.receiver = Some(rx);
        self.thread_handle = Some(handle);

        Ok(())
    }

    /// Background thread capture loop.
    ///
    /// Reads MJPEG frames from V4L2 and sends raw JPEG bytes as `Frame::Jpeg`.
    /// Uses `try_send` to drop frames when the channel is full rather than blocking.
    fn capture_loop(device: Device, tx: mpsc::Sender<FrameResult>, buffer_count: usize) {
        // Create mmap stream
        let mut stream = match MmapStream::with_buffers(&device, Type::VideoCapture, buffer_count as u32) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.blocking_send(Err(CameraError::Stream(e.to_string())));
                return;
            }
        };

        loop {
            // Get next frame
            let frame_result = match CaptureStream::next(&mut stream) {
                Ok((frame_data, _metadata)) => {
                    // Copy raw JPEG bytes (buffer is borrowed and only valid until next call)
                    Ok(Frame::Jpeg(frame_data.to_vec()))
                }
                Err(e) => Err(CameraError::Stream(e.to_string())),
            };

            // Send frame through channel, drop if full
            match tx.try_send(frame_result) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Consumer too slow — drop frame silently
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    // Receiver dropped — exit thread
                    break;
                }
            }
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &CameraConfig {
        &self.config
    }
}
