use crate::{Camera, CameraConfig, CameraError, Frame};
use deli_base::Tensor;
use rslibcamlitelib::{ExternalCallback, LibCamClient, StreamFormat, StreamParams};
use tokio::sync::mpsc;

type FrameResult = Result<Frame, CameraError>;

/// Callback handler that bridges rslibcamlite frames to a tokio channel.
struct FrameCallback {
    tx: mpsc::Sender<FrameResult>,
    width: usize,
    height: usize,
}

impl ExternalCallback for FrameCallback {
    unsafe fn callbackLowres(&mut self, bytes: *mut u8, count: usize) {
        let slice = unsafe { std::slice::from_raw_parts(bytes, count) };
        let data = slice.to_vec();

        let frame = Tensor::new(vec![self.height, self.width, 3], data)
            .map(Frame::Rgb)
            .map_err(|e| CameraError::Stream(e.to_string()));

        let _ = self.tx.try_send(frame);
    }

    unsafe fn callbackH264(
        &mut self,
        _bytes: *mut u8,
        _count: usize,
        _timestamp_us: i64,
        _keyframe: bool,
    ) {
        // Not used — only lowres RGB stream is configured
    }
}

/// RPi Camera implementation using rslibcamlite.
///
/// Captures RGB frames from the Raspberry Pi camera using the lowres stream.
///
/// **Note:** The `config.device()` field is ignored. rslibcamlite always uses
/// the system's default camera.
///
/// **Note:** Camera setup is deferred until the first `recv()` call.
pub struct RPiCamera {
    config: CameraConfig,
    receiver: Option<mpsc::Receiver<FrameResult>>,
    client: Option<LibCamClient>,
}

impl std::fmt::Debug for RPiCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RPiCamera")
            .field("config", &self.config)
            .field("receiver", &self.receiver.is_some())
            .field("client", &self.client.is_some())
            .finish()
    }
}

impl Camera for RPiCamera {
    async fn recv(&mut self) -> Result<Frame, CameraError> {
        self.ensure_started()?;

        let receiver = self
            .receiver
            .as_mut()
            .ok_or_else(|| CameraError::Channel("Receiver not initialized".to_string()))?;

        receiver.recv().await.ok_or_else(|| {
            CameraError::Stream(
                "Capture stopped; recreate RPiCamera to restart".to_string(),
            )
        })?
    }
}

impl Drop for RPiCamera {
    fn drop(&mut self) {
        // Drop the receiver to signal the callback to stop sending
        drop(self.receiver.take());

        // Stop the client capture
        if let Some(ref client) = self.client {
            client.stop();
        }
        self.client.take();
    }
}

impl RPiCamera {
    /// Create a new RPi Camera with the given configuration.
    ///
    /// The camera is not opened until the first `recv()` call.
    ///
    /// **Note:** The `config.device()` field is ignored. rslibcamlite always uses
    /// the system's default camera. If the device field is set to a non-default
    /// value, a warning is logged.
    pub fn new(config: CameraConfig) -> Result<Self, CameraError> {
        if config.device() != "/dev/video0" {
            log::warn!(
                "RPiCamera ignores config.device() (got: {}). Uses system default camera.",
                config.device()
            );
        }

        Ok(Self {
            config,
            receiver: None,
            client: None,
        })
    }

    /// Start capture if not already running.
    ///
    /// This is called automatically on the first `recv()` call. Creates a
    /// `LibCamClient`, configures the lowres RGB stream, registers a frame
    /// callback, and starts capture in a background thread.
    fn ensure_started(&mut self) -> Result<(), CameraError> {
        if self.receiver.is_some() {
            return Ok(());
        }

        let (tx, rx) = mpsc::channel(self.config.buffer_count() as usize);

        let client = LibCamClient::new();

        // Configure lowres RGB stream
        let params = StreamParams {
            width: self.config.width(),
            height: self.config.height(),
            format: StreamFormat::STREAM_FORMAT_RGB,
            framerate: self.config.fps() as u8,
        };
        client.client.setupLowres(&params);

        // Register callback that bridges frames into the tokio channel
        let callback = Box::new(FrameCallback {
            tx,
            width: self.config.width() as usize,
            height: self.config.height() as usize,
        });
        client.setCallbacks(callback);

        // Start capture in background thread
        client.start(true);

        self.receiver = Some(rx);
        self.client = Some(client);

        Ok(())
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &CameraConfig {
        &self.config
    }
}
