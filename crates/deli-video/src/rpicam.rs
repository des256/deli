use crate::{CameraConfig, CameraError, VideoFrame};
use deli_base::Tensor;
use futures_core::Stream;
use rslibcamlitelib::{ExternalCallback, LibCamClient, StreamFormat, StreamParams};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

type VideoFrameResult = Result<VideoFrame, CameraError>;

/// Callback handler that bridges rslibcamlite frames to a tokio channel.
struct VideoFrameCallback {
    tx: mpsc::Sender<VideoFrameResult>,
    width: usize,
    height: usize,
}

impl ExternalCallback for VideoFrameCallback {
    unsafe fn callbackLowres(&mut self, bytes: *mut u8, count: usize) {
        let slice = unsafe { std::slice::from_raw_parts(bytes, count) };
        let data = slice.to_vec();

        let frame = Tensor::new(vec![self.height, self.width, 3], data)
            .map(VideoFrame::Rgb)
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
/// Implements `Stream<Item = Result<VideoFrame, CameraError>>` for async frame capture.
/// Captures RGB frames from the Raspberry Pi camera using the lowres stream.
///
/// **Note:** The `config.device()` field is ignored. rslibcamlite always uses
/// the system's default camera.
///
/// **Note:** Camera setup is deferred until the first poll.
pub struct RPiCamera {
    config: CameraConfig,
    receiver: Option<mpsc::Receiver<VideoFrameResult>>,
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

impl Stream for RPiCamera {
    type Item = Result<VideoFrame, CameraError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Lazy start: begin capture on first poll
        if this.receiver.is_none() {
            if let Err(e) = this.ensure_started() {
                return Poll::Ready(Some(Err(e)));
            }
        }

        match this.receiver.as_mut().unwrap().poll_recv(cx) {
            Poll::Ready(Some(result)) => Poll::Ready(Some(result)),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
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
    /// The camera is not opened until the first poll.
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
        let callback = Box::new(VideoFrameCallback {
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
