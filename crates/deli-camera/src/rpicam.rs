use crate::convert::yuyv_to_rgb;
use crate::{Camera, CameraConfig, CameraError};
use deli_base::Tensor;
use deli_image::DecodedImage;
use std::thread::{self, JoinHandle};
use tokio::sync::mpsc;

type FrameResult = Result<Tensor<u8>, CameraError>;

/// Pixel format negotiated during camera initialization.
#[derive(Debug, Clone, Copy)]
enum CaptureFormat {
    Mjpeg,
    Yuyv { width: u32, height: u32 },
}

/// RPi Camera implementation using libcamera.
///
/// **Note:** The `config.device()` field is ignored. RPiCamera always uses the first
/// available camera (index 0). libcamera identifies cameras by string ID, not device paths.
///
/// **Note:** The camera is released after format negotiation in `new()` and re-acquired
/// when capture starts on the first `recv()` call. If another process acquires the camera
/// between construction and first `recv()`, the capture thread will fail with a device error.
pub struct RPiCamera {
    config: CameraConfig,
    format: CaptureFormat,
    receiver: Option<mpsc::Receiver<FrameResult>>,
    thread_handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for RPiCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RPiCamera")
            .field("config", &self.config)
            .field("format", &self.format)
            .field("receiver", &self.receiver.is_some())
            .field("thread_handle", &self.thread_handle.is_some())
            .finish()
    }
}

impl Camera for RPiCamera {
    async fn recv(&mut self) -> Result<Tensor<u8>, CameraError> {
        // Ensure capture thread is running
        self.ensure_started()?;

        // Receive next frame from channel
        let receiver = self
            .receiver
            .as_mut()
            .ok_or_else(|| CameraError::Channel("Receiver not initialized".to_string()))?;

        receiver.recv().await.ok_or_else(|| {
            CameraError::Stream(
                "Capture thread terminated; recreate RPiCamera to restart".to_string(),
            )
        })?
    }
}

impl Drop for RPiCamera {
    fn drop(&mut self) {
        // Drop the receiver to signal the thread to stop
        drop(self.receiver.take());

        // Wait for the thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl RPiCamera {
    /// Create a new RPi Camera with the given configuration.
    ///
    /// Negotiates pixel format: tries MJPEG first, falls back to YUYV if MJPEG is not supported.
    ///
    /// **Note:** The `config.device()` field is ignored. This camera always uses the first
    /// available camera (index 0) from libcamera. If the device field is set to a non-default
    /// value, a warning is logged.
    ///
    /// # Errors
    ///
    /// Returns `CameraError::Device` if:
    /// - No cameras are detected
    /// - Camera acquisition fails
    /// - Neither MJPEG nor YUYV formats are supported
    /// - Camera configuration fails
    pub fn new(config: CameraConfig) -> Result<Self, CameraError> {
        use libcamera::{
            camera_manager::CameraManager,
            pixel_format::PixelFormat,
            stream::StreamRole,
        };

        // Warn if device field is not default (it's ignored for libcamera)
        if config.device() != "/dev/video0" {
            log::warn!(
                "RPiCamera ignores config.device() (got: {}). Always uses first camera (index 0).",
                config.device()
            );
        }

        // Create camera manager and get first camera
        let mgr = CameraManager::new()?;
        let cameras = mgr.cameras();
        let cam = cameras
            .get(0)
            .ok_or_else(|| CameraError::Device("No cameras detected".to_string()))?;

        let mut cam = cam.acquire()?;

        // Try MJPEG first
        let mjpeg_format = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);
        let mut cfgs = cam
            .generate_configuration(&[StreamRole::ViewFinder])
            .ok_or_else(|| CameraError::Device("Failed to generate configuration".to_string()))?;

        let no_stream_cfg = || CameraError::Device("No stream configuration available".to_string());

        cfgs.get_mut(0).ok_or_else(no_stream_cfg)?.set_pixel_format(mjpeg_format);
        cfgs.get_mut(0).ok_or_else(no_stream_cfg)?.set_size(config.width(), config.height());

        let status = cfgs.validate();
        let actual_format = cfgs.get(0).ok_or_else(no_stream_cfg)?.get_pixel_format();

        let format = if actual_format == mjpeg_format && !status.is_invalid() {
            // MJPEG is supported
            CaptureFormat::Mjpeg
        } else {
            // MJPEG not supported, try YUYV
            let yuyv_format = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);
            cfgs.get_mut(0).ok_or_else(no_stream_cfg)?.set_pixel_format(yuyv_format);
            cfgs.get_mut(0).ok_or_else(no_stream_cfg)?.set_size(config.width(), config.height());

            let status = cfgs.validate();
            let actual_format = cfgs.get(0).ok_or_else(no_stream_cfg)?.get_pixel_format();

            if actual_format == yuyv_format && !status.is_invalid() {
                // Use validated size — camera may adjust resolution
                let actual_size = cfgs.get(0).ok_or_else(no_stream_cfg)?.get_size();
                CaptureFormat::Yuyv {
                    width: actual_size.width,
                    height: actual_size.height,
                }
            } else {
                return Err(CameraError::Device(
                    "no supported pixel format (tried MJPEG and YUYV)".to_string(),
                ));
            }
        };

        // Format negotiation successful. Camera manager and active camera are dropped here;
        // capture_loop() will re-acquire on first recv() call (lazy initialization).
        Ok(Self {
            config,
            format,
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

        let config = self.config.clone();
        let format = self.format;
        let buffer_count = self.config.buffer_count() as usize;
        let (tx, rx) = mpsc::channel(buffer_count);

        // Spawn capture thread
        let handle = thread::spawn(move || {
            Self::capture_loop(config, format, tx, buffer_count);
        });

        self.receiver = Some(rx);
        self.thread_handle = Some(handle);

        Ok(())
    }

    /// Background thread capture loop.
    ///
    /// Sets up libcamera, captures frames, decodes based on format (MJPEG or YUYV),
    /// and sends Tensor<u8> through the channel. Uses `try_send` to drop frames when
    /// the channel is full rather than blocking.
    fn capture_loop(
        config: CameraConfig,
        format: CaptureFormat,
        tx: mpsc::Sender<FrameResult>,
        buffer_count: usize,
    ) {
        use libcamera::{
            camera_manager::CameraManager,
            framebuffer_allocator::FrameBufferAllocator,
            framebuffer_map::MemoryMappedFrameBuffer,
            pixel_format::PixelFormat,
            stream::StreamRole,
        };

        // Create camera manager and acquire first camera
        let mgr = match CameraManager::new() {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
                return;
            }
        };

        let cameras = mgr.cameras();
        let cam = match cameras.get(0) {
            Some(c) => c,
            None => {
                let _ = tx.blocking_send(Err(CameraError::Device("No cameras detected".to_string())));
                return;
            }
        };

        let mut cam = match cam.acquire() {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
                return;
            }
        };

        // Configure stream with negotiated format
        let mut cfgs = match cam.generate_configuration(&[StreamRole::ViewFinder]) {
            Some(c) => c,
            None => {
                let _ = tx.blocking_send(Err(CameraError::Device("Failed to generate configuration".to_string())));
                return;
            }
        };

        let pixel_format = match format {
            CaptureFormat::Mjpeg => PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0),
            CaptureFormat::Yuyv { .. } => PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0),
        };

        match cfgs.get_mut(0) {
            Some(stream_cfg) => {
                stream_cfg.set_pixel_format(pixel_format);
                stream_cfg.set_size(config.width(), config.height());
            }
            None => {
                let _ = tx.blocking_send(Err(CameraError::Device("No stream configuration available".to_string())));
                return;
            }
        }

        if let Err(e) = cam.configure(&mut cfgs) {
            let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
            return;
        }

        let stream = match cfgs.get(0) {
            Some(cfg) => match cfg.stream() {
                Some(s) => s,
                None => {
                    let _ = tx.blocking_send(Err(CameraError::Device("No stream in configuration".to_string())));
                    return;
                }
            },
            None => {
                let _ = tx.blocking_send(Err(CameraError::Device("No stream configuration".to_string())));
                return;
            }
        };

        // Allocate frame buffers
        let mut alloc = FrameBufferAllocator::new(&cam);
        let buffers = match alloc.alloc(&stream) {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
                return;
            }
        };

        // Memory-map buffers
        let buffers: Vec<_> = buffers
            .into_iter()
            .map(|buf| MemoryMappedFrameBuffer::new(buf))
            .collect::<Result<Vec<_>, _>>();

        let buffers = match buffers {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
                return;
            }
        };

        // Create requests
        let mut reqs = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let mut req = match cam.create_request(None) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(Err(CameraError::Device(format!("Failed to create request: {e}"))));
                    return;
                }
            };
            if let Err(e) = req.add_buffer(&stream, buf) {
                let _ = tx.blocking_send(Err(CameraError::Device(format!("Failed to add buffer: {e}"))));
                return;
            }
            reqs.push(req);
        }

        // Set up request completion callback using std::sync::mpsc bridge
        let (callback_tx, callback_rx) = std::sync::mpsc::channel();
        cam.on_request_completed(move |req| {
            let _ = callback_tx.send(req);
        });

        // Start camera and queue all requests
        if let Err(e) = cam.start(None) {
            let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
            return;
        }

        for req in &mut reqs {
            if let Err((_, e)) = cam.queue_request(req) {
                let _ = tx.blocking_send(Err(CameraError::Device(e.to_string())));
                return;
            }
        }

        // Main capture loop
        loop {
            // Receive completed request from callback
            let mut req = match callback_rx.recv() {
                Ok(r) => r,
                Err(_) => break, // Callback sender dropped, exit
            };

            // Get framebuffer for our stream
            let framebuffer = match req.buffer(&stream) {
                Some(fb) => fb,
                None => {
                    let _ = tx.try_send(Err(CameraError::Stream("No framebuffer in request".to_string())));
                    continue;
                }
            };

            // Read frame data - MUST copy before reusing request
            let planes = framebuffer.data();
            let frame_data = match planes.first() {
                Some(plane) => {
                    let metadata = match framebuffer.metadata() {
                        Some(m) => m,
                        None => {
                            let _ = tx.try_send(Err(CameraError::Stream("No metadata in framebuffer".to_string())));
                            continue;
                        }
                    };
                    let bytes_used = match metadata.planes().first() {
                        Some(p) => p.bytes_used as usize,
                        None => {
                            let _ = tx.try_send(Err(CameraError::Stream("No plane metadata".to_string())));
                            continue;
                        }
                    };
                    plane[..bytes_used].to_vec() // Copy before reuse
                }
                None => {
                    let _ = tx.try_send(Err(CameraError::Stream("No data plane in framebuffer".to_string())));
                    continue;
                }
            };

            // Decode based on format
            let tensor_result = match format {
                CaptureFormat::Mjpeg => {
                    // Decode MJPEG via deli_image
                    match deli_image::decode_image(&frame_data) {
                        Ok(DecodedImage::U8(t)) => Ok(t),
                        Ok(_) => Err(CameraError::Decode(deli_image::ImageError::Decode(
                            "Unexpected pixel format (expected U8)".to_string(),
                        ))),
                        Err(e) => Err(CameraError::Decode(e)),
                    }
                }
                CaptureFormat::Yuyv { width, height } => {
                    // Convert YUYV to RGB
                    let rgb_data = match yuyv_to_rgb(&frame_data, width, height) {
                        Some(data) => data,
                        None => {
                            let _ = tx.try_send(Err(CameraError::Stream(format!(
                                "YUYV frame too short: got {} bytes, expected {} for {}x{}",
                                frame_data.len(), (width as usize) * (height as usize) * 2, width, height
                            ))));
                            continue;
                        }
                    };
                    Tensor::new(vec![height as usize, width as usize, 3], rgb_data)
                        .map_err(|e| CameraError::Stream(e.to_string()))
                }
            };

            // Send tensor through channel (drop if full)
            match tx.try_send(tensor_result) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Consumer too slow — drop frame silently
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    // Receiver dropped — exit thread
                    break;
                }
            }

            // Reuse request and re-queue
            req.reuse(libcamera::request::ReuseFlag::ReuseBuffers);
            if let Err((_, e)) = cam.queue_request(&mut req) {
                let _ = tx.try_send(Err(CameraError::Device(e.to_string())));
                break;
            }
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &CameraConfig {
        &self.config
    }
}
