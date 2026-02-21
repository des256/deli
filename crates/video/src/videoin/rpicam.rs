use {
    crate::*,
    base::Vec2,
    image::{Image, PixelFormat as ImagePixelFormat},
    shiguredo_libcamera::{
        Camera, CameraManager, ConfigStatus, FrameBuffer, FrameBufferAllocator, FrameStatus,
        PixelFormat, Request, Size, Stream, StreamRole,
    },
    std::sync::{Arc, mpsc},
};

#[derive(Debug, Clone, Default)]
pub struct RpiCamConfig {
    pub index: Option<usize>,
    pub size: Option<Vec2<usize>>,
    pub format: Option<ImagePixelFormat>,
    pub frame_rate: Option<f32>,
}

/// Memory-mapped DMA-buf region for zero-copy access to camera frame data.
struct MmapBuffer {
    base: *mut libc::c_void,
    base_len: usize,
    data_offset: usize,
    data_len: usize,
}

unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    fn data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self.base as *const u8).add(self.data_offset),
                self.data_len,
            )
        }
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base, self.base_len);
        }
    }
}

struct PendingFrame {
    cookie: usize,
    frame: Option<VideoFrame>,
}

pub(crate) struct RpiCamera {
    manager: Option<CameraManager>,
    camera: Option<Camera>,
    allocator: Option<FrameBufferAllocator>,
    buffers: Vec<FrameBuffer>,
    requests: Vec<Request>,
    mmaps: Arc<Vec<MmapBuffer>>,
    stream: Option<Stream>,
    frame_rx: Option<mpsc::Receiver<PendingFrame>>,
}

impl RpiCamera {
    pub fn new() -> Self {
        Self {
            manager: None,
            camera: None,
            allocator: None,
            buffers: Vec::new(),
            requests: Vec::new(),
            mmaps: Arc::new(Vec::new()),
            stream: None,
            frame_rx: None,
        }
    }
}

impl VideoInDevice for RpiCamera {
    fn open(&mut self, config: &VideoInConfig) -> Result<VideoInConfig, VideoError> {
        self.close();

        #[allow(irrefutable_let_patterns)]
        let config = if let VideoInConfig::RpiCam(config) = config {
            config
        } else {
            return Err(VideoError::Device(
                "RpiCamera::open requires VideoInConfig::RpiCam".to_string(),
            ));
        };

        // Create camera manager and acquire camera
        let manager = CameraManager::new()?;
        if manager.cameras_count() == 0 {
            return Err(VideoError::Device("No cameras found".to_string()));
        }
        let index = config.index.unwrap_or(0);
        let mut camera = manager.get_camera(index)?;
        camera.acquire()?;

        // Generate and apply configuration
        let mut cam_config = camera.generate_configuration(&[StreamRole::VideoRecording])?;
        {
            let mut sc = cam_config.at(0)?;
            if let Some(format) = &config.format {
                let fourcc = format.as_fourcc();
                sc.set_pixel_format(PixelFormat::from_fourcc(fourcc));
            }
            if let Some(size) = &config.size {
                sc.set_size(Size::new(size.x as u32, size.y as u32));
            }
        }

        let validate_status = cam_config.validate()?;
        match validate_status {
            ConfigStatus::Invalid => {
                return Err(VideoError::Device(
                    "Invalid camera configuration".to_string(),
                ));
            }
            _ => {}
        }
        camera.configure(&mut cam_config)?;

        // Read back the actual (possibly adjusted) configuration
        let (actual_format, actual_size, stream);
        {
            let sc = cam_config.at(0)?;
            let pf = sc.pixel_format();
            actual_format = ImagePixelFormat::from_fourcc(pf.fourcc)?;
            let sz = sc.size();
            actual_size = Vec2::new(sz.width as usize, sz.height as usize);
            stream = sc
                .stream()
                .ok_or(VideoError::Device("No stream available".to_string()))?;
        }

        let actual_frame_rate = config.frame_rate.unwrap_or(30.0);

        // Allocate buffers
        let allocator = FrameBufferAllocator::new(&camera);
        let buffer_count = allocator.allocate(&stream)?;

        // Get buffer handles and mmap each buffer covering all planes
        let mut buffers = Vec::with_capacity(buffer_count);
        let mut mmap_buffers = Vec::with_capacity(buffer_count);

        for i in 0..buffer_count {
            let buffer = allocator.get_buffer(&stream, i)?;
            let plane0 = buffer
                .plane(0)
                .ok_or(VideoError::Device(format!("Buffer {i} has no planes")))?;

            // Find the total extent across all planes (they share the same fd)
            let mut mmap_len = (plane0.offset as usize) + (plane0.length as usize);
            for p in 1..buffer.planes_count() {
                if let Some(plane) = buffer.plane(p) {
                    let end = (plane.offset as usize) + (plane.length as usize);
                    if end > mmap_len {
                        mmap_len = end;
                    }
                }
            }

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    mmap_len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    plane0.fd,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(VideoError::Device(format!(
                    "Failed to mmap buffer {i}: {}",
                    std::io::Error::last_os_error(),
                )));
            }

            mmap_buffers.push(MmapBuffer {
                base: ptr,
                base_len: mmap_len,
                data_offset: 0,
                data_len: mmap_len,
            });
            buffers.push(buffer);
        }

        let mmaps = Arc::new(mmap_buffers);

        // Create requests, each bound to one buffer
        let mut requests = Vec::with_capacity(buffer_count);
        for i in 0..buffer_count {
            let request = camera.create_request(i as u64)?;
            request.add_buffer(&stream, &buffers[i])?;
            requests.push(request);
        }

        // Set up frame delivery: callback copies data from mmap and sends through channel
        let (frame_tx, frame_rx) = mpsc::channel::<PendingFrame>();
        let mmaps_cb = Arc::clone(&mmaps);
        let stream_cb = stream.clone();
        let cb_size = actual_size;
        let cb_format = actual_format;

        camera.on_request_completed(move |completed| {
            let cookie = completed.cookie() as usize;

            let frame = (|| -> Option<VideoFrame> {
                let buffer = completed.find_buffer(&stream_cb)?;
                if buffer.metadata().status != FrameStatus::Success {
                    return None;
                }
                if cookie >= mmaps_cb.len() {
                    return None;
                }

                let raw_data = mmaps_cb[cookie].data();

                let data = match cb_format {
                    ImagePixelFormat::Yuyv => {
                        let expected = cb_size.x * cb_size.y * 2;
                        raw_data[..expected.min(raw_data.len())].to_vec()
                    }
                    ImagePixelFormat::Jpeg => raw_data.to_vec(),
                    ImagePixelFormat::Srggb10p => raw_data.to_vec(),
                    ImagePixelFormat::Yu12 => {
                        let expected = cb_size.x * cb_size.y * 3 / 2;
                        raw_data[..expected.min(raw_data.len())].to_vec()
                    }
                    _ => return None,
                };

                Some(VideoFrame {
                    image: Image::new(cb_size, data, cb_format),
                })
            })();

            // Always send so the request gets re-queued in blocking_capture
            let _ = frame_tx.send(PendingFrame { cookie, frame });
        });

        // Start capturing and queue all requests
        camera.start()?;
        for request in &requests {
            camera.queue_request(request)?;
        }

        // Store state
        self.manager = Some(manager);
        self.camera = Some(camera);
        self.allocator = Some(allocator);
        self.buffers = buffers;
        self.requests = requests;
        self.mmaps = mmaps;
        self.stream = Some(stream);
        self.frame_rx = Some(frame_rx);

        Ok(VideoInConfig::RpiCam(RpiCamConfig {
            index: Some(index),
            size: Some(actual_size),
            format: Some(actual_format),
            frame_rate: Some(actual_frame_rate),
        }))
    }

    fn close(&mut self) {
        // Stop camera and release it (disconnects callback, drops frame_tx + mmaps Arc clone)
        if let Some(camera) = self.camera.take() {
            let _ = camera.stop();
            let _ = camera.release();
        }

        self.frame_rx.take();
        self.requests.clear();
        self.mmaps = Arc::new(Vec::new());
        self.buffers.clear();
        self.stream.take();
        self.allocator.take();
        self.manager.take();
    }

    fn blocking_capture(&mut self) -> Result<VideoFrame, VideoError> {
        let frame_rx = self
            .frame_rx
            .as_ref()
            .ok_or(VideoError::Stream("Camera not open".to_string()))?;

        loop {
            let pending = match frame_rx.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(pending) => pending,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    return Err(VideoError::Stream("Capture timeout".to_string()));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(VideoError::Stream("Frame channel closed".to_string()));
                }
            };

            // Reuse the completed request and re-queue it for the next capture.
            // Don't re-add the buffer — reuse() preserves buffer associations.
            if let Some(camera) = self.camera.as_ref() {
                if pending.cookie < self.requests.len() {
                    let request = &self.requests[pending.cookie];
                    request.reuse();
                    camera.queue_request(request).map_err(|e| {
                        VideoError::Stream(format!("Failed to re-queue request: {e}"))
                    })?;
                }
            }

            // Skip startup/failed frames, loop to wait for the next one
            if let Some(frame) = pending.frame {
                return Ok(frame);
            }
        }
    }
}
