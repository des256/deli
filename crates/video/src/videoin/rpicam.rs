use {
    crate::{VideoCapture, VideoData, VideoError, VideoFrame},
    libcamera_rs::{Camera, CameraManager, FrameBufferAllocator, StreamRole},
};

pub(crate) struct RpiCamera {
    camera: Camera,
}

impl RpiCamera {
    pub fn open(
        index: usize,
        width: usize,
        height: usize,
        format: VideoFormat,
        fps: f32,
    ) -> Result<Self, VideoError> {
        let manager = CameraManager::new()?;
        let mut camera = manager.get_camera(index)?;
        camera.acquire()?;
        let mut config = camera.generate_configuration(&[StreamRole::VideoRecording])?;
        config.set_pixel_format(match format {});
        config.set_resolution(width, height);
        config.set_framerate(fps as u32);
        config.validate()?;
        camera.configure(&config)?;
        let config = config.at(0)?;
        let stream = config.stream()?;
        let allocator = FrameBufferAllocator::new(&camera);
        allocator.allocate(&stream)?;
        camera.on_request_completed(move |completed| {
            if let Some(buffer) = completed.find_buffer(&stream) {
                let meta = buffer.metadata();
                // TODO: send frame to channel
            }
        });
        camera.start()?;
        Ok(Self { camera })
    }
}

impl VideoCapture for RpiCamera {
    fn capture(&mut self) -> Result<VideoFrame, VideoError> {
        // read frame
        // return
        Ok(VideoFrame {
            data: VideoData::Jpeg(vec![0; 1]),
            width: 1,
            height: 1,
        })
    }
}
