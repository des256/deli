use {
    crate::*,
    base::Vec2,
    image::{Image, PixelFormat},
};

pub(crate) struct Realsense {}

impl Realsense {
    pub fn new() -> Self {
        Self {}
    }
}

impl VideoInDevice for Realsense {
    fn open(&mut self, _config: &VideoInConfig) -> Result<VideoInConfig, VideoError> {
        todo!("RealSense camera support not implemented")
    }

    fn close(&mut self) {
        // No-op stub
    }

    fn blocking_capture(&mut self) -> Result<VideoFrame, VideoError> {
        // Return a minimal placeholder frame
        let image = Image::new(Vec2::new(1, 1), vec![0; 1], PixelFormat::Jpeg);
        Ok(VideoFrame { image })
    }
}
