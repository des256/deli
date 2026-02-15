/// Configuration for camera capture.
#[derive(Clone, Debug)]
pub struct CameraConfig {
    device: String,
    width: u32,
    height: u32,
    fps: u32,
    buffer_count: u32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device: "/dev/video0".to_string(),
            width: 640,
            height: 480,
            fps: 30,
            buffer_count: 4,
        }
    }
}

impl CameraConfig {
    /// Set the device path (e.g., "/dev/video0").
    pub fn with_device(mut self, device: String) -> Self {
        self.device = device;
        self
    }

    /// Set the capture width in pixels.
    pub fn with_width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set the capture height in pixels.
    pub fn with_height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set the frames per second.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Set the buffer count for the capture stream.
    pub fn with_buffer_count(mut self, buffer_count: u32) -> Self {
        self.buffer_count = buffer_count;
        self
    }

    // Getters
    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn fps(&self) -> u32 {
        self.fps
    }

    pub fn buffer_count(&self) -> u32 {
        self.buffer_count
    }
}
