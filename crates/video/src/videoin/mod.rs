use {crate::*, base::Vec2, std::path::PathBuf, tokio::sync::mpsc};

// capacity of the video input channel
const CHANNEL_CAPACITY: usize = 4;

#[derive(Debug, Clone)]
pub enum VideoInDevice {
    #[cfg(feature = "v4l2")]
    V4l2(Option<PathBuf>),
    #[cfg(feature = "rpicam")]
    RpiCam(Option<usize>),
    #[cfg(feature = "realsense")]
    RealSense(Option<usize>),
}

#[derive(Debug, Clone)]
pub struct VideoInConfig {
    pub device: Option<VideoInDevice>,
    pub size: Vec2<usize>,
    pub format: VideoFormat,
    pub frame_rate: f32,
}

impl Default for VideoInConfig {
    fn default() -> Self {
        Self {
            device: None,
            size: Vec2::new(0, 0),
            format: VideoFormat::Yuyv,
            frame_rate: 0.0,
        }
    }
}

pub struct VideoIn {
    receiver: mpsc::Receiver<VideoFrame>,
    new_config_sender: mpsc::Sender<VideoInConfig>,
    current_config: VideoInConfig,
}

impl VideoIn {
    pub async fn open() -> Self {
        // channel for receiving video frames
        let (sender, receiver) = mpsc::channel::<VideoFrame>(4);

        // channel for sending new video configurations
        let (new_config_sender, mut new_config_receiver) =
            mpsc::channel::<VideoInConfig>(CHANNEL_CAPACITY);

        // current video configuration
        let current_config = VideoInConfig::default();

        // spawn separate task for video capture loop
        tokio::task::spawn_blocking(move || {
            // wait for new video configuration
            while let Some(config) = new_config_receiver.blocking_recv() {
                // TODO: prepare video capture configuration

                // reconnect loop
                while new_config_receiver.len() == 0 {
                    // decode default device behavior
                    let config_device = match config.device {
                        Some(ref device) => device.clone(),

                        // if rpicam is enabled, use that (we're on a raspberry pi)
                        #[cfg(feature = "rpicam")]
                        None => VideoInDevice::RpiCam(None),

                        // otherwise V4L2 is the default
                        #[cfg(all(not(feature = "rpicam"), feature = "v4l2"))]
                        None => VideoInDevice::V4l2(None),

                        // and if neither of those are enabled, check for RealSense
                        #[cfg(all(
                            not(feature = "rpicam"),
                            not(feature = "v4l2"),
                            feature = "realsense"
                        ))]
                        None => VideoInDevice::RealSense(None),
                    };

                    // start video capture
                    match config_device {
                        #[cfg(feature = "rpicam")]
                        VideoInDevice::RpiCam(ref index) => {
                            let camera = match RpiCamera::open() {
                                Ok(camera) => camera,
                                Err(error) => {
                                    log::error!("Failed to open RPi camera: {}", error);
                                    continue;
                                }
                            };
                            // TODO: wait until new configuration appears, or until some error occurs
                        }
                        #[cfg(feature = "v4l2")]
                        VideoInDevice::V4l2(ref path) => {
                            let camera = match v4l2::V4l2Camera::open(
                                path.clone(),
                                config.size.x,
                                config.size.y,
                                config.format.clone(),
                                config.frame_rate,
                            ) {
                                Ok(camera) => Box::new(camera),
                                Err(error) => {
                                    log::error!("Failed to open V4L2 camera: {}", error);
                                    continue;
                                }
                            };
                            while new_config_receiver.len() == 0 {
                                // TODO: read video frame
                                // TODO: send video frame through channel
                            }
                        }
                        #[cfg(feature = "realsense")]
                        VideoInDevice::RealSense(ref index) => {
                            match RealsenseCamera::open(index.clone()) {
                                Ok(camera) => Box::new(camera),
                                Err(error) => {
                                    log::error!("Failed to open RealSense camera: {}", error);
                                    continue;
                                }
                            }
                            while new_config_receiver.len() == 0 {
                                // TODO: read video frame
                                // TODO: send video frame through channel
                            }
                        }
                    };
                }
            }
        });

        // start default configuration
        if let Err(error) = new_config_sender.send(current_config.clone()).await {
            log::error!("Failed to send default video config: {}", error);
        }

        Self {
            receiver,
            new_config_sender,
            current_config,
        }
    }

    pub fn config(&self) -> VideoInConfig {
        self.current_config.clone()
    }

    pub async fn select(&mut self, config: VideoInConfig) {
        if let Err(error) = self.new_config_sender.send(config.clone()).await {
            log::error!("Failed to send new video config: {}", error);
        }
        self.current_config = config;
    }

    pub async fn capture(&mut self) -> Result<VideoFrame, VideoError> {
        match self.receiver.recv().await {
            Some(frame) => Ok(frame),
            None => Err(VideoError::Stream("Video input channel closed".to_string())),
        }
    }

    pub async fn list_devices() -> Result<Vec<VideoInDevice>, VideoError> {
        // TODO: implement video input device listing
        Ok(vec![])
    }
}

#[cfg(feature = "rpicam")]
mod rpicam;

#[cfg(feature = "v4l2")]
mod v4l2;

#[cfg(feature = "realsense")]
mod realsense;
