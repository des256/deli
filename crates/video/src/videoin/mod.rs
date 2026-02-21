use {
    crate::*,
    base::Vec2,
    image::PixelFormat,
    std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    tokio::{
        sync::mpsc,
        task::{JoinHandle, spawn_blocking},
    },
};

// capacity of the video input channel
const CHANNEL_CAPACITY: usize = 4;

// delay before reconnecting after failure
const WAIT_BEFORE_RECONNECT_MS: u64 = 100;

#[derive(Debug, Clone)]
pub enum VideoInConfig {
    #[cfg(feature = "v4l2")]
    V4l2(v4l2::V4l2Config),
    #[cfg(feature = "rpicam")]
    RpiCam(rpicam::RpiCamConfig),
    #[cfg(feature = "realsense")]
    Realsense(realsense::RealsenseConfig),
}

pub(crate) trait VideoInDevice: Send {
    fn open(&mut self, config: &VideoInConfig) -> Result<VideoInConfig, VideoError>; // open the device, return config that was actually set
    fn close(&mut self); // close the device, if open
    fn blocking_capture(&mut self) -> Result<VideoFrame, VideoError>; // capture a frame
}

pub struct VideoIn {
    sender: mpsc::Sender<VideoFrame>,
    receiver: mpsc::Receiver<VideoFrame>,
    cancel: Arc<AtomicBool>,
    size: Vec2<usize>,
    format: PixelFormat,
    frame_rate: f32,
    join_handle: Option<JoinHandle<()>>,
}

impl VideoIn {
    fn create_device(config: &VideoInConfig) -> Result<Box<dyn VideoInDevice>, VideoError> {
        match config {
            #[cfg(feature = "v4l2")]
            VideoInConfig::V4l2(_) => Ok(Box::new(v4l2::V4l2::new())),
            #[cfg(feature = "rpicam")]
            VideoInConfig::RpiCam(_) => Ok(Box::new(rpicam::RpiCamera::new())),
            #[cfg(feature = "realsense")]
            VideoInConfig::Realsense(_) => Ok(Box::new(realsense::Realsense::new())),
        }
    }

    async fn spawn_worker(
        sender: mpsc::Sender<VideoFrame>,
        config: VideoInConfig,
        cancel: Arc<AtomicBool>,
    ) -> Result<(JoinHandle<()>, VideoInConfig), VideoError> {
        let mut device = Self::create_device(&config)?;

        // Send the resolved config back from the worker thread via oneshot channel.
        // device.open() must run on the same OS thread as blocking_capture() because
        // some backends (e.g. librealsense2) have thread affinity requirements.
        let (init_tx, init_rx) = tokio::sync::oneshot::channel::<Result<VideoInConfig, VideoError>>();

        let join_handle = spawn_blocking({
            move || {
                // Open device on the worker thread
                let mut config = match device.open(&config) {
                    Ok(config) => {
                        let _ = init_tx.send(Ok(config.clone()));
                        config
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };

                while !cancel.load(Ordering::Relaxed) {
                    // keep pumping frames until capturing fails
                    log::info!("video worker: starting capture loop");
                    loop {
                        match device.blocking_capture() {
                            Ok(frame) => {
                                if let Err(error) = sender.blocking_send(frame) {
                                    log::error!("Failed to send video frame: {}", error);
                                    return; // main closed the channel, so drop everything
                                }
                            }
                            Err(e) => {
                                log::error!("video worker: capture failed: {}", e);
                                break;
                            }
                        }
                    }

                    // close, wait, and reopen the device
                    while !cancel.load(Ordering::Relaxed) {
                        log::info!("video worker: reconnecting...");
                        device.close();
                        std::thread::sleep(std::time::Duration::from_millis(
                            WAIT_BEFORE_RECONNECT_MS,
                        ));
                        if let Ok(new_config) = device.open(&config) {
                            config = new_config;
                            break;
                        }
                        // if device.open returns error, just stay in the loop
                    }
                }
                device.close();
            }
        });

        let config = init_rx
            .await
            .map_err(|_| VideoError::Device("Worker thread died during init".to_string()))??;

        Ok((join_handle, config))
    }

    fn decode_config(config: VideoInConfig) -> (Vec2<usize>, PixelFormat, f32) {
        match config {
            #[cfg(feature = "v4l2")]
            VideoInConfig::V4l2(config) => (
                config.size.unwrap(),
                config.format.unwrap(),
                config.frame_rate.unwrap(),
            ),
            #[cfg(feature = "rpicam")]
            VideoInConfig::RpiCam(config) => (
                config.size.unwrap(),
                config.format.unwrap(),
                config.frame_rate.unwrap(),
            ),
            #[cfg(feature = "realsense")]
            VideoInConfig::Realsense(config) => (
                config.color.unwrap(),
                PixelFormat::Rgb8,
                config.frame_rate.unwrap(),
            ),
        }
    }

    pub async fn open(config: Option<VideoInConfig>) -> Result<Self, VideoError> {
        // if no config is provided, use a platform default
        #[cfg(feature = "rpicam")]
        let config = config.unwrap_or(VideoInConfig::RpiCam(rpicam::RpiCamConfig {
            index: None,
            size: None,
            format: None,
            frame_rate: None,
        }));
        #[cfg(all(feature = "v4l2", not(feature = "rpicam")))]
        let config = config.unwrap_or(VideoInConfig::V4l2(v4l2::V4l2Config {
            path: None,
            size: None,
            format: None,
            frame_rate: None,
        }));
        #[cfg(all(feature = "realsense", not(feature = "rpicam"), not(feature = "v4l2")))]
        let config = config.unwrap_or(VideoInConfig::Realsense(realsense::RealsenseConfig {
            index: None,
            color: None,
            depth: None,
            ir: None,
            frame_rate: None,
        }));

        // channel for receiving video frames
        let (sender, receiver) = mpsc::channel::<VideoFrame>(CHANNEL_CAPACITY);

        // external cancelation flag
        let cancel = Arc::new(AtomicBool::new(false));

        // spawn the worker
        let (join_handle, config) =
            Self::spawn_worker(sender.clone(), config.clone(), Arc::clone(&cancel)).await?;

        let (size, format, frame_rate) = Self::decode_config(config);

        Ok(Self {
            sender,
            receiver,
            cancel,
            size,
            format,
            frame_rate,
            join_handle: Some(join_handle),
        })
    }

    pub fn size(&self) -> Vec2<usize> {
        self.size
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn frame_rate(&self) -> f32 {
        self.frame_rate
    }

    pub async fn select(&mut self, config: VideoInConfig) -> Result<(), VideoError> {
        // cancel the current worker
        self.cancel.store(true, Ordering::Relaxed);
        self.join_handle.take().unwrap().await.unwrap();

        // reset cancel flag
        self.cancel.store(false, Ordering::Relaxed);

        // spawn a new worker
        let (join_handle, config) = Self::spawn_worker(
            self.sender.clone(),
            config.clone(),
            Arc::clone(&self.cancel),
        )
        .await?;
        self.join_handle = Some(join_handle);
        let (size, format, frame_rate) = Self::decode_config(config);
        self.size = size;
        self.format = format;
        self.frame_rate = frame_rate;
        Ok(())
    }

    pub async fn capture(&mut self) -> Result<VideoFrame, VideoError> {
        match self.receiver.recv().await {
            Some(frame) => Ok(frame),
            None => Err(VideoError::Stream("Video input channel closed".to_string())),
        }
    }
}

impl Drop for VideoIn {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.join_handle.take().unwrap().abort();
    }
}

#[cfg(feature = "rpicam")]
pub mod rpicam;

#[cfg(feature = "v4l2")]
pub mod v4l2;

#[cfg(feature = "realsense")]
pub mod realsense;
