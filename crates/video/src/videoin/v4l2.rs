use {
    crate::*,
    base::*,
    image::*,
    std::path::PathBuf,
    tokio::sync::mpsc as tokio_mpsc,
    v4l::{
        Device, Format, FourCC, buffer::Type, io::mmap::Stream as MmapStream,
        io::traits::CaptureStream, video::Capture,
    },
};

const CHANNEL_CAPACITY: usize = 4;

#[derive(Debug, Clone, Default)]
pub struct V4l2Config {
    pub path: Option<PathBuf>,
    pub size: Option<Vec2<usize>>,
    pub format: Option<PixelFormat>,
    pub frame_rate: Option<f32>,
}

pub struct V4l2Listener {
    frame_rx: tokio_mpsc::Receiver<VideoFrame>,
    size: Vec2<usize>,
    format: PixelFormat,
    frame_rate: f32,
}

pub fn create(config: Option<V4l2Config>) -> Result<V4l2Listener, VideoError> {
    let (frame_tx, frame_rx) = tokio_mpsc::channel::<VideoFrame>(CHANNEL_CAPACITY);
    let config = config.unwrap_or_default();
    let device = match &config.path {
        Some(path) => Device::with_path(path)?,
        None => Device::new(0)?,
    };
    let device_format = Capture::format(&device)?;
    let desired_size = match &config.size {
        Some(size) => *size,
        None => Vec2::new(device_format.width as usize, device_format.height as usize),
    };
    let desired_fourcc = match &config.format {
        Some(format) => format.as_fourcc(),
        None => u32::from_le_bytes(device_format.fourcc.repr),
    };
    let actual_format = Capture::set_format(
        &device,
        &Format::new(
            desired_size.x as u32,
            desired_size.y as u32,
            FourCC::from(desired_fourcc),
        ),
    )?;
    let size = Vec2::new(actual_format.width as usize, actual_format.height as usize);
    let format = PixelFormat::from_fourcc(u32::from_le_bytes(actual_format.fourcc.repr))?;
    let desired_frame_rate = match config.frame_rate {
        Some(frame_rate) => frame_rate,
        None => {
            let params = Capture::params(&device)?;
            params.interval.denominator as f32 / params.interval.numerator as f32
        }
    };
    let actual_params = Capture::set_params(
        &device,
        &v4l::video::capture::Parameters::with_fps(desired_frame_rate as u32),
    )?;
    let frame_rate =
        actual_params.interval.denominator as f32 / actual_params.interval.numerator as f32;
    std::thread::spawn(move || {
        let mut stream = match MmapStream::with_buffers(&device, Type::VideoCapture, 4u32) {
            Ok(stream) => stream,
            Err(error) => {
                log_fatal!("Failed to create V4L2 stream: {}", error);
            }
        };
        while let Ok(frame_data) = CaptureStream::next(&mut stream) {
            let color = Image::new(size, frame_data.0.to_vec(), format);
            if let Err(error) = frame_tx.blocking_send(VideoFrame {
                color,
                depth: None,
                left: None,
                right: None,
            }) {
                log_error!("failed to send video frame: {}", error);
            }
        }
    });
    Ok(V4l2Listener {
        frame_rx,
        size,
        format,
        frame_rate,
    })
}

impl V4l2Listener {
    pub fn size(&self) -> Vec2<usize> {
        self.size
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn frame_rate(&self) -> f32 {
        self.frame_rate
    }

    pub async fn recv(&mut self) -> Option<VideoFrame> {
        self.frame_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<VideoFrame> {
        match self.frame_rx.try_recv() {
            Ok(frame) => Some(frame),
            _ => None,
        }
    }
}
