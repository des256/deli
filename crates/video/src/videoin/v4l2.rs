use {
    crate::*,
    base::Vec2,
    image::{Image, PixelFormat},
    std::path::PathBuf,
    v4l::{
        Device, Format, FourCC, buffer::Type, io::mmap::Stream as MmapStream,
        io::traits::CaptureStream, video::Capture,
    },
};

#[derive(Debug, Clone, Default)]
pub struct V4l2Config {
    pub path: Option<PathBuf>,
    pub size: Option<Vec2<usize>>,
    pub format: Option<PixelFormat>,
    pub frame_rate: Option<f32>,
}

pub(crate) struct V4l2 {
    stream: Option<MmapStream<'static>>,
    path: Option<PathBuf>,
    size: Vec2<usize>,
    format: PixelFormat,
    frame_rate: f32,
}

impl V4l2 {
    pub fn new() -> Self {
        Self {
            stream: None,
            path: None,
            size: Vec2::new(0, 0),
            format: PixelFormat::Yuyv,
            frame_rate: 0.0,
        }
    }
}

impl VideoInDevice for V4l2 {
    fn open(&mut self, config: &VideoInConfig) -> Result<VideoInConfig, VideoError> {
        // close stream
        self.stream.take();

        // unpack config
        #[allow(irrefutable_let_patterns)]
        let config = if let VideoInConfig::V4l2(config) = config {
            config
        } else {
            return Err(VideoError::Device(
                "V4l2::open should be called with VideoInConfig::V4l2".to_string(),
            ));
        };

        // create device from path or default
        let device = match &config.path {
            Some(path) => Device::with_path(path)?,
            None => Device::new(0)?,
        };
        self.path = config.path.clone();
        let device_format = Capture::format(&device)?;

        // build size
        let desired_size = match config.size {
            Some(size) => size,
            None => Vec2::new(device_format.width as usize, device_format.height as usize),
        };

        // build pixel format
        let desired_fourcc = match &config.format {
            Some(format) => match format {
                PixelFormat::Yuyv => FourCC::new(b"YUYV"),
                PixelFormat::Jpeg => FourCC::new(b"MJPG"),
                PixelFormat::Srggb10p => FourCC::new(b"pRAA"),
                PixelFormat::Yu12 => FourCC::new(b"YU12"),
                PixelFormat::Rgb8 | PixelFormat::Argb8 => {
                    return Err(VideoError::Device(
                        "RGB formats are not supported for V4L2 capture".to_string(),
                    ));
                }
            },
            None => device_format.fourcc,
        };

        // set the format and get the actual format back
        let actual_format = Capture::set_format(
            &device,
            &Format::new(desired_size.x as u32, desired_size.y as u32, desired_fourcc),
        )?;

        // extract size and pixel format
        self.size = Vec2::new(actual_format.width as usize, actual_format.height as usize);
        self.format = PixelFormat::from_fourcc(u32::from_le_bytes(actual_format.fourcc.repr))?;

        // build frame rate
        let desired_frame_rate = match config.frame_rate {
            Some(frame_rate) => frame_rate,
            None => {
                let params = Capture::params(&device)?;
                params.interval.denominator as f32 / params.interval.numerator as f32
            }
        };

        // set the frame rate and get the actual frame rate back
        let actual_params = Capture::set_params(
            &device,
            &v4l::video::capture::Parameters::with_fps(desired_frame_rate as u32),
        )?;

        // extract the frame rate
        self.frame_rate =
            actual_params.interval.denominator as f32 / actual_params.interval.numerator as f32;

        // create the stream
        self.stream = match MmapStream::with_buffers(&device, Type::VideoCapture, 4u32) {
            Ok(stream) => Some(stream),
            Err(error) => {
                return Err(VideoError::Stream(error.to_string()));
            }
        };

        Ok(VideoInConfig::V4l2(V4l2Config {
            path: config.path.clone(),
            size: Some(self.size),
            format: Some(self.format),
            frame_rate: Some(self.frame_rate),
        }))
    }

    fn close(&mut self) {
        self.stream.take();
    }

    fn blocking_capture(&mut self) -> Result<VideoFrame, VideoError> {
        if let Some(ref mut stream) = self.stream.as_mut() {
            match CaptureStream::next(*stream) {
                Ok((frame_data, _metadata)) => {
                    let color = Image::new(self.size, frame_data.to_vec(), self.format);
                    Ok(VideoFrame { color, depth: None, left: None, right: None })
                }
                Err(error) => Err(VideoError::Stream(error.to_string())),
            }
        } else {
            Err(VideoError::Stream("No stream".to_string()))
        }
    }
}
