use {
    crate::*,
    base::Vec2,
    std::path::PathBuf,
    v4l::{
        Device, Format, FourCC, buffer::Type, io::mmap::Stream as MmapStream,
        io::traits::CaptureStream, video::Capture,
    },
};

pub(crate) struct V4l2Camera {
    size: Vec2<usize>,
    stream: MmapStream<'static>,
}

impl V4l2Camera {
    pub fn open(
        path: Option<PathBuf>,
        width: usize,
        height: usize,
        format: VideoFormat,
        fps: f32,
    ) -> Result<Self, VideoError> {
        let device = match path {
            Some(path) => Device::with_path(path)?,
            None => Device::new(0)?,
        };
        let format = Format::new(
            width as u32,
            height as u32,
            match format {
                VideoFormat::Rgb => FourCC::new(b"RGB3"),
                VideoFormat::Argb => FourCC::new(b"ARGB"),
                VideoFormat::Yuyv => FourCC::new(b"YUYV"),
                VideoFormat::Jpeg => FourCC::new(b"MJPG"),
            },
        );
        let format = Capture::set_format(&device, &format)?;
        let params = v4l::video::capture::Parameters::with_fps(fps as u32);
        Capture::set_params(&device, &params)?;
        let stream = match MmapStream::with_buffers(&device, Type::VideoCapture, 4u32) {
            Ok(stream) => stream,
            Err(error) => {
                return Err(VideoError::Stream(error.to_string()));
            }
        };
        Ok(Self {
            size: Vec2::new(format.width as usize, format.height as usize),
            stream,
        })
    }

    fn capture(&mut self) -> Result<VideoFrame, VideoError> {
        match CaptureStream::next(&mut self.stream) {
            Ok((frame_data, _metadata)) => Ok(VideoFrame {
                data: VideoData::Jpeg(frame_data.to_vec()),
                size: self.size,
            }),
            Err(error) => Err(VideoError::Stream(error.to_string())),
        }
    }
}
