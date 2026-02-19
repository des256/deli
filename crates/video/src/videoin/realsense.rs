use crate::{Camera, VideoData, VideoError, VideoFrame};

pub(crate) struct RealsenseCamera {}

impl RealsenseCamera {
    pub fn open(index: usize) -> Result<Self, VideoError> {
        // TODO: open RealSense camera with index
        Ok(Self {})
    }
}

impl Camera for RealsenseCamera {
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
