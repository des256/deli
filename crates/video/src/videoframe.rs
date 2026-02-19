use base::{Tensor, Vec2};

#[derive(Debug, Clone)]
pub enum VideoData {
    Rgb(Tensor<u8>),
    Jpeg(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: VideoData,
    pub size: Vec2<usize>,
}
