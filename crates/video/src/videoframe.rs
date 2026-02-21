use base::{Tensor, Vec2};

#[derive(Debug, Clone, Copy)]
pub enum VideoFormat {
    Yuyv,
    Jpeg,
    Srggb10p,
}

#[derive(Debug, Clone)]
pub enum VideoData {
    Yuyv(Tensor<u8>),
    Jpeg(Vec<u8>),
    Srggb10p(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: VideoData,
    pub size: Vec2<usize>,
}
