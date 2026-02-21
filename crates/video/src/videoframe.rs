use {base::Vec2, image::Image};

#[derive(Debug, Clone)]
pub struct DepthImage {
    pub size: Vec2<usize>,
    pub data: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct IrImage {
    pub size: Vec2<usize>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub color: Image,
    pub depth: Option<DepthImage>,
    pub left: Option<IrImage>,
    pub right: Option<IrImage>,
    // TODO: timestamp
}
