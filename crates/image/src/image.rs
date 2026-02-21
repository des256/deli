use {crate::*, base::Vec2};

#[derive(Debug, Clone)]
pub struct Image {
    pub size: Vec2<usize>,
    pub data: Vec<u8>,
    pub format: PixelFormat,
}

impl Image {
    pub fn new(size: Vec2<usize>, data: Vec<u8>, format: PixelFormat) -> Self {
        Self { size, data, format }
    }
}
