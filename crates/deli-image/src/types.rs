use deli_base::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedImage {
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    F32(Tensor<f32>),
}

impl DecodedImage {
    pub fn shape(&self) -> &[usize] {
        match self {
            DecodedImage::U8(t) => &t.shape,
            DecodedImage::U16(t) => &t.shape,
            DecodedImage::F32(t) => &t.shape,
        }
    }

    pub fn height(&self) -> usize {
        self.shape()[0]
    }

    pub fn width(&self) -> usize {
        self.shape()[1]
    }

    pub fn channels(&self) -> usize {
        self.shape()[2]
    }
}
