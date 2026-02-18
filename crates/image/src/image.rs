use base::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub enum Image {
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    F32(Tensor<f32>),
}

impl Image {
    pub fn shape(&self) -> &[usize] {
        match self {
            Image::U8(t) => &t.shape,
            Image::U16(t) => &t.shape,
            Image::F32(t) => &t.shape,
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
