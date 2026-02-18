use deli_base::Tensor;

pub enum AudioSample {
    Pcm(Tensor<i16>),
}
