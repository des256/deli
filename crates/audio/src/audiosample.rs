use base::Tensor;

#[derive(Debug, Clone)]
pub enum AudioData {
    Pcm(Tensor<i16>),
}

#[derive(Debug, Clone)]
pub struct AudioSample {
    pub data: AudioData,
    pub sample_rate: usize,
}
