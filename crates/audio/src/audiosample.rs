use base::Tensor;

pub enum AudioData {
    Pcm(Tensor<i16>),
}

pub struct AudioSample {
    pub data: AudioData,
    pub sample_rate: usize,
}
