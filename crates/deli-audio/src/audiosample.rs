use deli_base::Tensor;

/* old situation:
pub enum AudioSample {
    Pcm(Tensor<i16>),
}
*/

pub enum AudioData {
    Pcm(Tensor<i16>),
}

pub struct AudioSample {
    pub data: AudioData,
    pub sample_rate: usize,
}
