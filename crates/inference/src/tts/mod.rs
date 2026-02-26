pub struct TtsInput {
    pub text: String,
}

pub struct TtsOutput {
    pub audio: Vec<i16>,
}

//pub mod kokoro;
pub mod pocket;
