pub struct TtsInput {
    pub id: u64,
    pub text: String,
}

pub struct TtsOutput {
    pub id: u64,
    pub audio: Vec<i16>,
}

//pub mod kokoro;
pub mod pocket;
