#[derive(Clone, Debug)]
pub struct TtsPayload<T: Clone + Send + 'static> {
    pub payload: T,
    pub id: u64,
}

#[derive(Clone, Debug)]
pub struct TtsInput<T: Clone + Send + 'static> {
    pub payload: T,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct TtsOutput<T: Clone + Send + 'static> {
    pub payload: TtsPayload<T>,
    pub data: Vec<i16>,
}

//pub mod kokoro;
pub mod pocket;
