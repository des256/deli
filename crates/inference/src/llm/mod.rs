pub struct LlmInput<T: Clone + Send + 'static> {
    pub payload: T,
    pub prompt: String,
}

pub enum LlmOutput<T: Clone + Send + 'static> {
    Token { payload: T, token: String },
    Eos { payload: T },
}

pub mod phi3;
pub mod llama32;
pub mod gemma3;
