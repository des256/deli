pub struct LlmInput<T: Clone + Send + 'static> {
    pub payload: T,
    pub prompt: String,
}

pub enum LlmOutput<T: Clone + Send + 'static> {
    Token { payload: T, token: String },
    Eos { payload: T },
}

pub mod gemma3;
pub mod llama3;
pub mod phi3;

mod history;
pub use history::*;
