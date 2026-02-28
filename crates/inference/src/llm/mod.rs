pub struct LlmInput<T: Clone + Send + 'static> {
    pub payload: T,
    pub prompt: String,
}

pub enum LlmOutput<T: Clone + Send + 'static> {
    Token { payload: T, token: String },
    Eos { payload: T },
}

pub mod phi3;

/*
pub(crate) mod generate;
pub(crate) mod gemma3;
pub(crate) mod llama32;
pub(crate) mod smollm3;
pub use gemma3::Gemma3;
pub use llama32::Llama32;
pub use phi3::Phi3;
pub use smollm3::Smollm3;
*/
