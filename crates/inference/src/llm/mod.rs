pub enum LlmToken {
    Text(String),
    Eos,
}

mod phi3;
pub use phi3::*;

pub(crate) mod generate;

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
