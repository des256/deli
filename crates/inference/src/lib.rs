use std::fmt;

mod asr;
pub use asr::*;

mod tts;
pub use tts::*;

//mod diar;
//pub use diar::*;

mod inference;
pub use inference::*;

mod llm;
pub use llm::*;

mod vad;
pub use vad::*;

#[derive(Debug)]
pub enum InferError {
    Candle(String),
    Shape(String),
    Io(String),
    Runtime(String),
    TensorError(String),
    TokenizerError(String),
    Onnx(String),
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::Candle(msg) => write!(f, "candle error: {msg}"),
            InferError::Shape(msg) => write!(f, "shape error: {msg}"),
            InferError::Io(msg) => write!(f, "io error: {msg}"),
            InferError::Runtime(msg) => write!(f, "runtime error: {msg}"),
            InferError::TensorError(msg) => write!(f, "tensor error: {msg}"),
            InferError::TokenizerError(msg) => write!(f, "tokenizer error: {msg}"),
            InferError::Onnx(msg) => write!(f, "onnx error: {msg}"),
        }
    }
}
