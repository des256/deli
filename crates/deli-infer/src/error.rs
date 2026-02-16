use std::fmt;

pub type Result<T> = std::result::Result<T, InferError>;

#[derive(Debug)]
pub enum InferError {
    Candle(String),
    Shape(String),
    Io(String),
    Runtime(String),
    TensorError(String),
    TokenizerError(String),
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
        }
    }
}

impl std::error::Error for InferError {}

impl From<candle_core::Error> for InferError {
    fn from(err: candle_core::Error) -> Self {
        InferError::Candle(err.to_string())
    }
}

impl From<std::io::Error> for InferError {
    fn from(err: std::io::Error) -> Self {
        InferError::Io(err.to_string())
    }
}
