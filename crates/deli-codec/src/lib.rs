mod primitives;

pub use deli_codec_derive::Codec;

use std::fmt;

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    UnexpectedEof,
    InvalidUtf8,
    InvalidBool(u8),
    InvalidVariant(u32),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "unexpected end of buffer"),
            DecodeError::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            DecodeError::InvalidBool(v) => write!(f, "invalid bool value: {v}"),
            DecodeError::InvalidVariant(v) => write!(f, "invalid enum variant: {v}"),
        }
    }
}

impl std::error::Error for DecodeError {}

pub trait Codec: Sized {
    fn encode(&self, buf: &mut Vec<u8>);
    fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, DecodeError>;

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.encode(&mut buf);
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut pos = 0;
        Self::decode(bytes, &mut pos)
    }
}
