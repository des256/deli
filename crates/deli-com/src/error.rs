use std::fmt;

#[derive(Debug)]
pub enum ComError {
    Io(std::io::Error),
    Decode(deli_codec::DecodeError),
    ConnectionClosed,
    MessageTooLarge(u32),
}

impl fmt::Display for ComError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComError::Io(err) => write!(f, "io error: {err}"),
            ComError::Decode(err) => write!(f, "decode error: {err}"),
            ComError::ConnectionClosed => write!(f, "connection closed"),
            ComError::MessageTooLarge(len) => write!(f, "message too large: {len} bytes"),
        }
    }
}

impl std::error::Error for ComError {}

impl From<std::io::Error> for ComError {
    fn from(err: std::io::Error) -> Self {
        ComError::Io(err)
    }
}

impl From<deli_codec::DecodeError> for ComError {
    fn from(err: deli_codec::DecodeError) -> Self {
        ComError::Decode(err)
    }
}
