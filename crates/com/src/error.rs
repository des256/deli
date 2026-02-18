use std::fmt;

#[derive(Debug)]
pub enum ComError {
    Io(std::io::Error),
    Decode(codec::DecodeError),
    ConnectionClosed,
    MessageTooLarge(u32),
    WebSocket(tokio_websockets::Error),
}

impl fmt::Display for ComError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComError::Io(err) => write!(f, "io error: {err}"),
            ComError::Decode(err) => write!(f, "decode error: {err}"),
            ComError::ConnectionClosed => write!(f, "connection closed"),
            ComError::MessageTooLarge(len) => write!(f, "message too large: {len} bytes"),
            ComError::WebSocket(err) => write!(f, "websocket error: {err}"),
        }
    }
}

impl std::error::Error for ComError {}

impl From<std::io::Error> for ComError {
    fn from(err: std::io::Error) -> Self {
        ComError::Io(err)
    }
}

impl From<codec::DecodeError> for ComError {
    fn from(err: codec::DecodeError) -> Self {
        ComError::Decode(err)
    }
}

impl From<tokio_websockets::Error> for ComError {
    fn from(err: tokio_websockets::Error) -> Self {
        ComError::WebSocket(err)
    }
}
