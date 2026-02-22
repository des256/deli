use {codec::DecodeError, com::ComError, std::io};

#[test]
fn test_error_has_all_variants() {
    // Test that ComError has all required variants
    let _io_err = ComError::Io(io::Error::new(io::ErrorKind::Other, "test"));
    let _decode_err = ComError::Decode(DecodeError::UnexpectedEof);
    let _closed = ComError::ConnectionClosed;
    let _too_large = ComError::MessageTooLarge(1000);
}

#[test]
fn test_from_io_error() {
    let io_err = io::Error::new(io::ErrorKind::BrokenPipe, "pipe broken");
    let com_err: ComError = io_err.into();
    match com_err {
        ComError::Io(_) => {} // Expected
        _ => panic!("Expected ComError::Io variant"),
    }
}

#[test]
fn test_from_decode_error() {
    let decode_err = DecodeError::InvalidUtf8;
    let com_err: ComError = decode_err.into();
    match com_err {
        ComError::Decode(_) => {} // Expected
        _ => panic!("Expected ComError::Decode variant"),
    }
}

#[test]
fn test_display_io() {
    let err = ComError::Io(io::Error::new(io::ErrorKind::ConnectionReset, "reset"));
    let display = format!("{}", err);
    assert!(display.contains("io error"));
}

#[test]
fn test_display_decode() {
    let err = ComError::Decode(DecodeError::UnexpectedEof);
    let display = format!("{}", err);
    assert!(display.contains("decode error"));
}

#[test]
fn test_display_connection_closed() {
    let err = ComError::ConnectionClosed;
    let display = format!("{}", err);
    assert!(display.contains("connection closed"));
}

#[test]
fn test_display_message_too_large() {
    let err = ComError::MessageTooLarge(100_000_000);
    let display = format!("{}", err);
    assert!(display.contains("message too large"));
    assert!(display.contains("100000000"));
}

#[test]
fn test_error_trait() {
    use std::error::Error;
    let err = ComError::ConnectionClosed;
    let _: &dyn Error = &err; // Verify it implements Error trait
}

#[test]
fn test_websocket_error_variant() {
    // Test that ComError::WebSocket variant exists
    let ws_err = tokio_websockets::Error::AlreadyClosed;
    let com_err = ComError::WebSocket(ws_err);
    match com_err {
        ComError::WebSocket(_) => {} // Expected
        _ => panic!("Expected ComError::WebSocket variant"),
    }
}

#[test]
fn test_from_websocket_error() {
    let ws_err = tokio_websockets::Error::AlreadyClosed;
    let com_err: ComError = ws_err.into();
    match com_err {
        ComError::WebSocket(_) => {} // Expected
        _ => panic!("Expected ComError::WebSocket variant"),
    }
}

#[test]
fn test_display_websocket() {
    let err = ComError::WebSocket(tokio_websockets::Error::AlreadyClosed);
    let display = format!("{}", err);
    assert!(display.contains("websocket error"));
}
