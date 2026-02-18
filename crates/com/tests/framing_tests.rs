use com::{ComError, framing};
use tokio::io::AsyncWriteExt;

#[tokio::test]
async fn test_round_trip_u32() {
    let (mut writer, mut reader) = tokio::io::duplex(1024);

    let value: u32 = 42;
    framing::write_message(&mut writer, &value)
        .await
        .expect("write failed");

    let decoded: u32 = framing::read_message(&mut reader)
        .await
        .expect("read failed");

    assert_eq!(decoded, value);
}

#[tokio::test]
async fn test_round_trip_string() {
    let (mut writer, mut reader) = tokio::io::duplex(1024);

    let value = "Hello, MCSP!".to_string();
    framing::write_message(&mut writer, &value)
        .await
        .expect("write failed");

    let decoded: String = framing::read_message(&mut reader)
        .await
        .expect("read failed");

    assert_eq!(decoded, value);
}

#[tokio::test]
async fn test_eof_returns_connection_closed() {
    use tokio::io::AsyncWriteExt;

    let (mut writer, mut reader) = tokio::io::duplex(1024);

    // Write an incomplete message (just 2 bytes of the 4-byte length prefix)
    writer.write_all(&[0x01, 0x02]).await.unwrap();
    // Then close the writer
    drop(writer);

    // Attempt to read - should fail with ConnectionClosed because we can't read full length prefix
    let result: Result<u32, ComError> = framing::read_message(&mut reader).await;

    match result {
        Err(ComError::ConnectionClosed) => {} // Expected
        other => panic!("Expected ConnectionClosed, got {:?}", other),
    }
}

#[tokio::test]
async fn test_message_too_large_rejected() {
    let (mut writer, mut reader) = tokio::io::duplex(16);

    // Write a length that exceeds MAX_MESSAGE_SIZE
    let huge_length = framing::MAX_MESSAGE_SIZE + 1;
    writer
        .write_all(&huge_length.to_le_bytes())
        .await
        .expect("write length failed");

    let result: Result<u32, ComError> = framing::read_message(&mut reader).await;

    match result {
        Err(ComError::MessageTooLarge(len)) => {
            assert_eq!(len, huge_length);
        }
        other => panic!("Expected MessageTooLarge, got {:?}", other),
    }
}

#[tokio::test]
async fn test_multiple_messages_in_sequence() {
    let (mut writer, mut reader) = tokio::io::duplex(1024);

    // Write three messages
    framing::write_message(&mut writer, &10u32).await.unwrap();
    framing::write_message(&mut writer, &20u32).await.unwrap();
    framing::write_message(&mut writer, &30u32).await.unwrap();

    // Read them back
    let v1: u32 = framing::read_message(&mut reader).await.unwrap();
    let v2: u32 = framing::read_message(&mut reader).await.unwrap();
    let v3: u32 = framing::read_message(&mut reader).await.unwrap();

    assert_eq!(v1, 10);
    assert_eq!(v2, 20);
    assert_eq!(v3, 30);
}

#[tokio::test]
async fn test_max_message_size_constant_exists() {
    // Verify MAX_MESSAGE_SIZE is exported and has expected value
    assert_eq!(framing::MAX_MESSAGE_SIZE, 64 * 1024 * 1024);
}

/// A Codec type that produces a payload of a specific size for testing write-side validation.
struct OversizedPayload(usize);

impl codec::Codec for OversizedPayload {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.resize(self.0, 0);
    }

    fn decode(_buf: &[u8], _pos: &mut usize) -> Result<Self, codec::DecodeError> {
        Err(codec::DecodeError::UnexpectedEof)
    }
}

#[tokio::test]
async fn test_write_message_rejects_oversized_payload() {
    let (mut writer, _reader) = tokio::io::duplex(64);

    let oversized = OversizedPayload((framing::MAX_MESSAGE_SIZE as usize) + 1);
    let result = framing::write_message(&mut writer, &oversized).await;

    match result {
        Err(ComError::MessageTooLarge(len)) => {
            assert_eq!(len, framing::MAX_MESSAGE_SIZE + 1);
        }
        other => panic!("Expected MessageTooLarge, got {:?}", other),
    }
}
