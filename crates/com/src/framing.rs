use {crate::ComError, codec::Codec, tokio::io::{AsyncReadExt, AsyncWriteExt}};

pub const MAX_MESSAGE_SIZE: u32 = 64 * 1024 * 1024; // 64 MB

/// Write a length-prefixed message to an async writer.
///
/// Encodes `value` using `Codec::to_bytes()`, writes a 4-byte little-endian
/// length prefix, then writes the payload.
pub async fn write_message<T: Codec, W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    value: &T,
) -> Result<(), ComError> {
    let payload = value.to_bytes();
    let len = u32::try_from(payload.len()).map_err(|_| ComError::MessageTooLarge(u32::MAX))?;

    if len > MAX_MESSAGE_SIZE {
        return Err(ComError::MessageTooLarge(len));
    }

    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(&payload).await?;

    Ok(())
}

/// Read a length-prefixed message from an async reader.
///
/// Reads a 4-byte little-endian length, validates it against `MAX_MESSAGE_SIZE`,
/// reads the payload, and decodes `T` using `Codec::from_bytes()`.
///
/// Returns `ComError::ConnectionClosed` if EOF is encountered.
/// Returns `ComError::MessageTooLarge` if length exceeds `MAX_MESSAGE_SIZE`.
pub async fn read_message<T: Codec, R: AsyncReadExt + Unpin>(
    reader: &mut R,
) -> Result<T, ComError> {
    // Read 4-byte length prefix
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(ComError::ConnectionClosed);
        }
        Err(e) => return Err(e.into()),
    }

    let len = u32::from_le_bytes(len_buf);

    // Validate length
    if len > MAX_MESSAGE_SIZE {
        return Err(ComError::MessageTooLarge(len));
    }

    // Read payload
    let mut payload = vec![0u8; len as usize];
    match reader.read_exact(&mut payload).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(ComError::ConnectionClosed);
        }
        Err(e) => return Err(e.into()),
    }

    // Decode
    T::from_bytes(&payload).map_err(ComError::from)
}
