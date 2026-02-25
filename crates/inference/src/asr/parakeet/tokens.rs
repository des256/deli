use {crate::*, std::path::Path};

/// Load token vocabulary from a SentencePiece `.model` file.
///
/// Parses the protobuf to extract piece strings in vocabulary order.
pub fn load_tokens<P: AsRef<Path>>(path: P) -> Result<Vec<String>, InferError> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).map_err(|e| InferError::Io(format!("{}: {}", path.display(), e)))?;

    parse_sentencepiece_model(&data)
}

/// Minimal protobuf parser for SentencePiece ModelProto.
///
/// Extracts piece strings from field 1 (repeated SentencePiece).
fn parse_sentencepiece_model(data: &[u8]) -> Result<Vec<String>, InferError> {
    let mut pieces = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let (tag, wire_type, new_pos) = read_tag(data, pos)?;
        pos = new_pos;

        if tag == 1 && wire_type == 2 {
            // Field 1 (pieces): length-delimited sub-message
            let (sub_data, new_pos) = read_bytes(data, pos)?;
            pos = new_pos;

            if let Some(piece) = parse_sentencepiece(sub_data)? {
                pieces.push(piece);
            }
        } else {
            pos = skip_field(data, pos, wire_type)?;
        }
    }

    if pieces.is_empty() {
        return Err(InferError::Runtime(
            "No pieces found in SentencePiece model".to_string(),
        ));
    }

    Ok(pieces)
}

/// Parse a SentencePiece sub-message, returning the piece string.
fn parse_sentencepiece(data: &[u8]) -> Result<Option<String>, InferError> {
    let mut pos = 0;
    let mut piece = None;

    while pos < data.len() {
        let (tag, wire_type, new_pos) = read_tag(data, pos)?;
        pos = new_pos;

        if tag == 1 && wire_type == 2 {
            let (bytes, new_pos) = read_bytes(data, pos)?;
            pos = new_pos;
            piece = Some(String::from_utf8_lossy(bytes).into_owned());
        } else {
            pos = skip_field(data, pos, wire_type)?;
        }
    }

    Ok(piece)
}

fn read_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize), InferError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            return Err(InferError::Runtime(
                "Unexpected end of protobuf data".to_string(),
            ));
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err(InferError::Runtime("Varint too long".to_string()));
        }
    }
}

fn read_tag(data: &[u8], pos: usize) -> Result<(u32, u8, usize), InferError> {
    let (value, new_pos) = read_varint(data, pos)?;
    let tag = (value >> 3) as u32;
    let wire_type = (value & 0x07) as u8;
    Ok((tag, wire_type, new_pos))
}

fn read_bytes<'a>(data: &'a [u8], pos: usize) -> Result<(&'a [u8], usize), InferError> {
    let (len, pos) = read_varint(data, pos)?;
    let len = len as usize;
    let end = pos + len;
    if end > data.len() {
        return Err(InferError::Runtime(
            "Length-delimited field exceeds data".to_string(),
        ));
    }
    Ok((&data[pos..end], end))
}

fn skip_field(data: &[u8], pos: usize, wire_type: u8) -> Result<usize, InferError> {
    match wire_type {
        0 => {
            let (_, new_pos) = read_varint(data, pos)?;
            Ok(new_pos)
        }
        1 => Ok(pos + 8),
        2 => {
            let (_, new_pos) = read_bytes(data, pos)?;
            Ok(new_pos)
        }
        5 => Ok(pos + 4),
        _ => Err(InferError::Runtime(format!(
            "Unknown wire type: {}",
            wire_type
        ))),
    }
}
