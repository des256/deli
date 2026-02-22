use crate::error::{InferError, Result};
use std::fs;
use std::path::Path;

/// Load tokens from a tokens.txt file.
///
/// The file format is: `<token_text> <token_id>` per line.
/// Token IDs must be 0-based contiguous integers.
///
/// Returns a Vec<String> where index = token_id.
pub fn load_tokens<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path.as_ref()).map_err(|e| {
        InferError::Io(format!(
            "Failed to read tokens file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let mut tokens: Vec<(usize, String)> = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Split on whitespace - last token is the ID
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(InferError::Runtime(format!(
                "Malformed tokens file at line {}: expected '<token> <id>', got '{}'",
                line_num + 1,
                line
            )));
        }

        // Last part is the ID, everything before is the token text
        let id_str = parts[parts.len() - 1];
        let token_text = parts[..parts.len() - 1].join(" ");

        let id: usize = id_str.parse().map_err(|_| {
            InferError::Runtime(format!(
                "Malformed tokens file at line {}: invalid ID '{}'",
                line_num + 1,
                id_str
            ))
        })?;

        tokens.push((id, token_text));
    }

    // Sort by ID to ensure correct ordering
    tokens.sort_by_key(|(id, _)| *id);

    // Verify IDs are contiguous starting from 0
    for (i, (id, _)) in tokens.iter().enumerate() {
        if *id != i {
            return Err(InferError::Runtime(format!(
                "Non-contiguous token IDs: expected {}, got {}",
                i, id
            )));
        }
    }

    Ok(tokens.into_iter().map(|(_, text)| text).collect())
}
