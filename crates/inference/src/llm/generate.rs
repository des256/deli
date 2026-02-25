use {crate::*, base::*, tokenizers::Tokenizer, tokio::sync::mpsc};

/// Run the autoregressive generation loop, sending tokens through `tx`.
///
/// This function is designed to run inside `tokio::task::spawn_blocking`.
/// It processes the prompt on the first pass (full sequence + empty KV cache),
/// then generates one token at a time using cached keys/values.
pub(crate) fn generate(
    session: &mut onnx::Session,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    max_tokens: usize,
    input_names: &[String],
    output_names: &[String],
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    token_ids: &[u32],
    kv_dtype: onnx::ffi::ONNXTensorElementDataType,
    tx: &mpsc::Sender<LlmToken>,
) {
    // Initial state: convert token IDs to i64
    let mut input_ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
    let prompt_len = input_ids.len();
    let mut attention_mask: Vec<i64> = vec![1; prompt_len];
    let mut position_ids: Vec<i64> = (0..prompt_len as i64).collect();

    // Track all generated token IDs for incremental decoding.
    // Decoding tokens one at a time loses inter-token spaces in SentencePiece
    // tokenizers. Instead, decode the full sequence and emit only the delta.
    let mut all_generated_ids: Vec<u32> = Vec::new();
    let mut prev_decoded_len = 0;

    // Create empty KV cache tensors (num_layers × 2 for key and value)
    let mut kv_cache: Vec<onnx::Value> = Vec::with_capacity(num_layers * 2);
    for _ in 0..(num_layers * 2) {
        let cache_tensor = match onnx::Value::empty_typed(
            &session.onnx,
            &[1, num_kv_heads, 0, head_dim],
            kv_dtype,
        ) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create empty KV cache tensor: {}", e);
                return;
            }
        };
        kv_cache.push(cache_tensor);
    }

    for _ in 0..max_tokens {
        let input_ids_tensor = match make_i64_tensor(&session, &input_ids) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create attention mask tensor: {}", e);
                return;
            }
        };
        let attention_mask_tensor = match make_i64_tensor(&session, &attention_mask) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create position ids tensor: {}", e);
                return;
            }
        };
        let position_ids_tensor = match make_i64_tensor(&session, &position_ids) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create position ids tensor: {}", e);
                return;
            }
        };

        // Build input list by matching discovered names
        let mut inputs = Vec::with_capacity(3 + kv_cache.len());
        for name in input_names {
            if name == "input_ids" {
                inputs.push((name.as_str(), &input_ids_tensor));
            } else if name == "attention_mask" {
                inputs.push((name.as_str(), &attention_mask_tensor));
            } else if name == "position_ids" {
                inputs.push((name.as_str(), &position_ids_tensor));
            } else if let Some(cache_idx) = parse_kv_cache_index(name) {
                if cache_idx < kv_cache.len() {
                    inputs.push((name.as_str(), &kv_cache[cache_idx]));
                }
            }
        }

        if inputs.len() != input_names.len() {
            log_error!(
                "Built {} inputs but model expects {}",
                inputs.len(),
                input_names.len()
            );
            return;
        }

        // Run inference
        let outputs = {
            let output_refs: Vec<&str> = output_names.iter().map(|s| s.as_str()).collect();
            match session.run(&inputs, &output_refs) {
                Ok(out) => out,
                Err(e) => {
                    log_error!("Failed to run inference: {}", e);
                    return;
                }
            }
        };

        // Find logits output by name
        let logits_idx = match output_names.iter().position(|n| n == "logits") {
            Some(idx) => idx,
            None => {
                log_error!("logits output not found in model outputs");
                return;
            }
        };

        let logits_data = match outputs[logits_idx].extract_as_f32() {
            Ok(data) => data,
            Err(e) => {
                log_error!("Failed to extract logits data: {}", e);
                return;
            }
        };

        // Derive vocab_size from tensor data length
        let seq_len = input_ids.len();
        if seq_len == 0 || logits_data.is_empty() {
            log_error!("Empty logits tensor or input");
            return;
        }
        let vocab_size = logits_data.len() / seq_len;
        let last_pos_offset = (seq_len - 1) * vocab_size;

        if last_pos_offset + vocab_size > logits_data.len() {
            log_error!(
                "Logits shape mismatch: expected at least {} elements, got {}",
                last_pos_offset + vocab_size,
                logits_data.len()
            );
            return;
        }
        let last_pos_logits = &logits_data[last_pos_offset..last_pos_offset + vocab_size];

        // Argmax to find next token
        let next_token_id = match last_pos_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as i64)
        {
            Some(id) => id,
            None => {
                log_error!("Empty logits slice during argmax");
                return;
            }
        };

        if eos_token_ids.contains(&(next_token_id as u32)) {
            tx.blocking_send(LlmToken::Eos).unwrap();
            break;
        }

        all_generated_ids.push(next_token_id as u32);

        // Decode the full generated sequence to get correct spacing,
        // then emit only the new characters since last decode.
        let full_decoded = match tokenizer.decode(&all_generated_ids, true) {
            Ok(text) => text,
            Err(e) => {
                log_error!("Decode failed: {}", e);
                return;
            }
        };

        if full_decoded.len() > prev_decoded_len {
            let delta = &full_decoded[prev_decoded_len..];
            prev_decoded_len = full_decoded.len();
            if tx.blocking_send(LlmToken::Text(delta.to_string())).is_err() {
                break;
            }
        }

        // Update KV cache (all outputs except logits)
        kv_cache = outputs
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != logits_idx)
            .map(|(_, v)| v)
            .collect();

        // Prepare next iteration: single token input
        input_ids = vec![next_token_id];
        attention_mask.push(1);
        position_ids = vec![attention_mask.len() as i64 - 1];
    }
}

/// Create a 1D ONNX tensor with shape [1, len] from i64 data.
fn make_i64_tensor(session: &onnx::Session, data: &[i64]) -> Result<onnx::Value, InferError> {
    onnx::Value::from_slice::<i64>(&session.onnx, &[1, data.len()], data)
        .map_err(|e| InferError::Onnx(e.to_string()))
}

/// Parse a KV cache input name like "past_key_values.3.key" into a linear index.
/// Returns `layer * 2` for keys, `layer * 2 + 1` for values.
fn parse_kv_cache_index(name: &str) -> Option<usize> {
    let idx_str = name.strip_prefix("past_key_values.")?;
    let dot_pos = idx_str.find('.')?;
    let layer: usize = idx_str[..dot_pos].parse().ok()?;
    match &idx_str[dot_pos + 1..] {
        "key" => Some(layer * 2),
        "value" => Some(layer * 2 + 1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kv_cache_index_key() {
        assert_eq!(parse_kv_cache_index("past_key_values.0.key"), Some(0));
        assert_eq!(parse_kv_cache_index("past_key_values.3.key"), Some(6));
        assert_eq!(parse_kv_cache_index("past_key_values.15.key"), Some(30));
    }

    #[test]
    fn test_parse_kv_cache_index_value() {
        assert_eq!(parse_kv_cache_index("past_key_values.0.value"), Some(1));
        assert_eq!(parse_kv_cache_index("past_key_values.3.value"), Some(7));
        assert_eq!(parse_kv_cache_index("past_key_values.15.value"), Some(31));
    }

    #[test]
    fn test_parse_kv_cache_index_rejects_invalid() {
        assert_eq!(parse_kv_cache_index("input_ids"), None);
        assert_eq!(parse_kv_cache_index("attention_mask"), None);
        assert_eq!(parse_kv_cache_index("position_ids"), None);
        assert_eq!(parse_kv_cache_index("past_key_values.0.unknown"), None);
        assert_eq!(parse_kv_cache_index("past_key_values.abc.key"), None);
        assert_eq!(parse_kv_cache_index("present.0.key"), None);
    }
}
