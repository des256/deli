use crate::error::{InferError, Result};
use onnx::Session;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub(super) struct GenerateParams {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub eos_token_id: u32,
    pub max_tokens: usize,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub token_ids: Vec<u32>,
}

/// Run the autoregressive generation loop, sending tokens through `tx`.
///
/// This function is designed to run inside `tokio::task::spawn_blocking`.
/// It processes the prompt on the first pass (full sequence + empty KV cache),
/// then generates one token at a time using cached keys/values.
pub(super) fn generate(params: GenerateParams, tx: tokio::sync::mpsc::Sender<Result<String>>) {
    let GenerateParams {
        session,
        tokenizer,
        eos_token_id,
        max_tokens,
        input_names,
        output_names,
        num_layers,
        num_kv_heads,
        head_dim,
        token_ids,
    } = params;

    // Initial state: convert token IDs to i64
    let mut input_ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
    let prompt_len = input_ids.len();
    let mut attention_mask: Vec<i64> = vec![1; prompt_len];
    let mut position_ids: Vec<i64> = (0..prompt_len as i64).collect();

    // Create empty KV cache tensors (num_layers × 2 for key and value)
    let mut kv_cache: Vec<onnx::Value> = Vec::with_capacity(num_layers * 2);
    for _ in 0..(num_layers * 2) {
        let cache_tensor = match onnx::Value::from_slice::<f32>(
            &[1, num_kv_heads, 0, head_dim],
            &[],
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(Err(InferError::Onnx(e.to_string())));
                return;
            }
        };
        kv_cache.push(cache_tensor);
    }

    for _ in 0..max_tokens {
        let input_ids_tensor = match make_i64_tensor(&input_ids) {
            Ok(t) => t,
            Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
        };
        let attention_mask_tensor = match make_i64_tensor(&attention_mask) {
            Ok(t) => t,
            Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
        };
        let position_ids_tensor = match make_i64_tensor(&position_ids) {
            Ok(t) => t,
            Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
        };

        // Build input list by matching discovered names
        let mut inputs = Vec::with_capacity(3 + kv_cache.len());
        for name in &input_names {
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
            let _ = tx.blocking_send(Err(InferError::Runtime(format!(
                "Built {} inputs but model expects {}",
                inputs.len(), input_names.len()
            ))));
            return;
        }

        // Run inference
        let outputs = {
            let mut guard = match session.lock() {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.blocking_send(Err(InferError::Runtime(format!(
                        "Session lock failed: {}", e
                    ))));
                    return;
                }
            };
            let output_refs: Vec<&str> = output_names.iter().map(|s| s.as_str()).collect();
            match guard.run(&inputs, &output_refs) {
                Ok(out) => out,
                Err(e) => {
                    let _ = tx.blocking_send(Err(InferError::Onnx(e.to_string())));
                    return;
                }
            }
        };

        // Find logits output by name
        let logits_idx = match output_names.iter().position(|n| n == "logits") {
            Some(idx) => idx,
            None => {
                let _ = tx.blocking_send(Err(InferError::Runtime(
                    "logits output not found in model outputs".to_string(),
                )));
                return;
            }
        };

        let logits_data = match outputs[logits_idx].extract_tensor::<f32>() {
            Ok(data) => data,
            Err(e) => {
                let _ = tx.blocking_send(Err(InferError::Onnx(e.to_string())));
                return;
            }
        };

        // Derive vocab_size from tensor data length
        let seq_len = input_ids.len();
        if seq_len == 0 || logits_data.is_empty() {
            let _ = tx.blocking_send(Err(InferError::Runtime(
                "Empty logits tensor or input".to_string(),
            )));
            return;
        }
        let vocab_size = logits_data.len() / seq_len;
        let last_pos_offset = (seq_len - 1) * vocab_size;

        if last_pos_offset + vocab_size > logits_data.len() {
            let _ = tx.blocking_send(Err(InferError::Runtime(format!(
                "Logits shape mismatch: expected at least {} elements, got {}",
                last_pos_offset + vocab_size, logits_data.len()
            ))));
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
                let _ = tx.blocking_send(Err(InferError::Runtime(
                    "Empty logits slice during argmax".to_string(),
                )));
                return;
            }
        };

        if next_token_id as u32 == eos_token_id {
            break;
        }

        let decoded = match tokenizer.decode(&[next_token_id as u32], true) {
            Ok(text) => text,
            Err(e) => {
                let _ = tx.blocking_send(Err(InferError::TokenizerError(format!(
                    "Decode failed: {}", e
                ))));
                return;
            }
        };

        if decoded.is_empty() {
            continue;
        }

        if tx.blocking_send(Ok(decoded)).is_err() {
            break;
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
fn make_i64_tensor(data: &[i64]) -> Result<onnx::Value> {
    onnx::Value::from_slice::<i64>(&[1, data.len()], data)
        .map_err(|e| InferError::Onnx(e.to_string()))
}

/// Parse a KV cache input name like "past_key_values.3.key" into a linear index.
/// Returns `layer * 2` for keys, `layer * 2 + 1` for values.
fn parse_kv_cache_index(name: &str) -> Option<usize> {
    let idx_str = name.strip_prefix("past_key_values.")?;
    let dot_pos = idx_str.find('.')?;
    let layer: usize = idx_str[..dot_pos].parse().ok()?;
    let kv_type = &idx_str[dot_pos + 1..];
    Some(layer * 2 + if kv_type == "key" { 0 } else { 1 })
}
