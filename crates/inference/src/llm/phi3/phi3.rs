use {
    crate::*,
    base::*,
    std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc as std_mpsc,
        },
        time::Duration,
    },
    tokenizers::Tokenizer,
    tokio::sync::mpsc as tokio_mpsc,
};

const PHI3_MODEL_PATH: &str = "data/phi3/phi3-mini-4k-instruct-cuda-int4-rtn-block-32.onnx";
const PHI3_TOKENIZER_PATH: &str = "data/phi3/tokenizer.json";

const EOS_TOKEN_IDS: &[u32] = &[32000, 32001, 32007];
const DEFAULT_MAX_TOKENS: usize = 512;
const NUM_KV_HEADS: usize = 32;
const HEAD_DIM: usize = 96;

pub(crate) fn generate(
    session: &mut onnx::Session,
    cancel: &AtomicBool,
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
    tx: &tokio_mpsc::Sender<LlmToken>,
) -> bool {
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
                return true;
            }
        };
        kv_cache.push(cache_tensor);
    }

    for _ in 0..max_tokens {
        let input_ids_tensor = match make_i64_tensor(&session, &input_ids) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create attention mask tensor: {}", e);
                return true;
            }
        };
        let attention_mask_tensor = match make_i64_tensor(&session, &attention_mask) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create position ids tensor: {}", e);
                return true;
            }
        };
        let position_ids_tensor = match make_i64_tensor(&session, &position_ids) {
            Ok(t) => t,
            Err(e) => {
                log_error!("Failed to create position ids tensor: {}", e);
                return true;
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
            return true;
        }

        // Run inference
        let outputs = {
            let output_refs: Vec<&str> = output_names.iter().map(|s| s.as_str()).collect();
            match session.run(&inputs, &output_refs) {
                Ok(out) => out,
                Err(e) => {
                    log_error!("Failed to run inference: {}", e);
                    return true;
                }
            }
        };

        // exit right away if cancellation requested
        if cancel.swap(false, Ordering::Relaxed) {
            return false;
        }

        // Find logits output by name
        let logits_idx = match output_names.iter().position(|n| n == "logits") {
            Some(idx) => idx,
            None => {
                log_error!("logits output not found in model outputs");
                return true;
            }
        };

        let logits_data = match outputs[logits_idx].extract_as_f32() {
            Ok(data) => data,
            Err(e) => {
                log_error!("Failed to extract logits data: {}", e);
                return true;
            }
        };

        // Derive vocab_size from tensor data length
        let seq_len = input_ids.len();
        if seq_len == 0 || logits_data.is_empty() {
            log_error!("Empty logits tensor or input");
            return true;
        }
        let vocab_size = logits_data.len() / seq_len;
        let last_pos_offset = (seq_len - 1) * vocab_size;

        if last_pos_offset + vocab_size > logits_data.len() {
            log_error!(
                "Logits shape mismatch: expected at least {} elements, got {}",
                last_pos_offset + vocab_size,
                logits_data.len()
            );
            return true;
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
                return true;
            }
        };

        if eos_token_ids.contains(&(next_token_id as u32)) {
            tx.blocking_send(LlmToken::Eos).unwrap();
            return true;
        }

        all_generated_ids.push(next_token_id as u32);

        // Decode the full generated sequence to get correct spacing,
        // then emit only the new characters since last decode.
        let full_decoded = match tokenizer.decode(&all_generated_ids, true) {
            Ok(text) => text,
            Err(e) => {
                log_error!("Decode failed: {}", e);
                return true;
            }
        };

        if full_decoded.len() > prev_decoded_len {
            let delta = &full_decoded[prev_decoded_len..];
            prev_decoded_len = full_decoded.len();
            if tx.blocking_send(LlmToken::Text(delta.to_string())).is_err() {
                return true;
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

    true
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

pub struct Phi3Handle {
    input_tx: std_mpsc::Sender<String>,
    cancel: Arc<AtomicBool>,
}

pub struct Phi3Listener {
    output_rx: tokio_mpsc::Receiver<LlmToken>,
}

pub fn create(
    onnx: &Arc<onnx::Onnx>,
    executor: &onnx::Executor,
) -> Result<(Phi3Handle, Phi3Listener), InferError> {
    let mut session = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            PHI3_MODEL_PATH,
        )
        .map_err(|e| InferError::Onnx(e.to_string()))?;
    let tokenizer = Tokenizer::from_file(PHI3_TOKENIZER_PATH)
        .map_err(|e| InferError::Runtime(format!("Failed to load tokenizer: {}", e)))?;
    let tokenizer = Arc::new(tokenizer);

    let input_count = session
        .input_count()
        .map_err(|e| InferError::Onnx(e.to_string()))?;
    let mut input_names = Vec::with_capacity(input_count);
    for i in 0..input_count {
        input_names.push(
            session
                .input_name(i)
                .map_err(|e| InferError::Onnx(e.to_string()))?,
        );
    }

    let output_count = session
        .output_count()
        .map_err(|e| InferError::Onnx(e.to_string()))?;
    let mut output_names = Vec::with_capacity(output_count);
    for i in 0..output_count {
        output_names.push(
            session
                .output_name(i)
                .map_err(|e| InferError::Onnx(e.to_string()))?,
        );
    }

    // Derive num_layers by counting KV cache inputs by name pattern
    let kv_key_count = input_names.iter().filter(|n| n.contains(".key")).count();
    let kv_value_count = input_names.iter().filter(|n| n.contains(".value")).count();
    if kv_key_count != kv_value_count {
        return Err(InferError::Runtime(format!(
            "KV cache key/value count mismatch: {} keys, {} values",
            kv_key_count, kv_value_count
        )));
    }
    let num_layers = kv_key_count;

    let first_kv_idx = input_names
        .iter()
        .position(|n| n.contains(".key") || n.contains(".value"))
        .ok_or_else(|| InferError::Runtime("No KV cache inputs found".to_string()))?;
    let kv_dtype = session
        .input_element_type(first_kv_idx)
        .map_err(|e| InferError::Onnx(e.to_string()))?;

    let (input_tx, input_rx) = std_mpsc::channel::<String>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<LlmToken>(32);
    let cancel = Arc::new(AtomicBool::new(false));

    std::thread::spawn({
        let cancel = Arc::clone(&cancel);
        move || {
            loop {
                match input_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(mut text) => {
                        // Skip stale prompts — always use the latest in the channel
                        while let Ok(newer) = input_rx.try_recv() {
                            text = newer;
                        }
                        cancel.swap(false, Ordering::Relaxed);

                        let encoding = match tokenizer.encode(text, false) {
                            Ok(encoding) => encoding,
                            Err(e) => {
                                log_error!("Tokenization failed: {}", e);
                                continue;
                            }
                        };
                        let token_ids = encoding.get_ids().to_vec();
                        if token_ids.is_empty() {
                            log_error!("Empty tokenization result");
                            continue;
                        }

                        generate(
                            &mut session,
                            &cancel,
                            &tokenizer,
                            &EOS_TOKEN_IDS.to_vec(),
                            DEFAULT_MAX_TOKENS,
                            &input_names,
                            &output_names,
                            num_layers,
                            NUM_KV_HEADS,
                            HEAD_DIM,
                            &token_ids,
                            kv_dtype,
                            &output_tx,
                        );
                    }
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {
                        cancel.swap(false, Ordering::Relaxed);
                    }
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                        log_error!("Input channel disconnected");
                        break;
                    }
                }
            }
        }
    });

    Ok((Phi3Handle { input_tx, cancel }, Phi3Listener { output_rx }))
}

impl Phi3Handle {
    // send prompt to LLM, starts LLM generation
    pub fn send(&self, text: &str) -> Result<(), std_mpsc::SendError<String>> {
        self.input_tx.send(text.to_string()) // note: blocks if input channel is full
    }

    // cancel LLM generation, flush input channel
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Phi3Listener {
    // receive tokens from LLM
    pub async fn recv(&mut self) -> Option<LlmToken> {
        self.output_rx.recv().await
    }

    // try-receive tokens from LLM
    pub fn try_recv(&mut self) -> Option<LlmToken> {
        match self.output_rx.try_recv() {
            Ok(token) => Some(token),
            _ => None,
        }
    }
}
