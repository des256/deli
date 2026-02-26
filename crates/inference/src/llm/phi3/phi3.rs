use {crate::*, base::*, std::sync::Arc, tokenizers::Tokenizer, tokio::sync::mpsc};

const PHI3_MODEL_PATH: &str = "data/phi3/phi3-mini-4k-instruct-cuda-int4-rtn-block-32.onnx";
const PHI3_TOKENIZER_PATH: &str = "data/phi3/tokenizer.json";

const EOS_TOKEN_IDS: &[u32] = &[32000, 32001, 32007];
const DEFAULT_MAX_TOKENS: usize = 512;
const NUM_KV_HEADS: usize = 32;
const HEAD_DIM: usize = 96;

/// Phi 3 ONNX-based language model with KV cache
///
/// Supports autoregressive text generation with streaming token output.
/// The ONNX model uses grouped-query attention with KV caching for efficient generation.
/// Note: Phi 3 does not use position_ids input (only input_ids + attention_mask + KV cache).
pub struct Phi3 {
    input_tx: mpsc::Sender<String>,
    output_rx: mpsc::Receiver<LlmToken>,
}

impl Phi3 {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: &onnx::Executor) -> Result<Self, InferError> {
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

        let (input_tx, mut input_rx) = mpsc::channel::<String>(32);
        let (output_tx, output_rx) = mpsc::channel::<LlmToken>(32);

        tokio::task::spawn_blocking({
            move || {
                while let Some(text) = input_rx.blocking_recv() {
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

                    generate::generate(
                        &mut session,
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
            }
        });

        Ok(Self {
            input_tx,
            output_rx,
        })
    }

    pub fn input_tx(&self) -> mpsc::Sender<String> {
        self.input_tx.clone()
    }

    pub async fn send(&mut self, text: &str) -> Result<(), InferError> {
        self.input_tx
            .send(text.to_string())
            .await
            .map_err(|e| InferError::Runtime(format!("Failed to send text: {}", e)))
    }

    pub async fn recv(&mut self) -> Option<LlmToken> {
        self.output_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<LlmToken> {
        match self.output_rx.try_recv() {
            Ok(token) => Some(token),
            _ => None,
        }
    }
}
