use crate::error::{InferError, Result};
use onnx::Session;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

use super::generate::{self, GenerateParams};

const EOS_TOKEN_ID: u32 = 128012;
const DEFAULT_MAX_TOKENS: usize = 512;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;

/// SmolLM3 3B ONNX-based language model with KV cache
///
/// Supports autoregressive text generation with streaming token output.
/// The ONNX model uses grouped-query attention with KV caching for efficient generation.
pub struct Smollm3 {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    eos_token_id: u32,
    max_tokens: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
    rx: Option<tokio::sync::mpsc::Receiver<Result<String>>>,
}

impl Smollm3 {
    /// Create a new Smollm3 instance from an ONNX session and tokenizer
    ///
    /// Loads the tokenizer, discovers all model I/O names dynamically,
    /// derives num_layers from the KV cache input count, and wraps the
    /// session in Arc<Mutex<>> for thread-safe access.
    pub fn new<P: AsRef<Path>>(session: Session, tokenizer_path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| InferError::Runtime(format!("Failed to load tokenizer: {}", e)))?;
        let tokenizer = Arc::new(tokenizer);

        let input_count = session.input_count()?;
        let mut input_names = Vec::with_capacity(input_count);
        for i in 0..input_count {
            input_names.push(session.input_name(i)?);
        }

        let output_count = session.output_count()?;
        let mut output_names = Vec::with_capacity(output_count);
        for i in 0..output_count {
            output_names.push(session.output_name(i)?);
        }

        // Derive num_layers: input_count = 3 base + (num_layers * 2) KV cache
        if input_count < 3 {
            return Err(InferError::Runtime(format!(
                "Invalid input count: expected at least 3, got {}",
                input_count
            )));
        }
        let kv_count = input_count - 3;
        if kv_count % 2 != 0 {
            return Err(InferError::Runtime(format!(
                "Expected even number of KV cache inputs, got {} (total inputs: {})",
                kv_count, input_count
            )));
        }
        let num_layers = kv_count / 2;

        let session = Arc::new(Mutex::new(session));

        Ok(Self {
            session,
            tokenizer,
            eos_token_id: EOS_TOKEN_ID,
            max_tokens: DEFAULT_MAX_TOKENS,
            num_layers,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            input_names,
            output_names,
            rx: None,
        })
    }

    /// Set the maximum number of tokens to generate.
    /// Default is 512.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Start autoregressive text generation for the given input text.
    ///
    /// Spawns a background task that generates tokens using the ONNX model with KV caching.
    /// Tokens can be received asynchronously via `recv()`.
    pub fn forward(&mut self, text: &str) -> Result<()> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| InferError::TokenizerError(format!("Tokenization failed: {}", e)))?;
        let token_ids = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(InferError::Runtime("Empty tokenization result".to_string()));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        self.rx = Some(rx);

        let params = GenerateParams {
            session: Arc::clone(&self.session),
            tokenizer: Arc::clone(&self.tokenizer),
            eos_token_id: self.eos_token_id,
            max_tokens: self.max_tokens,
            input_names: self.input_names.clone(),
            output_names: self.output_names.clone(),
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            token_ids,
        };

        tokio::task::spawn_blocking(move || {
            generate::generate(params, tx);
        });

        Ok(())
    }

    /// Receive the next generated token asynchronously.
    ///
    /// Returns `None` when generation is complete.
    pub async fn recv(&mut self) -> Option<Result<String>> {
        self.rx.as_mut()?.recv().await
    }
}
