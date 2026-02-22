use {
    crate::InferError,
    candle_core::{Device, Tensor},
    candle_transformers::{generation::{LogitsProcessor, Sampling}, models::quantized_llama::ModelWeights},
    std::{path::Path, sync::{Arc, Mutex}},
    tokenizers::Tokenizer,
};

/// Quantized Llama language model for text generation (TinyLlama 1.1B).
///
/// Llama uses the Llama architecture from candle-transformers with quantized weights
/// for reduced memory usage. The model is loaded from a GGUF file.
///
/// The model is wrapped in `Arc<Mutex<>>` because `ModelWeights::forward` requires
/// `&mut self` for KV cache management. This means inference operations serialize
/// (one at a time per Llama instance).
#[derive(Clone)]
pub struct Llama {
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    eos_token: u32,
}

impl Llama {
    /// Create a new Llama model from GGUF file and tokenizer.
    ///
    /// Validates the tokenizer contains the required `</s>` EOS token at
    /// construction time.
    ///
    /// # Arguments
    /// * `model_path` - Path to the quantized model GGUF file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `device` - Device to run inference on (CPU or CUDA)
    ///
    /// # Returns
    /// A new Llama instance ready for text generation
    ///
    /// # Errors
    /// Returns error if:
    /// - Model file doesn't exist or can't be read
    /// - Tokenizer file doesn't exist or can't be read
    /// - GGUF file is corrupted or has wrong format
    /// - Model loading fails
    /// - Primary EOS token `</s>` not found in tokenizer vocab
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, InferError> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            InferError::Runtime(format!("Failed to load tokenizer: {}", e))
        })?;

        // Validate EOS token at construction time (fail-fast for wrong tokenizer)
        let eos_token = *tokenizer
            .get_vocab(true)
            .get("</s>")
            .ok_or_else(|| InferError::TokenizerError("</s> token not found in vocab".to_string()))?;

        // Load GGUF model
        let mut file = std::fs::File::open(model_path.as_ref())
            .map_err(|e| InferError::Io(format!("Failed to open model file: {}", e)))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| InferError::Runtime(format!("Failed to read GGUF file: {}", e)))?;

        // Load model weights (from_gguf takes 3 args: ct, reader, device)
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| InferError::Runtime(format!("Failed to load model weights: {}", e)))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            eos_token,
        })
    }

    /// Generate text from a prompt using autoregressive sampling.
    ///
    /// Generation stops at `</s>` (EOS token).
    ///
    /// # Arguments
    /// * `prompt` - Input text to generate from
    /// * `sample_len` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// Generated text as a String
    ///
    /// # Errors
    /// Returns error if:
    /// - Prompt is empty
    /// - Tokenization fails
    /// - Model inference fails
    /// - Decoding fails
    pub async fn forward(&self, prompt: &str, sample_len: usize) -> Result<String, InferError> {
        // Validate prompt
        if prompt.is_empty() {
            return Err(InferError::Runtime("Prompt cannot be empty".to_string()));
        }

        // Clone Arcs for move into spawn_blocking
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let prompt = prompt.to_string();
        let eos_token = self.eos_token;

        tokio::task::spawn_blocking(move || -> Result<String, InferError> {
            // Tokenize prompt
            let tokens = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| InferError::TokenizerError(format!("Failed to encode prompt: {}", e)))?;
            let tokens = tokens.get_ids();

            if tokens.is_empty() {
                return Err(InferError::Runtime("Tokenization produced no tokens".to_string()));
            }

            // Lock the model for the entire generation
            let mut model = model.lock()
                .map_err(|e| InferError::Runtime(format!("Failed to lock model mutex: {}", e)))?;

            // Llama's KV cache auto-resets when index_pos == 0, no explicit clear needed

            // Create LogitsProcessor with greedy (ArgMax) sampling
            let mut logits_processor = LogitsProcessor::from_sampling(299792458, Sampling::ArgMax);

            // Forward all prompt tokens at once
            let input = Tensor::new(tokens, &device)
                .map_err(|e| InferError::TensorError(format!("Failed to create prompt tensor: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| InferError::TensorError(format!("Failed to unsqueeze tensor: {}", e)))?;

            let logits = model.forward(&input, 0)
                .map_err(|e| InferError::Runtime(format!("Model forward failed: {}", e)))?;

            let logits = logits.squeeze(0)
                .map_err(|e| InferError::TensorError(format!("Failed to squeeze logits: {}", e)))?;

            // Sample first token from prompt logits
            let mut next_token = logits_processor.sample(&logits)
                .map_err(|e| InferError::Runtime(format!("Sampling failed: {}", e)))?;

            // Check first token for EOS before adding to output
            let mut generated_tokens = Vec::new();
            if next_token != eos_token {
                generated_tokens.push(next_token);
            }

            // Generate loop (first token already sampled above)
            for index in 1..sample_len {
                // Check for EOS token
                if next_token == eos_token {
                    break;
                }

                // Forward single token
                let input = Tensor::new(&[next_token], &device)
                    .map_err(|e| InferError::TensorError(format!("Failed to create token tensor: {}", e)))?
                    .unsqueeze(0)
                    .map_err(|e| InferError::TensorError(format!("Failed to unsqueeze token tensor: {}", e)))?;

                let logits = model.forward(&input, tokens.len() + index)
                    .map_err(|e| InferError::Runtime(format!("Model forward failed at token {}: {}", index, e)))?;

                let logits = logits.squeeze(0)
                    .map_err(|e| InferError::TensorError(format!("Failed to squeeze logits: {}", e)))?;

                // Sample next token
                next_token = logits_processor.sample(&logits)
                    .map_err(|e| InferError::Runtime(format!("Sampling failed at token {}: {}", index, e)))?;

                generated_tokens.push(next_token);
            }

            // Decode generated tokens to text
            let text = tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| InferError::TokenizerError(format!("Failed to decode tokens: {}", e)))?;

            Ok(text)
        })
        .await
        .map_err(|e| InferError::Runtime(format!("Task join error: {}", e)))?
    }
}
