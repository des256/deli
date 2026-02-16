use crate::InferError;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_phi3::ModelWeights;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// Quantized Phi 4 language model for text generation.
///
/// Phi 4 uses the Phi 3 architecture from candle-transformers. The model is
/// loaded from a GGUF file and uses quantized weights for reduced memory usage.
///
/// The model is wrapped in `Arc<Mutex<>>` because `ModelWeights::forward` requires
/// `&mut self` for KV cache management. This means inference operations serialize
/// (one at a time per Phi4 instance).
#[derive(Clone)]
pub struct Phi4 {
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl Phi4 {
    /// Create a new Phi4 model from GGUF file and tokenizer.
    ///
    /// # Arguments
    /// * `model_path` - Path to the quantized model GGUF file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `device` - Device to run inference on (CPU or CUDA)
    ///
    /// # Returns
    /// A new Phi4 instance ready for text generation
    ///
    /// # Errors
    /// Returns error if:
    /// - Model file doesn't exist or can't be read
    /// - Tokenizer file doesn't exist or can't be read
    /// - GGUF file is corrupted or has wrong format
    /// - Model loading fails
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, InferError> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            InferError::Runtime(format!("Failed to load tokenizer: {}", e))
        })?;

        // Load GGUF model
        let mut file = std::fs::File::open(model_path.as_ref())
            .map_err(|e| InferError::Io(format!("Failed to open model file: {}", e)))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| InferError::Runtime(format!("Failed to read GGUF file: {}", e)))?;

        // Load model weights (use_flash_attn = false)
        let model = ModelWeights::from_gguf(false, content, &mut file, &device)
            .map_err(|e| InferError::Runtime(format!("Failed to load model weights: {}", e)))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    /// Generate text from a prompt using autoregressive sampling.
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

        tokio::task::spawn_blocking(move || -> Result<String, InferError> {
            // Tokenize prompt
            let tokens = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| InferError::TokenizerError(format!("Failed to encode prompt: {}", e)))?;
            let tokens = tokens.get_ids();

            if tokens.is_empty() {
                return Err(InferError::Runtime("Tokenization produced no tokens".to_string()));
            }

            // Look up EOS token
            let eos_token = *tokenizer
                .get_vocab(true)
                .get("<|endoftext|>")
                .ok_or_else(|| InferError::TokenizerError("<|endoftext|> token not found in vocab".to_string()))?;

            // Lock the model for the entire generation
            let mut model = model.lock()
                .map_err(|e| InferError::Runtime(format!("Failed to lock model mutex: {}", e)))?;

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

            let mut generated_tokens = vec![next_token];

            // Generate loop (first token already sampled above)
            for index in 1..sample_len {
                // Check for EOS
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
