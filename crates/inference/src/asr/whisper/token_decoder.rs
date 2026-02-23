// Token decoder for Whisper ASR - greedy decoding loop
// Ported from candle-examples/examples/whisper/main.rs (Apache-2.0/MIT)

use {
    crate::{
        asr::whisper::{config::Config, model::Whisper},
        error::{InferError, Result},
    },
    candle_core::{Device, IndexOp, Tensor},
    tokenizers::Tokenizer,
};

/// Result of decoding a single audio segment
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Decoded text
    pub text: String,
    /// True if decoding was truncated (hit max token limit without EOT)
    pub truncated: bool,
    /// Probability that the segment contains no speech (0.0-1.0)
    pub no_speech_prob: f64,
}

/// Token decoder for autoregressive generation
pub struct TokenDecoder {
    model: Whisper,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    max_target_positions: usize,
    num_mel_bins: usize,
}

impl TokenDecoder {
    /// Create a new token decoder
    pub fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        device: &Device,
        config: &Config,
    ) -> Result<Self> {
        // Look up special tokens
        let sot_token = token_id(&tokenizer, "<|startoftranscript|>")?;
        let transcribe_token = token_id(&tokenizer, "<|transcribe|>")?;
        let eot_token = token_id(&tokenizer, "<|endoftext|>")?;
        let no_speech_token = token_id(&tokenizer, "<|nocaptions|>")?;
        let no_timestamps_token = token_id(&tokenizer, "<|notimestamps|>")?;

        // Build suppress_tokens tensor from config
        let suppress_tokens = if config.suppress_tokens.is_empty() {
            Tensor::zeros((1,), candle_core::DType::F32, device)
                .map_err(|e| InferError::TensorError(e.to_string()))?
        } else {
            Tensor::new(config.suppress_tokens.as_slice(), device)
                .map_err(|e| InferError::TensorError(e.to_string()))?
        };

        Ok(Self {
            model,
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            max_target_positions: config.max_target_positions,
            num_mel_bins: config.num_mel_bins,
        })
    }

    /// Decode a single audio segment (mel spectrogram)
    /// Returns decoded text and truncation flag
    pub fn decode(&mut self, mel: &Tensor) -> Result<DecodingResult> {
        let device = mel.device();

        // Add batch dimension if needed: [80, 3000] -> [1, 80, 3000]
        let mel = if mel.rank() == 2 {
            mel.unsqueeze(0)
                .map_err(|e| InferError::TensorError(e.to_string()))?
        } else {
            mel.clone()
        };

        // Encode audio features
        let audio_features = self
            .model
            .encoder_forward(&mel)
            .map_err(|e| InferError::TensorError(e.to_string()))?;

        // Initialize decoder tokens: [SOT, transcribe, no_timestamps]
        let mut tokens = vec![
            self.sot_token,
            self.transcribe_token,
            self.no_timestamps_token,
        ];

        // Greedy decoding loop
        let mut truncated = false;
        let mut no_speech_prob: f64 = 0.0;
        let max_tokens = self.max_target_positions / 2; // Stop at half of max

        for step in 0..max_tokens {
            // Convert tokens to tensor: [1, seq_len]
            let tokens_tensor = Tensor::new(tokens.as_slice(), device)
                .map_err(|e| InferError::TensorError(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| InferError::TensorError(e.to_string()))?;

            // Decoder forward: [1, seq_len] -> [1, seq_len, vocab_size]
            let logits = self
                .model
                .decoder_forward(&tokens_tensor, &audio_features)
                .map_err(|e| InferError::TensorError(e.to_string()))?;

            // Get last token logits: [1, seq_len, vocab_size] -> [vocab_size]
            let last_logits = logits
                .i((0, tokens.len() - 1))
                .map_err(|e: candle_core::Error| InferError::TensorError(e.to_string()))?;

            // On first step, compute no_speech_prob from raw logits before suppression
            if step == 0 {
                let probs = candle_nn::ops::softmax_last_dim(
                    &last_logits
                        .unsqueeze(0)
                        .map_err(|e| InferError::TensorError(e.to_string()))?,
                )
                .map_err(|e| InferError::TensorError(e.to_string()))?;
                no_speech_prob = probs
                    .i((0, self.no_speech_token as usize))
                    .map_err(|e: candle_core::Error| InferError::TensorError(e.to_string()))?
                    .to_scalar::<f32>()
                    .map_err(|e: candle_core::Error| InferError::TensorError(e.to_string()))?
                    as f64;
            }

            // Suppress tokens by setting their logits to -inf
            let last_logits = self.suppress_logits(last_logits)?;

            // Greedy: argmax
            let next_token = last_logits
                .argmax(0)
                .map_err(|e: candle_core::Error| InferError::TensorError(e.to_string()))?
                .to_scalar::<u32>()
                .map_err(|e: candle_core::Error| InferError::TensorError(e.to_string()))?;

            tokens.push(next_token);

            // Stop on EOT
            if next_token == self.eot_token {
                break;
            }

            // Check if we hit the limit
            if tokens.len() >= max_tokens {
                truncated = true;
                break;
            }
        }

        // Decode tokens to text (skip initial special tokens)
        let text_tokens = if tokens.len() > 3 {
            &tokens[3..] // Skip SOT, transcribe, no_timestamps
        } else {
            &[]
        };

        let text = self
            .tokenizer
            .decode(text_tokens, true)
            .map_err(|e| InferError::TokenizerError(e.to_string()))?;

        Ok(DecodingResult {
            text,
            truncated,
            no_speech_prob,
        })
    }

    /// Run decoder on full audio (handles multi-chunk processing)
    pub fn run(&mut self, mel: &Tensor) -> Result<String> {
        use super::config::NO_SPEECH_THRESHOLD;
        const N_FRAMES: usize = 3000; // 30 seconds of audio

        let (_, _, total_frames) = mel
            .dims3()
            .map_err(|e| InferError::TensorError(e.to_string()))?;

        let mut segments = Vec::new();
        let mut seek = 0;

        while seek < total_frames {
            let end = (seek + N_FRAMES).min(total_frames);
            let segment_frames = end - seek;

            // Extract segment: [1, 80, segment_frames]
            let segment = mel
                .narrow(2, seek, segment_frames)
                .map_err(|e| InferError::TensorError(e.to_string()))?;

            // Pad to N_FRAMES if last segment is short
            let segment = if segment_frames < N_FRAMES {
                let pad_size = N_FRAMES - segment_frames;
                let zeros =
                    Tensor::zeros((1, self.num_mel_bins, pad_size), mel.dtype(), mel.device())
                        .map_err(|e| InferError::TensorError(e.to_string()))?;
                Tensor::cat(&[&segment, &zeros], 2)
                    .map_err(|e| InferError::TensorError(e.to_string()))?
            } else {
                segment
            };

            // Reset KV cache for new segment
            self.model.reset_kv_cache();

            // Decode segment
            let result = self.decode(&segment)?;

            // Skip segments with high no-speech probability
            if result.no_speech_prob > NO_SPEECH_THRESHOLD {
                seek = end;
                continue;
            }

            segments.push(result.text);
            seek = end;
        }

        Ok(segments.join(""))
    }

    /// Suppress tokens by setting their logits to -inf
    fn suppress_logits(&self, logits: Tensor) -> Result<Tensor> {
        if self.suppress_tokens.elem_count() <= 1 {
            return Ok(logits);
        }

        let suppress_indices = self
            .suppress_tokens
            .to_vec1::<u32>()
            .map_err(|e| InferError::TensorError(e.to_string()))?;

        // Convert logits to Vec, apply suppression, reconstruct tensor
        let mut logits_vec = logits
            .to_vec1::<f32>()
            .map_err(|e| InferError::TensorError(e.to_string()))?;

        for &idx in &suppress_indices {
            let idx = idx as usize;
            if idx < logits_vec.len() {
                logits_vec[idx] = f32::NEG_INFINITY;
            }
        }

        Tensor::from_vec(logits_vec, logits.shape(), logits.device())
            .map_err(|e| InferError::TensorError(e.to_string()))
    }
}

/// Look up token ID by string representation
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        InferError::TokenizerError(format!("Token '{}' not found in tokenizer", token))
    })
}
