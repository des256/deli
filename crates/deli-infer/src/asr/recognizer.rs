// SpeechRecognizer - Public API for Whisper ASR

use crate::asr::{audio::pcm_to_mel, config::Config, model::Whisper, token_decoder::TokenDecoder};
use crate::error::{InferError, Result};
use candle_core::{Device, Tensor as CandleTensor};
use candle_nn::VarBuilder;
use deli_base::Tensor as BaseTensor;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Whisper-based speech recognition system
pub struct SpeechRecognizer {
    model: Arc<Whisper>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
    device: Device,
}

impl SpeechRecognizer {
    /// Create a new speech recognizer from model, tokenizer, and config files
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        model_path: P1,
        tokenizer_path: P2,
        config_path: P3,
        device: Device,
    ) -> Result<Self> {
        // Load config
        let config_json = std::fs::read_to_string(config_path)
            .map_err(|e| InferError::Io(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_json)
            .map_err(|e| InferError::Runtime(format!("Failed to parse config: {}", e)))?;

        Self::new_with_config(model_path, tokenizer_path, config, device)
    }

    /// Create a new speech recognizer with an explicit config
    pub fn new_with_config<P1: AsRef<Path>, P2: AsRef<Path>>(
        model_path: P1,
        tokenizer_path: P2,
        config: Config,
        device: Device,
    ) -> Result<Self> {
        // Load model from safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], candle_core::DType::F32, &device)
                .map_err(|e| InferError::Runtime(format!("Failed to load model: {}", e)))?
        };

        // Validate model format by checking for expected keys
        // This prevents loading wrong model types (e.g., distil-whisper, non-Whisper models)
        // Note: Whisper::load already adds "model.encoder" and "model.decoder" prefixes
        // internally, so we pass vb directly without an extra "model" prefix
        let model = Whisper::load(vb, config.clone())
            .map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to build Whisper model. Expected OpenAI Whisper format from openai/whisper-tiny.en. Error: {}",
                    e
                ))
            })?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferError::Runtime(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
            device,
        })
    }

    /// Transcribe audio to text
    ///
    /// # Arguments
    /// * `audio` - Audio samples as 16-bit PCM (1D tensor)
    /// * `sample_rate` - Sample rate in Hz (must be 16000)
    ///
    /// # Returns
    /// Transcribed text
    ///
    /// # Errors
    /// Returns error if:
    /// - Sample rate is not 16000 Hz
    /// - Audio tensor is not 1D
    /// - Audio is empty
    /// - Inference fails
    pub async fn transcribe(&self, audio: &BaseTensor<i16>, sample_rate: u32) -> Result<String> {
        // Validate sample rate
        if sample_rate != 16000 {
            return Err(InferError::Runtime(format!(
                "Whisper requires 16000 Hz audio, got {} Hz. Please resample before calling transcribe.",
                sample_rate
            )));
        }

        // Validate input shape
        if audio.shape.len() != 1 {
            return Err(InferError::Shape(format!(
                "Audio must be 1D tensor, got shape {:?}",
                audio.shape
            )));
        }

        if audio.data.is_empty() {
            return Err(InferError::Runtime("Audio is empty".to_string()));
        }

        // Clone Arcs for move into spawn_blocking
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let config = self.config.clone();
        let device = self.device.clone();

        // Convert audio to Vec for move
        let audio_samples = audio.data.clone();

        // Run inference in blocking task
        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            // Convert i16 to f32 samples: [-32768, 32767] -> [-1.0, 1.0]
            let f32_samples: Vec<f32> = audio_samples
                .iter()
                .map(|&s| s as f32 / 32768.0)
                .collect();

            // Compute mel spectrogram
            let mel_data = pcm_to_mel(&config, &f32_samples);

            // Reshape to [1, 80, frames]
            let num_frames = mel_data.len() / config.num_mel_bins;
            let mel_shape = vec![1, config.num_mel_bins, num_frames];

            let mel = CandleTensor::from_vec(mel_data, mel_shape, &device)
                .map_err(|e| InferError::TensorError(format!("Failed to create mel tensor: {}", e)))?;

            // Create fresh TokenDecoder with clean KV-cache
            // This allows parallel transcriptions without state conflicts
            let mut decoder = TokenDecoder::new(
                (*model).clone(),
                (*tokenizer).clone(),
                &device,
                &config,
            )?;

            // Run decoding
            decoder.run(&mel)
        })
        .await
        .map_err(|e| InferError::Runtime(format!("Task join error: {}", e)))??;

        Ok(text)
    }
}
