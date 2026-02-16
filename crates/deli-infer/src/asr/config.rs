use serde::{Deserialize, Serialize};

/// Whisper model configuration.
///
/// Ported from candle-transformers.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub vocab_size: usize,
    pub max_target_positions: usize,
    pub decoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub suppress_tokens: Vec<u32>,
}

impl Config {
    /// Returns the hardcoded configuration for Whisper tiny.en model.
    pub fn tiny_en() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            // Suppress tokens: non-speech tokens to suppress during generation
            suppress_tokens: vec![
                1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91,
                92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303,
                1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600,
                4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907,
                13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724,
                22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282,
                49146, 50257, 50357, 50358, 50359, 50360, 50361
            ],
        }
    }
}

// Audio processing constants
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;  // seconds
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;  // 480000
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;  // 3000

// Decoding thresholds
pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
pub const LOGPROB_THRESHOLD: f64 = -1.0;
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

// Special token names (actual IDs will be looked up from tokenizer)
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";

// No-speech tokens for filtering
pub const NO_SPEECH_TOKENS: [u32; 2] = [220, 50257];
