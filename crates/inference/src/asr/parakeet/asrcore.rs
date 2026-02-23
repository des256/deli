use crate::error::{InferError, Result};
use onnx::Session;

const BLANK_ID: i64 = 1024;
const MAX_SYMBOLS_PER_STEP: usize = 10;
const VOCAB_SIZE: usize = 1025; // 1024 tokens + 1 blank
const ENCODER_DIM: usize = 1024;
const NUM_LAYERS: usize = 24;
const CACHE_CHANNEL_CONTEXT: usize = 70;
const CACHE_TIME_CONTEXT: usize = 8;
const DECODER_STATE_DIM: usize = 640;

/// Core ASR state for the streaming Parakeet FastConformer-Transducer.
///
/// Maintains both encoder caches (cross-chunk context) and decoder states
/// (prediction network) across chunks for true streaming inference.
pub(crate) struct AsrCore {
    pub(super) encoder: Session,
    pub(super) decoder_joint: Session,
    pub(super) tokens: Vec<String>,
    pub(super) blank_id: i64,
    // Decoder states (persist across chunks)
    pub(super) state1: onnx::Value,
    pub(super) state2: onnx::Value,
    pub(super) last_token: i64,
    // Encoder cache states (streaming cross-chunk context)
    cache_last_channel: onnx::Value,
    cache_last_time: onnx::Value,
    cache_last_channel_len: onnx::Value,
}

impl AsrCore {
    /// Create a new AsrCore with zero-initialized states
    pub fn new(encoder: Session, decoder_joint: Session, tokens: Vec<String>) -> Result<Self> {
        let state1 = zeros_f32(&[2, 1, DECODER_STATE_DIM as i64])?;
        let state2 = zeros_f32(&[2, 1, DECODER_STATE_DIM as i64])?;

        let cache_last_channel = zeros_f32(&[
            1,
            NUM_LAYERS as i64,
            CACHE_CHANNEL_CONTEXT as i64,
            ENCODER_DIM as i64,
        ])?;
        let cache_last_time = zeros_f32(&[
            1,
            NUM_LAYERS as i64,
            ENCODER_DIM as i64,
            CACHE_TIME_CONTEXT as i64,
        ])?;
        let cache_last_channel_len = onnx::Value::from_slice(&[1], &[0i64])
            .map_err(|e| {
                InferError::Runtime(format!("Failed to create cache_last_channel_len: {}", e))
            })?;

        Ok(Self {
            encoder,
            decoder_joint,
            tokens,
            blank_id: BLANK_ID,
            state1,
            state2,
            last_token: BLANK_ID,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        })
    }

    /// Reset all states for a new utterance
    pub fn reset(&mut self) -> Result<()> {
        self.state1 = zeros_f32(&[2, 1, DECODER_STATE_DIM as i64])?;
        self.state2 = zeros_f32(&[2, 1, DECODER_STATE_DIM as i64])?;
        self.last_token = BLANK_ID;
        self.cache_last_channel = zeros_f32(&[
            1,
            NUM_LAYERS as i64,
            CACHE_CHANNEL_CONTEXT as i64,
            ENCODER_DIM as i64,
        ])?;
        self.cache_last_time = zeros_f32(&[
            1,
            NUM_LAYERS as i64,
            ENCODER_DIM as i64,
            CACHE_TIME_CONTEXT as i64,
        ])?;
        self.cache_last_channel_len = onnx::Value::from_slice(&[1], &[0i64])
            .map_err(|e| {
                InferError::Runtime(format!("Failed to reset cache_last_channel_len: {}", e))
            })?;
        Ok(())
    }

    /// Decode a chunk of mel features into text.
    ///
    /// Features should be in channels-first layout: [128 * num_frames].
    /// Encoder caches and decoder states persist across chunks for streaming.
    pub fn decode_chunk(&mut self, features: &[f32], num_frames: usize) -> Result<String> {
        if features.len() != num_frames * 128 {
            return Err(InferError::Runtime(format!(
                "Features length {} does not match num_frames {} * 128",
                features.len(),
                num_frames
            )));
        }

        // Run encoder (caches are updated internally)
        let (encoder_out, encoder_out_len) = self.run_encoder(features, num_frames)?;

        // Greedy decode
        let token_ids = self.greedy_decode(&encoder_out, encoder_out_len)?;

        // Convert to text
        let text = self.tokens_to_text(&token_ids);
        Ok(text)
    }

    /// Run the streaming encoder with cache state
    fn run_encoder(&mut self, features: &[f32], num_frames: usize) -> Result<(Vec<f32>, usize)> {
        let processed_signal = onnx::Value::from_slice(&[1, 128, num_frames], features)
            .map_err(|e| InferError::Runtime(format!("Failed to create processed_signal: {}", e)))?;

        let processed_signal_length = onnx::Value::from_slice(&[1], &[num_frames as i64])
            .map_err(|e| {
                InferError::Runtime(format!("Failed to create processed_signal_length: {}", e))
            })?;

        // Single-speaker mode: pass zeros for speaker targets
        let spk_targets = onnx::Value::zeros::<f32>(&[1, num_frames as i64])
            .map_err(|e| InferError::Runtime(format!("Failed to create spk_targets: {e}")))?;
        let bg_spk_targets = onnx::Value::zeros::<f32>(&[1, num_frames as i64])
            .map_err(|e| InferError::Runtime(format!("Failed to create bg_spk_targets: {e}")))?;

        let mut outputs = self
            .encoder
            .run(
                &[
                    ("processed_signal", &processed_signal),
                    ("processed_signal_length", &processed_signal_length),
                    ("cache_last_channel", &self.cache_last_channel),
                    ("cache_last_time", &self.cache_last_time),
                    ("cache_last_channel_len", &self.cache_last_channel_len),
                    ("spk_targets", &spk_targets),
                    ("bg_spk_targets", &bg_spk_targets),
                ],
                &[
                    "encoded",
                    "encoded_len",
                    "cache_last_channel_next",
                    "cache_last_time_next",
                    "cache_last_channel_len_next",
                ],
            )
            .map_err(|e| InferError::Runtime(format!("Encoder inference failed: {}", e)))?;

        // Extract encoder output: [batch, 1024, T']
        let encoder_out_shape = outputs[0]
            .tensor_shape()
            .map_err(|e| InferError::Runtime(format!("Failed to get encoder output shape: {}", e)))?;
        let encoder_out_len = encoder_out_shape[2] as usize;

        let encoder_out_data = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract encoder output: {}", e)))?
            .to_vec();

        // Update encoder caches (take ownership from outputs vec)
        // outputs: [encoded, encoded_len, cache_channel, cache_time, cache_channel_len]
        self.cache_last_channel = outputs.remove(2);
        self.cache_last_time = outputs.remove(2); // was index 3, now 2
        self.cache_last_channel_len = outputs.remove(2); // was index 4, now 2

        Ok((encoder_out_data, encoder_out_len))
    }

    /// Perform greedy decoding frame-by-frame
    fn greedy_decode(&mut self, encoder_out: &[f32], num_frames: usize) -> Result<Vec<i64>> {
        let mut decoded_tokens = Vec::new();

        let mut encoder_frame = vec![0.0f32; ENCODER_DIM];

        for frame_idx in 0..num_frames {
            // Encoder output is [batch, 1024, T'] layout (channels-first).
            // Element [0, d, t] is at flat index d * num_frames + t.
            for d in 0..ENCODER_DIM {
                encoder_frame[d] = encoder_out[d * num_frames + frame_idx];
            }

            // Decoder-joint expects [B, T, D] — single frame is [1, 1, 1024]
            let encoder_outputs = onnx::Value::from_slice(&[1, 1, ENCODER_DIM], &encoder_frame)
                .map_err(|e| {
                    InferError::Runtime(format!("Failed to create encoder_outputs: {}", e))
                })?;

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                // Save states before decoder call (for rollback on blank)
                let state1_data = self
                    .state1
                    .extract_tensor::<f32>()
                    .map_err(|e| InferError::Runtime(format!("Failed to extract state1: {}", e)))?
                    .to_vec();
                let state2_data = self
                    .state2
                    .extract_tensor::<f32>()
                    .map_err(|e| InferError::Runtime(format!("Failed to extract state2: {}", e)))?
                    .to_vec();

                let logits = self.run_decoder_joint(&encoder_outputs)?;

                let predicted = argmax_masked(&logits, VOCAB_SIZE);

                if predicted == self.blank_id {
                    // Restore prediction network states on blank
                    self.state1 =
                        onnx::Value::from_slice(&[2, 1, DECODER_STATE_DIM], &state1_data)
                            .map_err(|e| {
                                InferError::Runtime(format!("Failed to restore state1: {}", e))
                            })?;
                    self.state2 =
                        onnx::Value::from_slice(&[2, 1, DECODER_STATE_DIM], &state2_data)
                            .map_err(|e| {
                                InferError::Runtime(format!("Failed to restore state2: {}", e))
                            })?;
                    break;
                }

                decoded_tokens.push(predicted);
                self.last_token = predicted;
            }
        }

        Ok(decoded_tokens)
    }

    /// Run decoder-joint with a single encoder frame and current state
    fn run_decoder_joint(&mut self, encoder_outputs: &onnx::Value) -> Result<Vec<f32>> {
        // targets: [1, 1] int64
        let targets = onnx::Value::from_slice(&[1, 1], &[self.last_token])
            .map_err(|e| InferError::Runtime(format!("Failed to create targets: {}", e)))?;

        let mut outputs = self
            .decoder_joint
            .run(
                &[
                    ("encoder_outputs", encoder_outputs),
                    ("targets", &targets),
                    ("input_states_1", &self.state1),
                    ("input_states_2", &self.state2),
                ],
                &["outputs", "prednet_lengths", "states_1", "states_2"],
            )
            .map_err(|e| InferError::Runtime(format!("Decoder-joint inference failed: {}", e)))?;

        // Extract logits: [1, ?, 1, 1025]
        let logits_data = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract logits: {}", e)))?
            .to_vec();

        // Update decoder states
        self.state1 = outputs.remove(2); // states_1
        self.state2 = outputs.remove(2); // states_2 (was index 3, now 2)

        Ok(logits_data)
    }

    /// Convert token IDs to text (SentencePiece: ▁ = space)
    fn tokens_to_text(&self, token_ids: &[i64]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx >= self.tokens.len() {
                    None
                } else {
                    Some(self.tokens[idx].replace('▁', " "))
                }
            })
            .collect::<String>()
            .trim_start()
            .to_string()
    }
}

fn zeros_f32(shape: &[i64]) -> Result<onnx::Value> {
    onnx::Value::zeros::<f32>(shape)
        .map_err(|e| InferError::Runtime(format!("Failed to create zero tensor: {e}")))
}

/// Find the index of the maximum value in a slice, restricted to valid range
fn argmax_masked(values: &[f32], max_idx: usize) -> i64 {
    let valid_range = &values[..max_idx.min(values.len())];
    valid_range
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(0)
}
