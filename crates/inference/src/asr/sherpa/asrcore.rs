use crate::error::{InferError, Result};
use onnx::Session;

/// Core ASR state (sessions and decoding state)
pub(crate) struct AsrCore {
    pub(super) encoder: Session,
    pub(super) decoder: Session,
    pub(super) joiner: Session,
    pub(super) tokens: Vec<String>,
    pub(super) context: Vec<i64>,
    pub(super) encoder_states: Vec<onnx::Value>,
    pub(super) encoder_state_names: Vec<String>,
}

impl AsrCore {
    /// Decode a chunk of log-mel features into text.
    pub fn decode_chunk(&mut self, features: &[f32], num_frames: usize) -> Result<String> {
        if features.len() != num_frames * 80 {
            return Err(InferError::Runtime(format!(
                "Features length {} does not match num_frames {} * 80",
                features.len(),
                num_frames
            )));
        }

        let (encoder_out, encoder_out_len) = self.run_encoder(features, num_frames)?;
        let token_ids = self.greedy_decode(&encoder_out, encoder_out_len)?;
        let text = self.tokens_to_text(&token_ids);
        Ok(text)
    }

    /// Run the encoder with state carry.
    fn run_encoder(&mut self, features: &[f32], num_frames: usize) -> Result<(Vec<f32>, usize)> {
        let x_shape = [1, num_frames, 80];
        let x = onnx::Value::from_slice(&x_shape, features).map_err(|e| {
            InferError::Runtime(format!("Failed to create encoder input tensor: {}", e))
        })?;

        // State output names: "new_" prefix on each state input name
        let state_output_names: Vec<String> = self
            .encoder_state_names
            .iter()
            .map(|n| format!("new_{}", n))
            .collect();

        let mut inputs: Vec<(&str, &onnx::Value)> = vec![("x", &x)];
        for (i, state) in self.encoder_states.iter().enumerate() {
            inputs.push((&self.encoder_state_names[i], state));
        }

        let mut output_names: Vec<&str> = vec!["encoder_out"];
        for name in &state_output_names {
            output_names.push(name);
        }

        let outputs = self
            .encoder
            .run(&inputs, &output_names)
            .map_err(|e| InferError::Runtime(format!("Encoder inference failed: {}", e)))?;

        // Determine output frame count from tensor shape: [1, T, encoder_dim]
        let encoder_out_shape = outputs[0].tensor_shape().map_err(|e| {
            InferError::Runtime(format!("Failed to get encoder output shape: {}", e))
        })?;
        let encoder_out_len = encoder_out_shape[1] as usize;

        let encoder_out_data = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract encoder output: {}", e)))?
            .to_vec();

        let new_states: Vec<_> = outputs.into_iter().skip(1).collect();
        self.encoder_states = new_states;

        Ok((encoder_out_data, encoder_out_len))
    }

    /// Perform greedy decoding frame-by-frame (max_sym_per_frame = 1).
    fn greedy_decode(&mut self, encoder_out: &[f32], num_frames: usize) -> Result<Vec<i64>> {
        let mut decoded_tokens = Vec::new();
        let encoder_dim = encoder_out.len() / num_frames;

        // Cache decoder output — only recompute when context changes (non-blank token)
        let mut decoder_out = self.run_decoder()?;

        for frame_idx in 0..num_frames {
            let frame_start = frame_idx * encoder_dim;
            let frame_end = frame_start + encoder_dim;
            let encoder_frame = &encoder_out[frame_start..frame_end];

            let logits = self.run_joiner(encoder_frame, &decoder_out)?;
            let predicted_token = argmax(&logits);

            if predicted_token != 0 {
                decoded_tokens.push(predicted_token);
                self.context.rotate_left(1);
                let last_idx = self.context.len() - 1;
                self.context[last_idx] = predicted_token;
                // Context changed, recompute decoder output
                decoder_out = self.run_decoder()?;
            }
        }

        Ok(decoded_tokens)
    }

    fn run_decoder(&mut self) -> Result<Vec<f32>> {
        let y_shape = [1, self.context.len()];
        let y = onnx::Value::from_slice(&y_shape, &self.context).map_err(|e| {
            InferError::Runtime(format!("Failed to create decoder input tensor: {}", e))
        })?;

        let mut outputs = self
            .decoder
            .run(&[("y", &y)], &["decoder_out"])
            .map_err(|e| InferError::Runtime(format!("Decoder inference failed: {}", e)))?;

        let value = outputs.remove(0);
        let decoder_out = value
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract decoder output: {}", e)))?
            .to_vec();

        Ok(decoder_out)
    }

    fn run_joiner(&mut self, encoder_frame: &[f32], decoder_out: &[f32]) -> Result<Vec<f32>> {
        let encoder_dim = encoder_frame.len();
        let encoder_input =
            onnx::Value::from_slice(&[1, encoder_dim], encoder_frame).map_err(|e| {
                InferError::Runtime(format!("Failed to create joiner encoder input: {}", e))
            })?;

        let decoder_dim = decoder_out.len();
        let decoder_input =
            onnx::Value::from_slice(&[1, decoder_dim], decoder_out).map_err(|e| {
                InferError::Runtime(format!("Failed to create joiner decoder input: {}", e))
            })?;

        let mut outputs = self
            .joiner
            .run(
                &[
                    ("encoder_out", &encoder_input),
                    ("decoder_out", &decoder_input),
                ],
                &["logit"],
            )
            .map_err(|e| InferError::Runtime(format!("Joiner inference failed: {}", e)))?;

        let value = outputs.remove(0);
        let logits = value
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract joiner logits: {}", e)))?
            .to_vec();

        Ok(logits)
    }

    /// Convert token IDs to text (replaces `▁` with space, trims leading space).
    fn tokens_to_text(&self, token_ids: &[i64]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                if id as usize >= self.tokens.len() {
                    None
                } else {
                    Some(self.tokens[id as usize].replace('▁', " "))
                }
            })
            .collect::<String>()
            .trim_start()
            .to_string()
    }

    /// Discover encoder state tensor names by inspecting model inputs.
    ///
    /// All inputs except `x` are treated as state tensors.
    /// Corresponding outputs use `new_` prefix (e.g., `cached_key_0` → `new_cached_key_0`).
    pub(super) fn discover_encoder_states(encoder: &Session) -> Result<Vec<String>> {
        let input_count = encoder.input_count().map_err(|e| {
            InferError::Runtime(format!("Failed to get encoder input count: {}", e))
        })?;

        let mut state_names = Vec::new();

        for i in 0..input_count {
            let name = encoder.input_name(i).map_err(|e| {
                InferError::Runtime(format!("Failed to get encoder input name {}: {}", i, e))
            })?;

            if name != "x" {
                state_names.push(name);
            }
        }

        if state_names.is_empty() {
            return Err(InferError::Runtime(
                "Encoder model has no state inputs (expected inputs besides 'x')".to_string(),
            ));
        }

        Ok(state_names)
    }

    /// Initialize encoder states with zero-filled tensors matching model metadata.
    pub(super) fn initialize_encoder_states(
        encoder: &Session,
        state_names: &[String],
    ) -> Result<Vec<onnx::Value>> {
        let input_count = encoder.input_count().map_err(|e| {
            InferError::Runtime(format!("Failed to get encoder input count: {}", e))
        })?;

        // Build name→index map
        let mut name_to_index = std::collections::HashMap::new();
        for i in 0..input_count {
            let name = encoder.input_name(i).map_err(|e| {
                InferError::Runtime(format!("Failed to get encoder input name: {}", e))
            })?;
            name_to_index.insert(name, i);
        }

        let mut states = Vec::new();

        for state_name in state_names {
            let index = *name_to_index.get(state_name).ok_or_else(|| {
                InferError::Runtime(format!("State input '{}' not found in encoder", state_name))
            })?;

            let shape = encoder.input_shape(index).map_err(|e| {
                InferError::Runtime(format!("Failed to get shape for {}: {}", state_name, e))
            })?;

            let elem_type = encoder.input_element_type(index).map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to get element type for {}: {}",
                    state_name, e
                ))
            })?;

            let zero_tensor = match elem_type {
                onnx::ffi::ONNXTensorElementDataType::Int64 => onnx::Value::zeros::<i64>(&shape),
                _ => onnx::Value::zeros::<f32>(&shape),
            }
            .map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to create zero tensor for {}: {}",
                    state_name, e
                ))
            })?;

            states.push(zero_tensor);
        }

        Ok(states)
    }
}

/// Find the index of the maximum value in a slice.
fn argmax(values: &[f32]) -> i64 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(0)
}
