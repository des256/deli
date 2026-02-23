use {
    crate::error::{InferError, Result},
    super::snapshot::{self, StateSnapshot},
    onnx::{Session, Value},
    rand_distr::{Distribution, Normal},
    std::collections::HashMap,
};

// Constants matching the reference default_parameters.py
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_LSD_STEPS: usize = 1;
const DEFAULT_EOS_THRESHOLD: f32 = -4.0;
#[cfg(test)]
const SAMPLE_RATE: usize = 24000;
const LATENT_DIM: usize = 32;
const CONDITIONING_DIM: usize = 1024;

/// Core inference engine for Pocket TTS.
///
/// Orchestrates 5 ONNX models:
/// - text_conditioner: token_ids → embeddings
/// - flow_lm_main: stateful transformer (18 KV-cache states)
/// - flow_lm_flow: stateless flow matching ODE solver
/// - mimi_encoder: audio → latents for voice cloning
/// - mimi_decoder: stateful neural codec (56 states)
pub(crate) struct PocketCore {
    text_conditioner: Session,
    flow_main: Session,
    flow_step: Session,
    mimi_encoder: Session,
    mimi_decoder: Session,

    // State tensor names for flow_lm_main
    flow_state_names: Vec<String>,
    flow_state_output_names: Vec<String>,

    // State tensor names for mimi_decoder
    mimi_state_names: Vec<String>,
    mimi_state_output_names: Vec<String>,

    // Current state tensors
    flow_states: Vec<Value>,
    mimi_states: Vec<Value>,

    // Voice-conditioned state snapshot (raw bytes + metadata)
    voice_snapshot: Option<StateSnapshot>,

    // Previous latent for AR loop
    prev_latent: Option<Vec<f32>>,
}

impl PocketCore {
    /// Create a new PocketCore instance
    ///
    /// Loads all 5 sessions and discovers state tensor names.
    pub fn new(
        text_conditioner: Session,
        flow_main: Session,
        flow_step: Session,
        mimi_encoder: Session,
        mimi_decoder: Session,
    ) -> Result<Self> {
        // Discover flow_lm_main state tensor names
        let flow_state_names = Self::discover_states(&flow_main, &["sequence", "text_embeddings"])?;
        let flow_state_output_names: Vec<String> = flow_state_names
            .iter()
            .map(|name| format!("out_{}", name))
            .collect();

        // Discover mimi_decoder state tensor names
        let mimi_state_names = Self::discover_states(&mimi_decoder, &["latent"])?;
        let mimi_state_output_names: Vec<String> = mimi_state_names
            .iter()
            .map(|name| format!("out_{}", name))
            .collect();

        // Initialize state tensors
        let flow_states = Self::initialize_states(&flow_main, &flow_state_names)?;
        let mimi_states = Self::initialize_states(&mimi_decoder, &mimi_state_names)?;

        Ok(PocketCore {
            text_conditioner,
            flow_main,
            flow_step,
            mimi_encoder,
            mimi_decoder,
            flow_state_names,
            flow_state_output_names,
            mimi_state_names,
            mimi_state_output_names,
            flow_states,
            mimi_states,
            voice_snapshot: None,
            prev_latent: None,
        })
    }

    /// Discover state tensor names from a session, excluding main inputs
    fn discover_states(session: &Session, exclude: &[&str]) -> Result<Vec<String>> {
        let input_count = session.input_count().map_err(|e| {
            InferError::Runtime(format!("Failed to get input count: {}", e))
        })?;

        let mut state_names = Vec::new();

        for i in 0..input_count {
            let name = session.input_name(i).map_err(|e| {
                InferError::Runtime(format!("Failed to get input name {}: {}", i, e))
            })?;

            if !exclude.contains(&name.as_str()) {
                state_names.push(name);
            }
        }

        Ok(state_names)
    }

    /// Initialize state tensors with correct default values.
    ///
    /// - Bool states → true (StreamingConv1d "first" flags need replicate-padding)
    /// - Float/Int64 → zeros
    ///
    /// Note: The reference Python uses NaN for KV caches, but the mimi decoder's
    /// ring-buffer attention reads ALL cache positions (including unwritten ones).
    /// ONNX Runtime propagates NaN through softmax (unlike PyTorch which treats
    /// it as -inf), so we use zeros for ONNX compatibility.
    fn initialize_states(session: &Session, state_names: &[String]) -> Result<Vec<Value>> {
        let input_count = session.input_count().map_err(|e| {
            InferError::Runtime(format!("Failed to get input count: {}", e))
        })?;

        // Build name→index map
        let mut name_to_index = HashMap::new();
        for i in 0..input_count {
            let name = session.input_name(i).map_err(|e| {
                InferError::Runtime(format!("Failed to get input name: {}", e))
            })?;
            name_to_index.insert(name, i);
        }

        let mut states = Vec::new();

        for state_name in state_names {
            let index = *name_to_index.get(state_name).ok_or_else(|| {
                InferError::Runtime(format!("State input '{}' not found", state_name))
            })?;

            let shape = session.input_shape(index).map_err(|e| {
                InferError::Runtime(format!("Failed to get shape for {}: {}", state_name, e))
            })?;

            let elem_type = session.input_element_type(index).map_err(|e| {
                InferError::Runtime(format!("Failed to get element type for {}: {}", state_name, e))
            })?;

            // Handle empty tensors (shape [0]) separately
            let tensor = if shape == [0] {
                Value::from_slice::<f32>(&[0], &[])?
            } else {
                match elem_type {
                    onnx::ffi::ONNXTensorElementDataType::Float => Value::zeros::<f32>(&shape),
                    onnx::ffi::ONNXTensorElementDataType::Int64 => Value::zeros::<i64>(&shape),
                    // Bool states are StreamingConv1d "first" flags.
                    // Must be true initially so replicate-padding works on first frame.
                    onnx::ffi::ONNXTensorElementDataType::Bool => {
                        let resolved: Vec<usize> = shape.iter()
                            .map(|&d| if d < 0 { 1 } else { d as usize })
                            .collect();
                        let total: usize = resolved.iter().product();
                        let true_data = vec![true; total];
                        Value::from_slice::<bool>(&resolved, &true_data)
                    }
                    _ => return Err(InferError::Runtime(format!(
                        "Unsupported element type {:?} for state {}",
                        elem_type, state_name
                    ))),
                }
                .map_err(|e| {
                    InferError::Runtime(format!("Failed to create initial tensor for {}: {}", state_name, e))
                })?
            };

            states.push(tensor);
        }

        Ok(states)
    }

    /// Run text conditioner: token_ids → embeddings
    pub fn run_text_conditioner(&mut self, token_ids: &[i64]) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        let tokens = Value::from_slice::<i64>(&[1, seq_len], token_ids)?;

        let outputs = self.text_conditioner
            .run(&[("token_ids", &tokens)], &["embeddings"])
            .map_err(|e| InferError::Onnx(e.to_string()))?;

        let embeddings = outputs[0].extract_tensor::<f32>().map_err(|e| {
            InferError::Runtime(format!("Failed to extract embeddings: {}", e))
        })?;

        Ok(embeddings.to_vec())
    }

    /// Encode voice audio to latents via mimi_encoder
    ///
    /// Input: audio samples (f32), sample_rate
    /// Output: latent embeddings [1, T, 1024]
    pub fn encode_voice(&mut self, audio: &[f32], _sample_rate: usize) -> Result<Vec<f32>> {
        let audio_len = audio.len();
        let audio_tensor = Value::from_slice::<f32>(&[1, 1, audio_len], audio)?;

        let outputs = self.mimi_encoder
            .run(&[("audio", &audio_tensor)], &["latents"])
            .map_err(|e| InferError::Onnx(e.to_string()))?;

        let latents = outputs[0].extract_tensor::<f32>().map_err(|e| {
            InferError::Runtime(format!("Failed to extract latents: {}", e))
        })?;

        Ok(latents.to_vec())
    }

    /// Condition voice: feed voice latents into flow_main to update KV-cache.
    ///
    /// Called FIRST (before text conditioning) per the reference implementation.
    /// Voice latents go into the text_embeddings input with an empty sequence.
    pub fn condition_voice(&mut self, voice_latents: &[f32], latent_frames: usize) -> Result<()> {
        let empty_seq = Value::from_slice::<f32>(&[1, 0, LATENT_DIM], &[])?;
        let text_emb = Value::from_slice::<f32>(&[1, latent_frames, CONDITIONING_DIM], voice_latents)?;

        let mut inputs = vec![("sequence", &empty_seq), ("text_embeddings", &text_emb)];
        for (i, state) in self.flow_states.iter().enumerate() {
            inputs.push((&self.flow_state_names[i], state));
        }

        let output_names: Vec<&str> = self.flow_state_output_names.iter().map(|s| s.as_str()).collect();
        let outputs = self.flow_main
            .run(&inputs, &output_names)
            .map_err(|e| InferError::Onnx(e.to_string()))?;

        self.flow_states = outputs;
        Ok(())
    }

    /// Condition text: feed text embeddings into flow_main.
    ///
    /// Called SECOND (after voice conditioning) per the reference implementation.
    pub fn condition_text(&mut self, embeddings: &[f32], seq_len: usize) -> Result<()> {
        let empty_seq = Value::from_slice::<f32>(&[1, 0, LATENT_DIM], &[])?;
        let text_emb = Value::from_slice::<f32>(&[1, seq_len, CONDITIONING_DIM], embeddings)?;

        let mut inputs = vec![("sequence", &empty_seq), ("text_embeddings", &text_emb)];
        for (i, state) in self.flow_states.iter().enumerate() {
            inputs.push((&self.flow_state_names[i], state));
        }

        let output_names: Vec<&str> = self.flow_state_output_names.iter().map(|s| s.as_str()).collect();
        let outputs = self.flow_main
            .run(&inputs, &output_names)
            .map_err(|e| InferError::Onnx(e.to_string()))?;

        self.flow_states = outputs;
        Ok(())
    }

    /// Generate one AR step: flow_main → LSD decode → latent
    ///
    /// Returns (latent_32d, is_eos)
    pub fn generate_step(&mut self) -> Result<(Vec<f32>, bool)> {
        // Sequence input: [1, 1, 32]
        // First step: NaN, subsequent steps: prev_latent
        let sequence_data: Vec<f32> = if let Some(ref prev) = self.prev_latent {
            prev.clone()
        } else {
            vec![f32::NAN; LATENT_DIM]
        };
        let sequence = Value::from_slice::<f32>(&[1, 1, LATENT_DIM], &sequence_data)?;

        // Empty text_embeddings [1, 0, 1024]
        let text_emb = Value::from_slice::<f32>(&[1, 0, CONDITIONING_DIM], &[])?;

        // Build inputs
        let mut inputs = vec![("sequence", &sequence), ("text_embeddings", &text_emb)];
        for (i, state) in self.flow_states.iter().enumerate() {
            inputs.push((&self.flow_state_names[i], state));
        }

        // Output names: conditioning, eos_logit, + all out_states
        let mut output_names = vec!["conditioning", "eos_logit"];
        output_names.extend(self.flow_state_output_names.iter().map(|s| s.as_str()));

        // Run flow_main
        let outputs = self.flow_main
            .run(&inputs, &output_names)
            .map_err(|e| InferError::Onnx(e.to_string()))?;

        // Extract conditioning and eos_logit (clone to avoid borrow conflicts)
        let conditioning = outputs[0].extract_tensor::<f32>().map_err(|e| {
            InferError::Runtime(format!("Failed to extract conditioning: {}", e))
        })?.to_vec();
        let eos_logit_tensor = outputs[1].extract_tensor::<f32>().map_err(|e| {
            InferError::Runtime(format!("Failed to extract eos_logit: {}", e))
        })?;
        let eos_logit = eos_logit_tensor[0];

        // Update states (skip first 2 outputs: conditioning, eos_logit)
        self.flow_states = outputs.into_iter().skip(2).collect();

        // LSD decode
        let latent = self.lsd_decode(&conditioning)?;

        // Store prev_latent for next step
        self.prev_latent = Some(latent.clone());

        // Check EOS
        let is_eos = eos_logit > DEFAULT_EOS_THRESHOLD;
        base::log_debug!("EOS logit: {}, threshold: {}, is_eos: {}", eos_logit, DEFAULT_EOS_THRESHOLD, is_eos);

        Ok((latent, is_eos))
    }

    /// LSD (Lagrangian Self Distillation) decode via flow_step
    ///
    /// Refines latent from noise using flow matching ODE solver.
    /// Reference: std = sqrt(temp), no noise clamping, 1 step default.
    fn lsd_decode(&mut self, conditioning: &[f32]) -> Result<Vec<f32>> {
        let num_steps = DEFAULT_LSD_STEPS;
        let temperature = DEFAULT_TEMPERATURE;

        // Initialize x ~ N(0, sqrt(temp)) — no clamping per reference defaults
        let mut rng = rand::thread_rng();
        let std = (temperature as f64).sqrt();
        let normal = Normal::new(0.0, std).map_err(|e| {
            InferError::Runtime(format!("Failed to create normal distribution: {}", e))
        })?;
        let mut x: Vec<f32> = (0..LATENT_DIM)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        // Iterative refinement
        for i in 0..num_steps {
            let s = i as f32 / num_steps as f32;
            let t = (i + 1) as f32 / num_steps as f32;

            // Prepare inputs
            let c_tensor = Value::from_slice::<f32>(&[1, CONDITIONING_DIM], conditioning)?;
            let s_tensor = Value::from_slice::<f32>(&[1, 1], &[s])?;
            let t_tensor = Value::from_slice::<f32>(&[1, 1], &[t])?;
            let x_tensor = Value::from_slice::<f32>(&[1, LATENT_DIM], &x)?;

            // Run flow_step
            let outputs = self.flow_step
                .run(&[
                    ("c", &c_tensor),
                    ("s", &s_tensor),
                    ("t", &t_tensor),
                    ("x", &x_tensor),
                ], &["flow_dir"])
                .map_err(|e| InferError::Onnx(e.to_string()))?;

            let flow_dir = outputs[0].extract_tensor::<f32>().map_err(|e| {
                InferError::Runtime(format!("Failed to extract flow_dir: {}", e))
            })?;

            // Update x: x += flow_dir / N
            for j in 0..LATENT_DIM {
                x[j] += flow_dir[j] / num_steps as f32;
            }
        }

        Ok(x)
    }

    /// Decode latent to audio via mimi_decoder
    ///
    /// Input: latent [32]
    /// Output: audio samples [samples]
    pub fn decode_audio(&mut self, latent: &[f32]) -> Result<Vec<f32>> {
        let latent_tensor = Value::from_slice::<f32>(&[1, 1, LATENT_DIM], latent)?;

        // Build inputs: latent + all mimi states
        let mut inputs = vec![("latent", &latent_tensor)];
        for (i, state) in self.mimi_states.iter().enumerate() {
            inputs.push((&self.mimi_state_names[i], state));
        }

        // Output names: audio_frame + all out_states
        let mut output_names = vec!["audio_frame"];
        output_names.extend(self.mimi_state_output_names.iter().map(|s| s.as_str()));

        // Run mimi_decoder
        let outputs = self.mimi_decoder
            .run(&inputs, &output_names)
            .map_err(|e| InferError::Onnx(e.to_string()))?;

        // Extract audio (clone to avoid borrow conflicts)
        let audio = outputs[0].extract_tensor::<f32>().map_err(|e| {
            InferError::Runtime(format!("Failed to extract audio: {}", e))
        })?.to_vec();

        // Update states (skip first output: audio_frame)
        self.mimi_states = outputs.into_iter().skip(1).collect();

        Ok(audio)
    }

    /// Snapshot voice-conditioned state for reuse across utterances.
    ///
    /// Captures flow_states and mimi_states as raw bytes so they can be
    /// restored for each new utterance without re-running voice conditioning.
    pub fn snapshot_state(&mut self) -> Result<()> {
        let flow_snapshots = snapshot::snapshot_values(&self.flow_states)?;
        let mimi_snapshots = snapshot::snapshot_values(&self.mimi_states)?;
        self.voice_snapshot = Some(StateSnapshot {
            flow_states: flow_snapshots,
            mimi_states: mimi_snapshots,
        });
        Ok(())
    }

    /// Restore state from voice-conditioned snapshot
    fn restore_snapshot(&mut self) -> Result<()> {
        let snap = self.voice_snapshot.as_ref().ok_or_else(|| {
            InferError::Runtime("No voice snapshot available".to_string())
        })?;
        self.flow_states = snapshot::restore_values(&snap.flow_states)?;
        self.mimi_states = snapshot::restore_values(&snap.mimi_states)?;
        Ok(())
    }

    /// Reset for new utterance.
    ///
    /// Restores voice-conditioned flow states from snapshot, and
    /// reinitializes mimi decoder states fresh. The mimi decoder is
    /// independent of voice/text conditioning and needs a clean state
    /// for each utterance (matching the reference `init_states(mimi)`).
    pub fn reset_for_utterance(&mut self) -> Result<()> {
        self.prev_latent = None;
        // Restore voice-conditioned flow states from snapshot
        let snap = self.voice_snapshot.as_ref().ok_or_else(|| {
            InferError::Runtime("No voice snapshot available".to_string())
        })?;
        self.flow_states = snapshot::restore_values(&snap.flow_states)?;
        // Reinitialize mimi states fresh (not from snapshot)
        self.mimi_states = Self::initialize_states(&self.mimi_decoder, &self.mimi_state_names)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(SAMPLE_RATE, 24000);
        assert_eq!(LATENT_DIM, 32);
        assert_eq!(CONDITIONING_DIM, 1024);
        assert_eq!(DEFAULT_LSD_STEPS, 1);
        assert_eq!(DEFAULT_TEMPERATURE, 0.7);
    }

    #[test]
    fn test_pocket_core_struct_exists() {
        // Compile-time verification that PocketCore has the expected structure
        // Cannot instantiate without real ONNX models, but we can verify the type exists
        fn assert_send<T: Send>() {}
        assert_send::<PocketCore>();
    }

    #[test]
    fn test_snapshot_struct_exists() {
        // Verify StateSnapshot compiles (TensorSnapshot is in snapshot module)
        fn assert_snapshot_fields() {
            let _: fn() -> StateSnapshot;
        }
        assert_snapshot_fields();
    }
}
