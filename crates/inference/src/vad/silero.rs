use {
    crate::error::{InferError, Result},
    std::sync::Arc,
};

const SILERO_VAD_MODEL_PATH: &str = "data/silero/silero_vad.onnx";

const SAMPLE_RATE: i64 = 16000;
const FRAME_SIZE: usize = 512;
const CONTEXT_SIZE: usize = 64;

pub struct SileroVad {
    session: onnx::Session,
    state: onnx::Value,
    context: Vec<f32>,
    sample_rate: i64,
}

impl SileroVad {
    /// Create a new SileroVad instance from an ONNX session
    ///
    /// The session should be loaded from a Silero VAD v5 model file.
    /// State and context are initialized to zeros.
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: &onnx::Executor) -> Result<Self> {
        let session = onnx.create_session(executor, SILERO_VAD_MODEL_PATH)?;
        // Initialize state tensor to zeros [2, 1, 128]
        let state = onnx::Value::zeros::<f32>(&onnx, &[2, 1, 128]).map_err(|e| {
            InferError::Runtime(format!("Failed to initialize VAD state tensor: {}", e))
        })?;

        // Initialize context buffer to zeros (64 samples)
        let context = vec![0.0f32; CONTEXT_SIZE];

        Ok(Self {
            session,
            state,
            context,
            sample_rate: SAMPLE_RATE,
        })
    }

    /// Process an audio frame and return speech probability
    ///
    /// # Arguments
    /// * `audio_frame` - Audio samples (must be exactly 512 samples, f32, normalized to [-1, 1])
    ///
    /// # Returns
    /// Speech probability in range [0.0, 1.0]
    pub fn process(&mut self, audio_frame: &[f32]) -> Result<f32> {
        // Validate frame size
        if audio_frame.len() != FRAME_SIZE {
            return Err(InferError::Runtime(format!(
                "Invalid frame size: expected {} samples, got {}",
                FRAME_SIZE,
                audio_frame.len()
            )));
        }

        // Prepend context to audio frame: [context (64) + frame (512)] = 576 samples
        let mut input_with_context = Vec::with_capacity(CONTEXT_SIZE + FRAME_SIZE);
        input_with_context.extend_from_slice(&self.context);
        input_with_context.extend_from_slice(audio_frame);

        // Create input tensor [1, 576]
        let input_tensor = onnx::Value::from_slice::<f32>(
            &self.session.onnx,
            &[1, CONTEXT_SIZE + FRAME_SIZE],
            &input_with_context,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create input tensor: {}", e)))?;

        // Create sample rate tensor [1]
        let sr_tensor =
            onnx::Value::from_slice::<i64>(&self.session.onnx, &[1], &[self.sample_rate])
                .map_err(|e| InferError::Runtime(format!("Failed to create SR tensor: {}", e)))?;

        // Run inference
        let inputs = [
            ("input", &input_tensor),
            ("state", &self.state),
            ("sr", &sr_tensor),
        ];

        let outputs = self
            .session
            .run(&inputs, &["output", "stateN"])
            .map_err(|e| InferError::Runtime(format!("VAD inference failed: {}", e)))?;

        // Extract output probability [1, 1] - copy data so we can take ownership of outputs
        let prob_data = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract output: {}", e)))?
            .to_vec();

        let probability = *prob_data
            .first()
            .ok_or_else(|| InferError::Runtime("Empty output tensor".to_string()))?;

        // Update state tensor with new state (take ownership from outputs vector)
        self.state = outputs
            .into_iter()
            .nth(1)
            .ok_or_else(|| InferError::Runtime("Missing stateN output".to_string()))?;

        // Update context with last 64 samples from current frame
        self.context
            .copy_from_slice(&audio_frame[FRAME_SIZE - CONTEXT_SIZE..]);

        Ok(probability)
    }

    /// Reset VAD state to initial condition
    ///
    /// Zeros the internal state tensor and context buffer.
    pub fn reset(&mut self) -> Result<()> {
        // Re-initialize state tensor to zeros
        self.state = onnx::Value::zeros::<f32>(&self.session.onnx, &[2, 1, 128])
            .map_err(|e| InferError::Runtime(format!("Failed to reset VAD state: {}", e)))?;

        // Zero context buffer
        self.context.fill(0.0);

        Ok(())
    }
}
