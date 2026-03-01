use {crate::*, std::sync::Arc};

const SILERO_VAD_MODEL_PATH: &str = "data/vad/silero/silero_vad.onnx";

const FRAME_SIZE: usize = 512;
const CONTEXT_SIZE: usize = 64;

pub struct Silero {
    session: onnx::Session,
    state_tensor: onnx::Value,
    sample_rate_tensor: onnx::Value,
    input_tensor: onnx::Value,
}

impl Silero {
    pub fn new(
        onnx: &Arc<onnx::Onnx>,
        executor: &onnx::Executor,
        sample_rate: usize,
    ) -> Result<Self, InferError> {
        let session = onnx
            .create_session(
                executor,
                &onnx::OptimizationLevel::EnableAll,
                4,
                SILERO_VAD_MODEL_PATH,
            )
            .map_err(|e| InferError::Runtime(format!("Failed to create VAD session: {}", e)))?;
        let state_tensor = onnx::Value::zeros::<f32>(&onnx, &[2, 1, 128]).map_err(|e| {
            InferError::Runtime(format!("Failed to initialize VAD state tensor: {}", e))
        })?;
        let sample_rate_tensor =
            onnx::Value::from_slice::<i64>(&session.onnx, &[1], &[sample_rate as i64])
                .map_err(|e| InferError::Runtime(format!("Failed to create SR tensor: {}", e)))?;
        let input_tensor =
            onnx::Value::zeros::<f32>(&session.onnx, &[1, (CONTEXT_SIZE + FRAME_SIZE) as i64])
                .map_err(|e| {
                    InferError::Runtime(format!("Failed to create input tensor: {}", e))
                })?;
        Ok(Self {
            session,
            state_tensor,
            sample_rate_tensor,
            input_tensor,
        })
    }

    pub fn process(&mut self, sample: &[i16]) -> Result<f32, InferError> {
        if sample.len() != FRAME_SIZE {
            return Err(InferError::Runtime(format!(
                "Invalid frame size: expected {} samples, got {}",
                FRAME_SIZE,
                sample.len()
            )));
        }
        let slice = self.input_tensor.as_slice_mut::<f32>();
        let mut context = [0f32; CONTEXT_SIZE];
        context.copy_from_slice(&slice[FRAME_SIZE..]);
        slice[..CONTEXT_SIZE].copy_from_slice(&context);
        for i in 0..FRAME_SIZE {
            slice[CONTEXT_SIZE + i] = sample[i] as f32 / 32768.0;
        }
        let inputs = [
            ("input", &self.input_tensor),
            ("state", &self.state_tensor),
            ("sr", &self.sample_rate_tensor),
        ];
        let mut outputs = self
            .session
            .run(&inputs, &["output", "stateN"])
            .map_err(|e| InferError::Runtime(format!("VAD inference failed: {}", e)))?;
        let probability = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("Failed to extract output: {}", e)))?[0];
        self.state_tensor = outputs.split_off(1).remove(0);
        Ok(probability)
    }
}
