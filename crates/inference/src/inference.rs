use {
    crate::*,
    base::*,
    std::{path::Path, sync::Arc},
};

const ONNX_VERSION: usize = 24;

#[derive(Debug)]
pub struct Inference {
    onnx: Arc<onnx::Onnx>,
}

impl Inference {
    pub fn new() -> Result<Self, InferError> {
        let onnx = match onnx::Onnx::new(ONNX_VERSION) {
            Ok(onnx) => onnx,
            Err(error) => return Err(InferError::Onnx(error.to_string())),
        };
        Ok(Self { onnx })
    }

    pub fn use_parakeet<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
    ) -> Result<
        (
            crate::asr::parakeet::ParakeetHandle<T>,
            crate::asr::parakeet::ParakeetListener<T>,
        ),
        InferError,
    > {
        crate::asr::parakeet::create(&self.onnx, &executor)
    }

    pub fn use_pocket<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        voice_path: impl AsRef<Path>,
        epoch: Epoch,
    ) -> Result<
        (
            crate::tts::pocket::PocketHandle<T>,
            crate::tts::pocket::PocketListener<T>,
        ),
        InferError,
    > {
        crate::tts::pocket::create(&self.onnx, &executor, voice_path.as_ref(), epoch)
    }

    pub fn use_phi3<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::phi3::Phi3Handle<T>,
            crate::llm::phi3::Phi3Listener<T>,
        ),
        InferError,
    > {
        crate::llm::phi3::create(&self.onnx, executor, epoch)
    }

    pub fn use_llama32<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::llama32::Llama32Handle<T>,
            crate::llm::llama32::Llama32Listener<T>,
        ),
        InferError,
    > {
        crate::llm::llama32::create(&self.onnx, executor, epoch)
    }

    pub fn use_gemma3<T: Clone + Send + 'static>(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<
        (
            crate::llm::gemma3::Gemma3Handle<T>,
            crate::llm::gemma3::Gemma3Listener<T>,
        ),
        InferError,
    > {
        crate::llm::gemma3::create(&self.onnx, executor, epoch)
    }

    pub fn use_silero(
        &self,
        executor: &onnx::Executor,
        sample_rate: usize,
    ) -> Result<crate::vad::Silero, InferError> {
        crate::vad::Silero::new(&self.onnx, &executor, sample_rate)
    }

    pub fn use_parakeet_diar(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::diar::parakeet::Sortformer, InferError> {
        crate::diar::parakeet::Sortformer::new(&self.onnx, executor)
    }

    pub fn onnx_session(&self, model_path: impl AsRef<Path>) -> Result<onnx::Session, InferError> {
        self.onnx
            .create_session(
                &onnx::Executor::Cpu,
                &onnx::OptimizationLevel::EnableAll,
                4,
                model_path.as_ref(),
            )
            .map_err(|e| InferError::Onnx(e.to_string()))
    }

    pub fn device(&self) -> &onnx::Executor {
        &onnx::Executor::Cpu
    }
}
