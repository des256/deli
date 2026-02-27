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

    pub fn use_phi3(
        &self,
        executor: &onnx::Executor,
        epoch: Epoch,
    ) -> Result<(crate::llm::phi3::Phi3Handle, crate::llm::phi3::Phi3Listener), InferError> {
        crate::llm::phi3::create(&self.onnx, executor, epoch)
    }

    pub fn use_silero(
        &self,
        executor: &onnx::Executor,
        sample_rate: usize,
    ) -> Result<crate::vad::Silero, InferError> {
        crate::vad::Silero::new(&self.onnx, &executor, sample_rate)
    }

    /*
    pub fn use_smollm3(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::llm::Smollm3, InferError> {
        crate::llm::Smollm3::new(&self.onnx, executor)
    }

    pub fn use_llama32(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::llm::Llama32, InferError> {
        crate::llm::Llama32::new(&self.onnx, executor)
    }

    pub fn use_gemma3(&self, executor: &onnx::Executor) -> Result<crate::llm::Gemma3, InferError> {
        crate::llm::Gemma3::new(&self.onnx, executor)
    }

    pub fn use_kokoro(
        &self,
        executor: &onnx::Executor,
        voice_path: impl AsRef<Path>,
    ) -> Result<crate::tts::Kokoro, InferError> {
        crate::tts::Kokoro::new(&self.onnx, &executor, voice_path)
    }

    pub fn use_sherpa(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::asr::sherpa::Sherpa, InferError> {
        crate::asr::sherpa::Sherpa::new(&self.onnx, &executor)
    }

    pub fn use_parakeet_diar(&self) -> Result<crate::diar::parakeet::Sortformer, InferError> {
        crate::diar::parakeet::Sortformer::new(session, config)
    }
    */
}
