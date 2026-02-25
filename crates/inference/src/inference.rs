use {crate::*, std::path::Path, std::sync::Arc};

const ONNX_VERSION: usize = 24;

#[derive(Debug)]
pub struct Inference {
    onnx: Arc<onnx::Onnx>,
}

impl Inference {
    pub fn new() -> Result<Self, InferError> {
        let onnx = onnx::Onnx::new(ONNX_VERSION)?;
        Ok(Self { onnx })
    }

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

    pub fn use_phi3(&self, executor: &onnx::Executor) -> Result<crate::llm::Phi3, InferError> {
        crate::llm::Phi3::new(&self.onnx, executor)
    }

    pub fn use_kokoro(
        &self,
        executor: &onnx::Executor,
        voice_path: impl AsRef<Path>,
    ) -> Result<crate::tts::Kokoro, InferError> {
        crate::tts::Kokoro::new(&self.onnx, &executor, voice_path)
    }

    pub fn use_pocket_tts(
        &self,
        executor: &onnx::Executor,
        voice_path: impl AsRef<Path>,
    ) -> Result<crate::tts::pocket::PocketTts, InferError> {
        crate::tts::pocket::PocketTts::new(&self.onnx, &executor, voice_path)
    }

    pub fn use_sherpa(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::asr::sherpa::Sherpa, InferError> {
        crate::asr::sherpa::Sherpa::new(&self.onnx, &executor)
    }

    pub fn use_parakeet(
        &self,
        executor: &onnx::Executor,
    ) -> Result<crate::asr::parakeet::Parakeet, InferError> {
        crate::asr::parakeet::Parakeet::new(&self.onnx, &executor)
    }

    pub fn use_silero_vad(&self) -> Result<crate::vad::SileroVad, InferError> {
        crate::vad::SileroVad::new(&self.onnx, &executor)
    }

    pub fn use_parakeet_diar(&self) -> Result<crate::diar::parakeet::Sortformer, InferError> {
        crate::diar::parakeet::Sortformer::new(session, config)
    }
}
