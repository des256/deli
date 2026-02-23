use {crate::error::InferError, candle_core::Device, onnx::Session, std::path::Path};

#[derive(Debug)]
enum OnnxDevice {
    Cpu,
    #[allow(dead_code)]
    Cuda(usize),
}

#[derive(Debug)]
pub struct Inference {
    device: Device,
    onnx_device: OnnxDevice,
}

impl Inference {
    pub fn cpu() -> Result<Self, InferError> {
        onnx::init()?;
        base::log_info!("Inference device: CPU");
        Ok(Self {
            device: Device::Cpu,
            onnx_device: OnnxDevice::Cpu,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self, InferError> {
        onnx::init()?;
        let device = Device::new_cuda(ordinal)?;
        if device.is_cuda() {
            base::log_info!("Inference device: CUDA (ordinal {})", ordinal);
        } else {
            base::log_warn!(
                "Inference device: requested CUDA ordinal {} but device reports non-CUDA",
                ordinal
            );
        }
        Ok(Self {
            device,
            onnx_device: OnnxDevice::Cuda(ordinal),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn use_pose_detector(
        &self,
        model_path: impl AsRef<Path>,
    ) -> Result<crate::pose_detector::PoseDetector, InferError> {
        crate::pose_detector::PoseDetector::new(model_path, self.device.clone())
    }

    pub fn use_whisper(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<crate::asr::Whisper, InferError> {
        crate::asr::Whisper::new(model_path, tokenizer_path, config_path, self.device.clone())
    }

    pub fn use_qwen3(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::llm::Qwen3, InferError> {
        crate::llm::Qwen3::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn use_phi3(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::llm::Phi3, InferError> {
        crate::llm::Phi3::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn use_llama(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::llm::Llama, InferError> {
        crate::llm::Llama::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn use_smollm2(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::llm::Smollm2, InferError> {
        crate::llm::Smollm2::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn onnx_session(&self, model_path: impl AsRef<Path>) -> Result<Session, InferError> {
        let path = model_path.as_ref();
        let session = match &self.onnx_device {
            OnnxDevice::Cpu => onnx::session_builder()?.with_cpu().commit_from_file(path)?,
            #[cfg(feature = "cuda")]
            OnnxDevice::Cuda(ordinal) => onnx::session_builder()?
                .with_cuda(*ordinal as i32)?
                .commit_from_file(path)?,
            #[cfg(not(feature = "cuda"))]
            OnnxDevice::Cuda(_) => {
                return Err(InferError::Runtime("CUDA feature not enabled".to_string()));
            }
        };
        Ok(session)
    }

    pub fn use_kokoro(
        &self,
        model_path: impl AsRef<Path>,
        voice_path: impl AsRef<Path>,
        espeak_data_path: Option<&str>,
    ) -> Result<crate::tts::Kokoro, InferError> {
        let session = self.onnx_session(model_path)?;
        crate::tts::Kokoro::new(session, voice_path, espeak_data_path)
    }

    pub fn use_streaming_asr<P: AsRef<Path>>(
        &self,
        encoder_path: P,
        decoder_path: P,
        joiner_path: P,
        tokens_path: P,
    ) -> Result<crate::asr::sherpa::Sherpa, InferError> {
        let encoder_session = self.onnx_session(&encoder_path)?;
        let decoder_session = self.onnx_session(&decoder_path)?;
        let joiner_session = self.onnx_session(&joiner_path)?;
        crate::asr::sherpa::Sherpa::new(
            encoder_session,
            decoder_session,
            joiner_session,
            tokens_path,
        )
    }
}
