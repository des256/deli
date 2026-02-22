use {
    crate::InferError,
    candle_core::Device,
    onnx::Session,
    std::{path::Path, sync::OnceLock},
};

static ONNX_INIT: OnceLock<()> = OnceLock::new();

fn ensure_onnx_init() {
    ONNX_INIT.get_or_init(|| {
        if let Err(e) = onnx::init() {
            base::log_error!("ONNX Runtime init failed: {}", e);
        }
    });
}

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
    pub fn cpu() -> Self {
        ensure_onnx_init();
        base::log_info!("Inference device: CPU");
        Self {
            device: Device::Cpu,
            onnx_device: OnnxDevice::Cpu,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self, InferError> {
        ensure_onnx_init();
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
    ) -> Result<crate::PoseDetector, InferError> {
        crate::PoseDetector::new(model_path, self.device.clone())
    }

    pub fn use_whisper(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<crate::Whisper, InferError> {
        crate::Whisper::new(model_path, tokenizer_path, config_path, self.device.clone())
    }

    pub fn use_qwen3(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::Qwen3, InferError> {
        crate::Qwen3::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn use_phi3(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::Phi3, InferError> {
        crate::Phi3::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn use_llama(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::Llama, InferError> {
        crate::Llama::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn use_smollm2(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::Smollm2, InferError> {
        crate::Smollm2::new(model_path, tokenizer_path, self.device.clone())
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
}
