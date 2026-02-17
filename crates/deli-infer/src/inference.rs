use crate::InferError;
use candle_core::Device;
use ort::session::Session as OrtSession;
use std::path::Path;
use std::sync::OnceLock;

static ORT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_ort_init() {
    ORT_INIT.get_or_init(|| {
        let _ = ort::init().commit();
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
        ensure_ort_init();
        Self {
            device: Device::Cpu,
            onnx_device: OnnxDevice::Cpu,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self, InferError> {
        ensure_ort_init();
        let device = Device::new_cuda(ordinal)?;
        Ok(Self {
            device,
            onnx_device: OnnxDevice::Cuda(ordinal),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn use_pose_detector(&self, model_path: impl AsRef<Path>) -> Result<crate::PoseDetector, InferError> {
        crate::PoseDetector::new(model_path, self.device.clone())
    }

    pub fn use_speech_recognizer(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<crate::SpeechRecognizer, InferError> {
        crate::SpeechRecognizer::new(model_path, tokenizer_path, config_path, self.device.clone())
    }

    pub fn use_qwen3(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<crate::Qwen3, InferError> {
        crate::Qwen3::new(model_path, tokenizer_path, self.device.clone())
    }

    pub fn onnx_session(&self, model_path: impl AsRef<Path>) -> Result<OrtSession, InferError> {
        let path = model_path.as_ref();
        let session = match &self.onnx_device {
            OnnxDevice::Cpu => {
                OrtSession::builder()?
                    .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default().build()])?
                    .commit_from_file(path)?
            }
            #[cfg(feature = "cuda")]
            OnnxDevice::Cuda(ordinal) => {
                OrtSession::builder()?
                    .with_execution_providers([
                        ort::execution_providers::CUDAExecutionProvider::default()
                            .with_device_id(*ordinal as i32)
                            .build(),
                        ort::execution_providers::CPUExecutionProvider::default().build(),
                    ])?
                    .commit_from_file(path)?
            }
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
