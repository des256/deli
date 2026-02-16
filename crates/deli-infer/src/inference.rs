use crate::InferError;
use candle_core::Device;
use std::path::Path;

#[derive(Debug)]
pub struct Inference {
    device: Device,
}

impl Inference {
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self, InferError> {
        let device = Device::new_cuda(ordinal)?;
        Ok(Self { device })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn use_pose_detector(&self, model_path: impl AsRef<Path>) -> Result<crate::PoseDetector, InferError> {
        crate::PoseDetector::new(model_path, self.device.clone())
    }
}
