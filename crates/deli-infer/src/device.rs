use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    Cpu,
    Cuda { device_id: i32 },
    TensorRt { device_id: i32, fp16: bool },
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            Device::Cuda { device_id } => write!(f, "CUDA(device_id={device_id})"),
            Device::TensorRt { device_id, fp16 } => {
                write!(f, "TensorRT(device_id={device_id}, fp16={fp16})")
            }
        }
    }
}
