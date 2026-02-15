use crate::{Backend, Device, InferError, ModelSource, Session};
use deli_base::Tensor;
use ndarray::ArrayD;
use ort::{inputs, session::Session as OrtSession, value::TensorRef};
use std::collections::HashMap;

pub struct OnnxBackend {
    device: Device,
}

impl OnnxBackend {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl Backend for OnnxBackend {
    fn load_model(
        &self,
        model: ModelSource,
    ) -> Result<Box<dyn Session>, InferError> {
        let device = &self.device;
        let mut builder = OrtSession::builder().map_err(|e| {
            InferError::BackendError(format!("failed to create session builder: {}", e))
        })?;

        // Map Device to ort execution providers
        builder = match device {
            Device::Cpu => {
                println!("[onnx] Using CPU execution provider");
                builder
            }
            #[cfg(feature = "cuda")]
            Device::Cuda { device_id } => {
                use ort::ep::ExecutionProvider;
                use ort::execution_providers::CUDAExecutionProvider;
                let ep = CUDAExecutionProvider::default().with_device_id(*device_id);
                let available = ep.is_available().unwrap_or(false);
                println!(
                    "[onnx] CUDA EP requested (device_id={}), available: {}",
                    device_id, available
                );
                builder
                    .with_execution_providers([ep.build()])
                    .map_err(|_| InferError::UnsupportedDevice(device.clone()))?
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda { .. } => {
                return Err(InferError::UnsupportedDevice(device.clone()));
            }
            #[cfg(feature = "tensorrt")]
            Device::TensorRt { device_id, fp16 } => {
                use ort::ep::ExecutionProvider;
                use ort::execution_providers::TensorRTExecutionProvider;
                let mut ep = TensorRTExecutionProvider::default().with_device_id(*device_id);
                if *fp16 {
                    ep = ep.with_fp16(true);
                }
                let available = ep.is_available().unwrap_or(false);
                println!(
                    "[onnx] TensorRT EP requested (device_id={}, fp16={}), available: {}",
                    device_id, fp16, available
                );
                builder
                    .with_execution_providers([ep.build()])
                    .map_err(|_| InferError::UnsupportedDevice(device.clone()))?
            }
            #[cfg(not(feature = "tensorrt"))]
            Device::TensorRt { .. } => {
                return Err(InferError::UnsupportedDevice(device.clone()));
            }
        };

        // Load model
        let session = match model {
            ModelSource::File(path) => builder.commit_from_file(path).map_err(|e| {
                InferError::ModelLoad(format!("failed to load model from file: {}", e))
            })?,
            ModelSource::Memory(bytes) => builder.commit_from_memory(&bytes).map_err(|e| {
                InferError::ModelLoad(format!("failed to load model from memory: {}", e))
            })?,
        };

        // Extract input and output names
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();

        Ok(Box::new(OnnxSession {
            session,
            input_names,
            output_names,
        }))
    }
}

pub struct OnnxSession {
    session: OrtSession,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl Session for OnnxSession {
    fn run(
        &mut self,
        inputs: &[(&str, Tensor<f32>)],
    ) -> Result<HashMap<String, Tensor<f32>>, InferError> {
        // Validate input names
        for (name, _) in inputs {
            if !self.input_names.contains(&name.to_string()) {
                return Err(InferError::InvalidInput {
                    name: name.to_string(),
                    expected_names: self.input_names.clone(),
                });
            }
        }

        // Convert inputs to ndarrays and create ort session inputs
        // We need to build the inputs! macro call dynamically
        // For now, we'll support up to 2 inputs as a simple implementation
        let outputs = match inputs.len() {
            1 => {
                let (name, tensor) = &inputs[0];
                let array = tensor_to_ndarray(tensor.clone())?;
                let tensor_ref = TensorRef::from_array_view(array.view()).map_err(|e| {
                    InferError::BackendError(format!("failed to create tensor ref: {}", e))
                })?;
                self.session
                    .run(inputs![*name => tensor_ref])
                    .map_err(|e| InferError::BackendError(format!("inference failed: {}", e)))?
            }
            2 => {
                let (name1, tensor1) = &inputs[0];
                let (name2, tensor2) = &inputs[1];
                let array1 = tensor_to_ndarray(tensor1.clone())?;
                let array2 = tensor_to_ndarray(tensor2.clone())?;
                let tensor_ref1 = TensorRef::from_array_view(array1.view()).map_err(|e| {
                    InferError::BackendError(format!("failed to create tensor ref 1: {}", e))
                })?;
                let tensor_ref2 = TensorRef::from_array_view(array2.view()).map_err(|e| {
                    InferError::BackendError(format!("failed to create tensor ref 2: {}", e))
                })?;
                self.session
                    .run(inputs![*name1 => tensor_ref1, *name2 => tensor_ref2])
                    .map_err(|e| InferError::BackendError(format!("inference failed: {}", e)))?
            }
            _ => {
                return Err(InferError::BackendError(
                    "only 1-2 inputs supported currently".to_string(),
                ));
            }
        };

        // Convert outputs to HashMap<String, Tensor<f32>>
        let mut result = HashMap::new();
        for output_name in &self.output_names {
            let value = &outputs[output_name.as_str()];

            // Extract as f32 array
            let array = value.try_extract_array::<f32>().map_err(|e| {
                InferError::UnsupportedDtype(format!(
                    "output '{}' is not f32: {}",
                    output_name, e
                ))
            })?;

            // Convert to Tensor<f32>
            let tensor = ndarray_to_tensor(array)?;
            result.insert(output_name.clone(), tensor);
        }

        Ok(result)
    }

    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

// Helper function to convert Tensor<f32> to ndarray::ArrayD<f32>
pub fn tensor_to_ndarray(tensor: Tensor<f32>) -> Result<ArrayD<f32>, InferError> {
    ArrayD::from_shape_vec(tensor.shape, tensor.data).map_err(|e| {
        InferError::BackendError(format!("failed to create ndarray from tensor: {}", e))
    })
}

// Helper function to convert ndarray::ArrayD<f32> to Tensor<f32>
pub fn ndarray_to_tensor(array: ndarray::ArrayView<'_, f32, ndarray::IxDyn>) -> Result<Tensor<f32>, InferError> {
    let shape = array.shape().to_vec();
    let data = array.iter().copied().collect();
    Tensor::new(shape, data)
        .map_err(|e| InferError::BackendError(format!("failed to create tensor: {}", e)))
}
