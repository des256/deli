use deli_infer::{create_registry, Backend, Device, InferError, ModelSource};

#[test]
fn test_device_cpu() {
    let device = Device::Cpu;
    assert!(matches!(device, Device::Cpu));
    let debug_str = format!("{:?}", device);
    assert!(debug_str.contains("Cpu"));
}

#[test]
fn test_device_cuda() {
    let device = Device::Cuda { device_id: 0 };
    if let Device::Cuda { device_id } = device {
        assert_eq!(device_id, 0);
    } else {
        panic!("Expected Cuda variant");
    }
}

#[test]
fn test_device_tensorrt() {
    let device = Device::TensorRt {
        device_id: 1,
        fp16: true,
    };
    if let Device::TensorRt { device_id, fp16 } = device {
        assert_eq!(device_id, 1);
        assert!(fp16);
    } else {
        panic!("Expected TensorRt variant");
    }
}

#[test]
fn test_infer_error_display() {
    let err1 = InferError::BackendError("test error".to_string());
    assert_eq!(err1.to_string(), "backend error: test error");

    let err2 = InferError::UnsupportedDevice(Device::Cpu);
    let msg = err2.to_string();
    assert!(msg.contains("unsupported device"));

    let err3 = InferError::ModelLoad("failed to load".to_string());
    assert_eq!(err3.to_string(), "model load error: failed to load");

    let err4 = InferError::UnsupportedDtype("int64".to_string());
    assert_eq!(err4.to_string(), "unsupported dtype: int64");

    let err5 = InferError::InvalidInput {
        name: "wrong_input".to_string(),
        expected_names: vec!["x".to_string(), "y".to_string()],
    };
    let msg = err5.to_string();
    assert!(msg.contains("wrong_input"));
    assert!(msg.contains("x"));
    assert!(msg.contains("y"));
}

#[test]
fn test_model_source_file() {
    let source = ModelSource::File("model.onnx".into());
    if let ModelSource::File(path) = source {
        assert_eq!(path.to_str().unwrap(), "model.onnx");
    } else {
        panic!("Expected File variant");
    }
}

#[test]
fn test_model_source_memory() {
    let bytes = vec![1, 2, 3, 4, 5];
    let source = ModelSource::Memory(bytes.clone());
    if let ModelSource::Memory(data) = source {
        assert_eq!(data, bytes);
    } else {
        panic!("Expected Memory variant");
    }
}

#[test]
fn test_backend_registry_empty() {
    #[cfg(not(feature = "onnx"))]
    {
        let registry = create_registry();
        assert_eq!(registry.list().len(), 0);
    }
    #[cfg(feature = "onnx")]
    {
        let registry = create_registry();
        assert_eq!(registry.list().len(), 1);
        assert!(registry.list().contains(&"onnx"));
    }
}

// Mock backend for testing registry
struct MockBackend {
    name: String,
}

impl Backend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn load_model(
        &self,
        _model: ModelSource,
        _device: Device,
    ) -> Result<Box<dyn deli_infer::Session>, InferError> {
        Err(InferError::BackendError("mock backend".to_string()))
    }
}

#[test]
fn test_backend_registry_register_and_get() {
    let mut registry = create_registry();
    registry.register(Box::new(MockBackend {
        name: "test".to_string(),
    }));

    #[cfg(feature = "onnx")]
    assert_eq!(registry.list().len(), 2); // onnx + test
    #[cfg(not(feature = "onnx"))]
    assert_eq!(registry.list().len(), 1);

    assert!(registry.list().contains(&"test"));

    let backend = registry.get("test");
    assert!(backend.is_some());
    assert_eq!(backend.unwrap().name(), "test");

    let missing = registry.get("nonexistent");
    assert!(missing.is_none());
}

#[test]
fn test_backend_registry_multiple() {
    let mut registry = create_registry();
    registry.register(Box::new(MockBackend {
        name: "backend1".to_string(),
    }));
    registry.register(Box::new(MockBackend {
        name: "backend2".to_string(),
    }));

    #[cfg(feature = "onnx")]
    assert_eq!(registry.list().len(), 3); // onnx + backend1 + backend2
    #[cfg(not(feature = "onnx"))]
    assert_eq!(registry.list().len(), 2);

    assert!(registry.get("backend1").is_some());
    assert!(registry.get("backend2").is_some());
}
