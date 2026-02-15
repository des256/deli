use deli_infer::{Device, InferError, ModelSource};

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
