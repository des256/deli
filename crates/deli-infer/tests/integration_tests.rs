#![cfg(feature = "onnx")]

use deli_infer::{create_registry, Device, InferError, ModelSource};
use deli_base::Tensor;
use std::fs;

#[test]
fn test_integration_onnx_add_from_file() {
    let registry = create_registry();
    let backend = registry.get("onnx").expect("onnx backend not found");

    let model_path = "tests/fixtures/test_add.onnx";
    let mut session = backend
        .load_model(ModelSource::File(model_path.into()), Device::Cpu)
        .expect("failed to load model");

    // Create input tensors: 2x3 arrays
    let x = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = Tensor::new(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();

    // Run inference
    let outputs = session.run(&[("X", x), ("Y", y)]).unwrap();

    // Verify output
    let z = outputs.get("Z").expect("output Z not found");
    assert_eq!(z.shape, vec![2, 3]);
    assert_eq!(z.data, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}

#[test]
fn test_integration_onnx_add_from_memory() {
    let registry = create_registry();
    let backend = registry.get("onnx").expect("onnx backend not found");

    // Load model from file to bytes
    let model_bytes =
        fs::read("tests/fixtures/test_add.onnx").expect("failed to read model file");

    let mut session = backend
        .load_model(ModelSource::Memory(model_bytes), Device::Cpu)
        .expect("failed to load model from memory");

    // Create input tensors
    let x = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = Tensor::new(vec![2, 3], vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).unwrap();

    // Run inference
    let outputs = session.run(&[("X", x), ("Y", y)]).unwrap();

    // Verify output
    let z = outputs.get("Z").expect("output Z not found");
    assert_eq!(z.shape, vec![2, 3]);
    assert_eq!(z.data, vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
}

#[test]
fn test_integration_invalid_input_name() {
    let registry = create_registry();
    let backend = registry.get("onnx").unwrap();

    let mut session = backend
        .load_model(
            ModelSource::File("tests/fixtures/test_add.onnx".into()),
            Device::Cpu,
        )
        .unwrap();

    let x = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Try with wrong input name
    let result = session.run(&[("wrong_input", x), ("Y", y)]);

    assert!(result.is_err());
    match result.unwrap_err() {
        InferError::InvalidInput { name, .. } => {
            assert_eq!(name, "wrong_input");
        }
        _ => panic!("expected InvalidInput error"),
    }
}

#[test]
fn test_integration_session_input_output_names() {
    let registry = create_registry();
    let backend = registry.get("onnx").unwrap();

    let session = backend
        .load_model(
            ModelSource::File("tests/fixtures/test_add.onnx".into()),
            Device::Cpu,
        )
        .unwrap();

    let input_names = session.input_names();
    assert_eq!(input_names.len(), 2);
    assert!(input_names.contains(&"X".to_string()));
    assert!(input_names.contains(&"Y".to_string()));

    let output_names = session.output_names();
    assert_eq!(output_names.len(), 1);
    assert_eq!(output_names[0], "Z");
}
