#![cfg(feature = "onnx")]

use deli_infer::backends::OnnxBackend;
use deli_infer::Backend;

#[test]
fn test_onnx_backend_name() {
    let backend = OnnxBackend;
    assert_eq!(backend.name(), "onnx");
}

#[test]
fn test_tensor_to_ndarray_conversion() {
    use deli_infer::backends::onnx::tensor_to_ndarray;
    use deli_math::Tensor;

    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let array = tensor_to_ndarray(tensor).unwrap();

    assert_eq!(array.shape(), &[2, 3]);
    assert_eq!(array.len(), 6);
    assert_eq!(array[[0, 0]], 1.0);
    assert_eq!(array[[1, 2]], 6.0);
}

#[test]
fn test_ndarray_to_tensor_conversion() {
    use deli_infer::backends::onnx::ndarray_to_tensor;
    use ndarray::ArrayD;

    let array = ArrayD::<f32>::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tensor = ndarray_to_tensor(array.view()).unwrap();

    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_create_registry_includes_onnx() {
    use deli_infer::create_registry;

    let registry = create_registry();
    let backends = registry.list();

    assert!(backends.contains(&"onnx"));
    assert!(registry.get("onnx").is_some());
}
