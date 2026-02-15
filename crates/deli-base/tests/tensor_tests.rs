use deli_base::{Tensor, TensorError};

#[test]
fn test_tensor_new_valid() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_new_shape_mismatch() {
    let result = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(TensorError::ShapeMismatch { .. })));
}

#[test]
fn test_tensor_new_overflow() {
    let result = Tensor::<f32>::new(vec![usize::MAX, 2], vec![]);
    assert!(matches!(result, Err(TensorError::ShapeOverflow)));
}

#[test]
fn test_tensor_zeros() {
    let tensor = Tensor::<f32>::zeros(vec![2, 3]).unwrap();
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, vec![0.0; 6]);
}

#[test]
fn test_tensor_from_scalar() {
    let tensor = Tensor::from_scalar(42.0);
    assert_eq!(tensor.shape, vec![]);
    assert_eq!(tensor.data, vec![42.0]);
}

#[test]
fn test_tensor_ndim() {
    let tensor = Tensor::new(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    assert_eq!(tensor.ndim(), 3);
}

#[test]
fn test_tensor_len() {
    let tensor = Tensor::new(vec![2, 3], vec![0.0; 6]).unwrap();
    assert_eq!(tensor.len(), 6);
}

#[test]
fn test_tensor_is_empty() {
    let tensor_empty = Tensor::<f32>::new(vec![0], vec![]).unwrap();
    assert!(tensor_empty.is_empty());

    let tensor_not_empty = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    assert!(!tensor_not_empty.is_empty());
}

#[test]
fn test_tensor_clone() {
    let tensor1 = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tensor2 = tensor1.clone();
    assert_eq!(tensor1, tensor2);
}

#[test]
fn test_tensor_debug() {
    let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("Tensor"));
    assert!(debug_str.contains("shape"));
    assert!(debug_str.contains("data"));
}
