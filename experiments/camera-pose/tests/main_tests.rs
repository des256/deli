use deli_base::Tensor;

// Helper function (will be in main.rs)
fn tensor_u8_to_f32(t: &Tensor<u8>) -> Result<Tensor<f32>, deli_base::TensorError> {
    Tensor::new(
        t.shape.clone(),
        t.data.iter().map(|&v| v as f32).collect(),
    )
}

#[test]
fn test_tensor_u8_to_f32_conversion() {
    let u8_tensor = Tensor::new(vec![2, 3, 3], vec![
        255, 0, 0,    128, 128, 128,  0, 255, 0,
        100, 50, 25,  200, 150, 100,  50, 100, 150,
    ]).unwrap();

    let f32_tensor = tensor_u8_to_f32(&u8_tensor).unwrap();

    assert_eq!(f32_tensor.shape, vec![2, 3, 3]);
    assert_eq!(f32_tensor.data.len(), 18);

    // Check first pixel (255, 0, 0) → (255.0, 0.0, 0.0)
    assert_eq!(f32_tensor.data[0], 255.0);
    assert_eq!(f32_tensor.data[1], 0.0);
    assert_eq!(f32_tensor.data[2], 0.0);

    // Check a middle pixel (200, 150, 100)
    assert_eq!(f32_tensor.data[12], 200.0);
    assert_eq!(f32_tensor.data[13], 150.0);
    assert_eq!(f32_tensor.data[14], 100.0);
}

#[test]
fn test_tensor_u8_to_f32_preserves_shape() {
    let u8_tensor = Tensor::new(vec![10, 20, 3], vec![0u8; 10 * 20 * 3]).unwrap();

    let f32_tensor = tensor_u8_to_f32(&u8_tensor).unwrap();

    assert_eq!(f32_tensor.shape, vec![10, 20, 3]);
    assert_eq!(f32_tensor.data.len(), 600);
}

#[test]
fn test_tensor_u8_to_f32_values_in_range() {
    let u8_tensor = Tensor::new(vec![1, 1, 3], vec![0, 127, 255]).unwrap();

    let f32_tensor = tensor_u8_to_f32(&u8_tensor).unwrap();

    assert_eq!(f32_tensor.data[0], 0.0);
    assert_eq!(f32_tensor.data[1], 127.0);
    assert_eq!(f32_tensor.data[2], 255.0);

    // Values should remain in [0, 255] range (no normalization)
    for &val in &f32_tensor.data {
        assert!(val >= 0.0 && val <= 255.0, "Value {} out of range", val);
    }
}
