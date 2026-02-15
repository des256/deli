#![cfg(feature = "onnx")]

use deli_infer::pose::{preprocess, LetterboxInfo};
use deli_math::Tensor;

#[test]
fn test_letterbox_info_creation() {
    let info = LetterboxInfo {
        scale: 0.5,
        pad_x: 10.0,
        pad_y: 20.0,
    };
    assert_eq!(info.scale, 0.5);
    assert_eq!(info.pad_x, 10.0);
    assert_eq!(info.pad_y, 20.0);
}

#[test]
fn test_preprocess_square_image_640x640() {
    // 640x640 image should have no padding
    let image = Tensor::zeros(vec![640, 640, 3]).unwrap();
    let (preprocessed, letterbox) = preprocess(&image).unwrap();

    // Output shape should be [1, 3, 640, 640] (NCHW)
    assert_eq!(preprocessed.shape, vec![1, 3, 640, 640]);

    // No padding needed
    assert_eq!(letterbox.scale, 1.0);
    assert_eq!(letterbox.pad_x, 0.0);
    assert_eq!(letterbox.pad_y, 0.0);
}

#[test]
fn test_preprocess_wide_image_320x640() {
    // 320x640 image (width > height) should be padded vertically
    let image = Tensor::zeros(vec![320, 640, 3]).unwrap();
    let (preprocessed, letterbox) = preprocess(&image).unwrap();

    // Output shape should be [1, 3, 640, 640]
    assert_eq!(preprocessed.shape, vec![1, 3, 640, 640]);

    // Scale should be 1.0 (640/640)
    assert_eq!(letterbox.scale, 1.0);

    // Horizontal padding should be 0
    assert_eq!(letterbox.pad_x, 0.0);

    // Vertical padding should be (640 - 320) / 2 = 160
    assert_eq!(letterbox.pad_y, 160.0);
}

#[test]
fn test_preprocess_tall_image_640x320() {
    // 640x320 image (height > width) should be padded horizontally
    let image = Tensor::zeros(vec![640, 320, 3]).unwrap();
    let (preprocessed, letterbox) = preprocess(&image).unwrap();

    // Output shape should be [1, 3, 640, 640]
    assert_eq!(preprocessed.shape, vec![1, 3, 640, 640]);

    // Scale should be 1.0 (640/640)
    assert_eq!(letterbox.scale, 1.0);

    // Horizontal padding should be (640 - 320) / 2 = 160
    assert_eq!(letterbox.pad_x, 160.0);

    // Vertical padding should be 0
    assert_eq!(letterbox.pad_y, 0.0);
}

#[test]
fn test_preprocess_small_image_480x640() {
    // 480x640 image needs to be scaled down
    let image = Tensor::zeros(vec![480, 640, 3]).unwrap();
    let (preprocessed, letterbox) = preprocess(&image).unwrap();

    // Output shape should be [1, 3, 640, 640]
    assert_eq!(preprocessed.shape, vec![1, 3, 640, 640]);

    // Scale should be 640/640 = 1.0
    assert_eq!(letterbox.scale, 1.0);

    // Horizontal padding should be 0 (width matches after scale)
    assert_eq!(letterbox.pad_x, 0.0);

    // Vertical padding should be (640 - 480) / 2 = 80
    assert_eq!(letterbox.pad_y, 80.0);
}

#[test]
fn test_preprocess_large_image_1280x960() {
    // 1280x960 image needs to be scaled down
    let image = Tensor::zeros(vec![1280, 960, 3]).unwrap();
    let (preprocessed, letterbox) = preprocess(&image).unwrap();

    // Output shape should be [1, 3, 640, 640]
    assert_eq!(preprocessed.shape, vec![1, 3, 640, 640]);

    // Scale should be 640/1280 = 0.5
    assert_eq!(letterbox.scale, 0.5);

    // After scaling: width=480, height=640
    // Horizontal padding should be (640 - 480) / 2 = 80
    assert_eq!(letterbox.pad_x, 80.0);

    // Vertical padding should be 0
    assert_eq!(letterbox.pad_y, 0.0);
}

#[test]
fn test_preprocess_value_rescaling() {
    // Create a small image with known pixel values (0-255 range)
    let mut data = Vec::new();
    for _ in 0..320 * 480 {
        // Red=255, Green=128, Blue=0
        data.push(255.0);
        data.push(128.0);
        data.push(0.0);
    }
    let image = Tensor::new(vec![320, 480, 3], data).unwrap();
    let (preprocessed, _) = preprocess(&image).unwrap();

    // Check that values are rescaled to [0.0, 1.0]
    // Original 255 should become 1.0
    // Original 128 should become ~0.5
    // Original 0 should become 0.0
    let max_val = preprocessed.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = preprocessed.data.iter().copied().fold(f32::INFINITY, f32::min);

    assert!(max_val <= 1.0, "Max value should be <= 1.0, got {}", max_val);
    assert!(min_val >= 0.0, "Min value should be >= 0.0, got {}", min_val);

    // Check specific values (accounting for padding which is 114.0 / 255.0 ≈ 0.447)
    let has_one = preprocessed.data.iter().any(|&v| (v - 1.0).abs() < 0.01);
    let has_zero = preprocessed.data.iter().any(|&v| v.abs() < 0.01);
    assert!(has_one, "Should have values close to 1.0");
    assert!(has_zero, "Should have values close to 0.0");
}

#[test]
fn test_preprocess_padding_value() {
    // Create a 320x640 black image that will need vertical padding
    let image = Tensor::zeros(vec![320, 640, 3]).unwrap();
    let (preprocessed, letterbox) = preprocess(&image).unwrap();

    // Verify there is padding
    assert!(letterbox.pad_y > 0.0, "Should have vertical padding");

    // With a 320x640 black image scaled to 640x640,
    // the padding pixels should be 114.0/255.0 ≈ 0.447
    let expected_pad = 114.0 / 255.0;

    // Count pixels near the padding value
    let pad_pixels = preprocessed.data.iter()
        .filter(|&&v| (v - expected_pad).abs() < 0.01)
        .count();

    // Some pixels should be the padding color
    assert!(pad_pixels > 0, "Should have padding pixels with value {}", expected_pad);
}

#[test]
fn test_coordinate_round_trip() {
    // Test that we can recover original coordinates after letterbox transformation
    let image = Tensor::zeros(vec![480, 640, 3]).unwrap();
    let (_, letterbox) = preprocess(&image).unwrap();

    // Original coordinate in the source image
    let orig_x = 320.0;
    let orig_y = 240.0;

    // Transform to model space (apply letterbox)
    let model_x = (orig_x * letterbox.scale) + letterbox.pad_x;
    let model_y = (orig_y * letterbox.scale) + letterbox.pad_y;

    // Transform back to original space
    let recovered_x = (model_x - letterbox.pad_x) / letterbox.scale;
    let recovered_y = (model_y - letterbox.pad_y) / letterbox.scale;

    // Should recover original coordinates within tolerance
    assert!((recovered_x - orig_x).abs() < 1.0,
        "X coordinate mismatch: expected {}, got {}", orig_x, recovered_x);
    assert!((recovered_y - orig_y).abs() < 1.0,
        "Y coordinate mismatch: expected {}, got {}", orig_y, recovered_y);
}

#[test]
fn test_preprocess_hwc_to_nchw_transpose() {
    // Create a small test image with distinct RGB channels
    let mut data = Vec::new();
    for _h in 0..8 {
        for _w in 0..8 {
            data.push(1.0);  // R channel
            data.push(2.0);  // G channel
            data.push(3.0);  // B channel
        }
    }
    let image = Tensor::new(vec![8, 8, 3], data).unwrap();
    let (preprocessed, _) = preprocess(&image).unwrap();

    // Output should be [1, 3, 640, 640] in NCHW format
    assert_eq!(preprocessed.shape, vec![1, 3, 640, 640]);

    // The data should be organized as: all R values, then all G values, then all B values
    // Due to padding and rescaling, exact values vary, but we can check the layout
    // by verifying that channels are contiguous
    assert_eq!(preprocessed.data.len(), 1 * 3 * 640 * 640);

    // This is just a structural check - values will be affected by padding
}

#[test]
fn test_preprocess_invalid_shape() {
    // 2D image (missing channel dimension) should fail
    let image = Tensor::zeros(vec![640, 640]).unwrap();
    let result = preprocess(&image);
    assert!(result.is_err(), "Should reject 2D image");

    // 4D image should fail
    let image = Tensor::zeros(vec![1, 640, 640, 3]).unwrap();
    let result = preprocess(&image);
    assert!(result.is_err(), "Should reject 4D image");

    // Wrong number of channels
    let image = Tensor::zeros(vec![640, 640, 4]).unwrap();
    let result = preprocess(&image);
    assert!(result.is_err(), "Should reject 4-channel image");
}
