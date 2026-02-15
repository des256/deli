#![cfg(feature = "onnx")]

use deli_infer::{Device, InferError, ModelSource, YoloPoseEstimator};
use deli_math::Tensor;

#[test]
fn test_estimator_construction_fails_for_missing_model() {
    let result = YoloPoseEstimator::new(
        ModelSource::File("nonexistent.onnx".into()),
        Device::Cpu,
    );
    assert!(result.is_err());
}

#[test]
fn test_estimator_builder_pattern() {
    // We can't construct a real estimator without a model file,
    // but we can verify the builder methods compile and chain correctly
    // by testing through the error path: construct would fail, but
    // the type signature is correct.

    // Verify the builder methods exist and return Self
    fn _verify_builder_api(estimator: YoloPoseEstimator) -> YoloPoseEstimator {
        estimator
            .with_conf_threshold(0.5)
            .with_iou_threshold(0.6)
    }

    // Verify accessor methods exist
    fn _verify_accessors(estimator: &YoloPoseEstimator) -> (f32, f32) {
        (estimator.conf_threshold(), estimator.iou_threshold())
    }
}

#[test]
fn test_estimator_validates_2d_input_shape() {
    // Create a mock test by using postprocess directly (estimator needs real model)
    // Instead, verify shape validation logic via the preprocess function
    let invalid_input = Tensor::new(vec![640, 640], vec![0.0; 640 * 640]).unwrap();

    // preprocess validates shape too - 2D input should fail
    let result = deli_infer::pose::preprocess(&invalid_input);
    assert!(matches!(result, Err(InferError::ShapeMismatch { .. })));
}

#[test]
fn test_estimator_validates_channel_count() {
    // 1-channel input should fail validation
    let invalid_input = Tensor::new(vec![640, 640, 1], vec![0.0; 640 * 640]).unwrap();

    let result = deli_infer::pose::preprocess(&invalid_input);
    assert!(matches!(result, Err(InferError::ShapeMismatch { .. })));
}

#[test]
fn test_postprocess_validates_output_shape() {
    // Verify that postprocess returns ShapeMismatch for wrong shapes
    let letterbox = deli_infer::pose::LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    // Wrong second dimension (10 instead of 56)
    let bad_output = Tensor::new(vec![1, 10, 5], vec![0.0; 50]).unwrap();
    let result = deli_infer::pose::postprocess(&bad_output, &letterbox, 0.25, 0.45);
    assert!(matches!(result, Err(InferError::ShapeMismatch { .. })));

    // 2D tensor instead of 3D
    let bad_output_2d = Tensor::new(vec![56, 5], vec![0.0; 280]).unwrap();
    let result = deli_infer::pose::postprocess(&bad_output_2d, &letterbox, 0.25, 0.45);
    assert!(matches!(result, Err(InferError::ShapeMismatch { .. })));
}
