#![cfg(feature = "onnx")]

use deli_infer::{Device, ModelSource, YoloPoseEstimator};
use deli_math::Tensor;
use std::path::Path;

/// Helper function to load a JPEG image as Tensor<f32> in HWC format
fn load_image_as_tensor(path: &Path) -> Result<Tensor<f32>, String> {
    use image::GenericImageView;

    let img = image::open(path).map_err(|e| format!("Failed to open image: {}", e))?;

    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    // Convert to HWC tensor with values in [0, 255] range
    let mut data = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            data.push(pixel[0] as f32); // R
            data.push(pixel[1] as f32); // G
            data.push(pixel[2] as f32); // B
        }
    }

    Tensor::new(vec![height as usize, width as usize, 3], data)
        .map_err(|e| format!("Failed to create tensor: {}", e))
}

#[test]
#[ignore = "Requires model file - run 'python crates/deli-infer/tests/fixtures/generate_pose_model.py' first"]
fn test_pose_inference_fp32() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model_path = fixtures_dir.join("yolo26n-pose.onnx");
    let image_path = fixtures_dir.join("test_pose_image.jpg");

    // Skip test if model or image not available
    if !model_path.exists() {
        eprintln!("Skipping test: model file not found at {:?}", model_path);
        eprintln!("Run: python crates/deli-infer/tests/fixtures/generate_pose_model.py");
        return;
    }
    if !image_path.exists() {
        eprintln!("Skipping test: test image not found at {:?}", image_path);
        eprintln!("Run: python crates/deli-infer/tests/fixtures/generate_pose_model.py");
        return;
    }

    // Load test image
    let image = load_image_as_tensor(&image_path).expect("Failed to load test image");
    let orig_h = image.shape[0];
    let orig_w = image.shape[1];

    println!("Loaded test image: {}x{}", orig_w, orig_h);

    // Create estimator
    let mut estimator = YoloPoseEstimator::new(
        ModelSource::File(model_path.to_path_buf()),
        Device::Cpu,
    )
    .expect("Failed to load model");

    // Run inference
    let detections = estimator
        .estimate(&image)
        .expect("Inference failed");

    println!("Detected {} poses", detections.len());

    // Validate detections
    for (i, detection) in detections.iter().enumerate() {
        println!("Detection {}: confidence={:.3}", i, detection.confidence);

        // Check confidence is in valid range
        assert!(
            detection.confidence >= 0.0 && detection.confidence <= 1.0,
            "Confidence out of range: {}",
            detection.confidence
        );

        // Check bbox is within image bounds
        assert!(
            detection.bbox.origin.x >= 0.0,
            "Bbox origin.x is negative: {}",
            detection.bbox.origin.x
        );
        assert!(
            detection.bbox.origin.y >= 0.0,
            "Bbox origin.y is negative: {}",
            detection.bbox.origin.y
        );

        let bbox_max_x = detection.bbox.origin.x + detection.bbox.size.x;
        let bbox_max_y = detection.bbox.origin.y + detection.bbox.size.y;
        assert!(
            bbox_max_x <= orig_w as f32,
            "Bbox extends beyond image width: {} > {}",
            bbox_max_x,
            orig_w
        );
        assert!(
            bbox_max_y <= orig_h as f32,
            "Bbox extends beyond image height: {} > {}",
            bbox_max_y,
            orig_h
        );

        // Check keypoints
        assert_eq!(
            detection.keypoints.len(),
            17,
            "Expected 17 keypoints, got {}",
            detection.keypoints.len()
        );

        for (kp_idx, keypoint) in detection.keypoints.iter().enumerate() {
            // Keypoint confidence should be in [0, 1]
            assert!(
                keypoint.confidence >= 0.0 && keypoint.confidence <= 1.0,
                "Keypoint {} confidence out of range: {}",
                kp_idx,
                keypoint.confidence
            );

            // Keypoints should be within image bounds OR have low confidence (occluded/invisible)
            if keypoint.confidence > 0.1 {
                // Only check visible keypoints
                assert!(
                    keypoint.position.x >= 0.0 && keypoint.position.x <= orig_w as f32,
                    "Visible keypoint {} x out of bounds: {} (image width: {})",
                    kp_idx,
                    keypoint.position.x,
                    orig_w
                );
                assert!(
                    keypoint.position.y >= 0.0 && keypoint.position.y <= orig_h as f32,
                    "Visible keypoint {} y out of bounds: {} (image height: {})",
                    kp_idx,
                    keypoint.position.y,
                    orig_h
                );
            }
        }
    }
}

#[test]
#[ignore = "Requires model file - run 'python crates/deli-infer/tests/fixtures/generate_pose_model.py' first"]
fn test_pose_inference_uint8() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model_path = fixtures_dir.join("yolo26n-pose-uint8.onnx");
    let image_path = fixtures_dir.join("test_pose_image.jpg");

    // Skip test if model or image not available
    if !model_path.exists() {
        eprintln!("Skipping test: uint8 model file not found at {:?}", model_path);
        eprintln!("Run: python crates/deli-infer/tests/fixtures/generate_pose_model.py");
        return;
    }
    if !image_path.exists() {
        eprintln!("Skipping test: test image not found at {:?}", image_path);
        eprintln!("Run: python crates/deli-infer/tests/fixtures/generate_pose_model.py");
        return;
    }

    // Load test image
    let image = load_image_as_tensor(&image_path).expect("Failed to load test image");

    println!("Testing uint8 quantized model");

    // Create estimator with uint8 model
    let mut estimator = YoloPoseEstimator::new(
        ModelSource::File(model_path.to_path_buf()),
        Device::Cpu,
    )
    .expect("Failed to load uint8 model");

    // Run inference
    let detections = estimator
        .estimate(&image)
        .expect("Inference failed with uint8 model");

    println!("Detected {} poses with uint8 model", detections.len());

    // Validate that uint8 model produces valid output structure
    // (quantization may affect accuracy, but structure should be the same)
    for detection in &detections {
        assert!(
            detection.confidence >= 0.0 && detection.confidence <= 1.0,
            "uint8 model: confidence out of range"
        );
        assert_eq!(detection.keypoints.len(), 17, "uint8 model: wrong keypoint count");
    }
}

#[test]
#[ignore = "Requires model file - run 'python crates/deli-infer/tests/fixtures/generate_pose_model.py' first"]
fn test_estimator_with_custom_thresholds() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model_path = fixtures_dir.join("yolo26n-pose.onnx");
    let image_path = fixtures_dir.join("test_pose_image.jpg");

    if !model_path.exists() || !image_path.exists() {
        return; // Skip if fixtures not available
    }

    let image = load_image_as_tensor(&image_path).expect("Failed to load test image");

    // Test with high confidence threshold (should return fewer detections)
    let mut estimator_high_conf = YoloPoseEstimator::new(
        ModelSource::File(model_path.to_path_buf()),
        Device::Cpu,
    )
    .expect("Failed to load model")
    .with_conf_threshold(0.7); // High threshold

    let detections_high = estimator_high_conf
        .estimate(&image)
        .expect("Inference failed");

    println!("High threshold (0.7): {} detections", detections_high.len());

    // Test with low confidence threshold (should return more detections)
    let mut estimator_low_conf = YoloPoseEstimator::new(
        ModelSource::File(model_path.to_path_buf()),
        Device::Cpu,
    )
    .expect("Failed to load model")
    .with_conf_threshold(0.1); // Low threshold

    let detections_low = estimator_low_conf
        .estimate(&image)
        .expect("Inference failed");

    println!("Low threshold (0.1): {} detections", detections_low.len());

    // Lower threshold should produce >= detections than higher threshold
    assert!(
        detections_low.len() >= detections_high.len(),
        "Lower threshold should produce more or equal detections"
    );
}
