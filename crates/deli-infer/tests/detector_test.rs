use deli_infer::Inference;

#[tokio::test]
async fn test_pose_detector_construction_fails_for_missing_file() {
    let inference = Inference::cpu();
    let result = inference.use_pose_detector("fake_model.safetensors");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_pose_detector_with_real_model() {
    let model_path = "../../models/yolov8n-pose.safetensors";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping real model test: {} not found", model_path);
        return;
    }

    // Construct detector with real model (auto-detects nano size)
    let inference = Inference::cpu();
    let detector = inference
        .use_pose_detector(model_path)
        .expect("Failed to load real model");

    // Create a synthetic 480x640 RGB frame (all 128s — gray)
    let data = vec![128.0f32; 480 * 640 * 3];
    let frame = deli_base::Tensor::new(vec![480, 640, 3], data).unwrap();

    // Run detection — with a uniform gray image, expect zero or few detections
    let detections = detector.detect(&frame).await.expect("Inference failed");
    eprintln!(
        "Real model inference: {} detections from gray 480x640 image",
        detections.len()
    );

    // Verify output structure if detections exist
    for det in &detections {
        assert!(det.confidence > 0.0);
        assert!(det.confidence <= 1.0);
        assert_eq!(det.keypoints.len(), 17);

        for kp in &det.keypoints {
            assert!(kp.confidence >= 0.0);
            assert!(kp.confidence <= 1.0);
        }
    }
}
