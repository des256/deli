use deli_image::Image;
use deli_infer::Inference;
use futures_util::{SinkExt, StreamExt};

fn cuda() -> Inference {
    Inference::cuda(0).expect("CUDA device required")
}

#[tokio::test]
async fn test_pose_detector_construction_fails_for_missing_file() {
    let inference = cuda();
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
    let inference = cuda();
    let mut detector = inference
        .use_pose_detector(model_path)
        .expect("Failed to load real model");

    // Create a synthetic 480x640 RGB frame (all 128s — gray)
    let data = vec![128.0f32; 480 * 640 * 3];
    let frame = deli_base::Tensor::new(vec![480, 640, 3], data).unwrap();

    // Send image via Sink and close
    detector.send(Image::F32(frame)).await.expect("Send failed");
    detector.close().await.expect("Close failed");

    // Read detection result from Stream
    let detections = detector
        .next()
        .await
        .expect("Stream ended unexpectedly")
        .expect("Inference failed");

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

    // Stream should end after all items consumed
    assert!(detector.next().await.is_none());
}
