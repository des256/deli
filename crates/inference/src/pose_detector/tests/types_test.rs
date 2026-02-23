use {base::{Rect, Vec2}, inference::{CocoKeypoint, InferError, Keypoint, PoseDetection}};

#[test]
fn test_keypoint_construction() {
    let kp = Keypoint {
        position: Vec2::new(100.5, 200.3),
        confidence: 0.95,
    };
    assert_eq!(kp.position.x, 100.5);
    assert_eq!(kp.position.y, 200.3);
    assert_eq!(kp.confidence, 0.95);
}

#[test]
fn test_pose_detection_construction() {
    let keypoints = [Keypoint {
        position: Vec2::new(0.0, 0.0),
        confidence: 0.0,
    }; 17];

    let detection = PoseDetection {
        bbox: Rect::new(Vec2::new(10.0, 20.0), Vec2::new(100.0, 150.0)),
        confidence: 0.87,
        keypoints,
    };

    assert_eq!(detection.bbox.origin.x, 10.0);
    assert_eq!(detection.confidence, 0.87);
    assert_eq!(detection.keypoints.len(), 17);
}

#[test]
fn test_coco_keypoint_enum() {
    assert_eq!(CocoKeypoint::Nose as usize, 0);
    assert_eq!(CocoKeypoint::LeftEye as usize, 1);
    assert_eq!(CocoKeypoint::RightEye as usize, 2);
    assert_eq!(CocoKeypoint::LeftEar as usize, 3);
    assert_eq!(CocoKeypoint::RightEar as usize, 4);
    assert_eq!(CocoKeypoint::LeftShoulder as usize, 5);
    assert_eq!(CocoKeypoint::RightShoulder as usize, 6);
    assert_eq!(CocoKeypoint::LeftElbow as usize, 7);
    assert_eq!(CocoKeypoint::RightElbow as usize, 8);
    assert_eq!(CocoKeypoint::LeftWrist as usize, 9);
    assert_eq!(CocoKeypoint::RightWrist as usize, 10);
    assert_eq!(CocoKeypoint::LeftHip as usize, 11);
    assert_eq!(CocoKeypoint::RightHip as usize, 12);
    assert_eq!(CocoKeypoint::LeftKnee as usize, 13);
    assert_eq!(CocoKeypoint::RightKnee as usize, 14);
    assert_eq!(CocoKeypoint::LeftAnkle as usize, 15);
    assert_eq!(CocoKeypoint::RightAnkle as usize, 16);
}

#[test]
fn test_coco_keypoint_into_usize() {
    let index: usize = CocoKeypoint::Nose.into();
    assert_eq!(index, 0);

    let index: usize = CocoKeypoint::RightAnkle.into();
    assert_eq!(index, 16);
}

#[test]
fn test_infer_error_display() {
    let err = InferError::Candle("tensor shape mismatch".to_string());
    assert!(err.to_string().contains("tensor shape mismatch"));

    let err = InferError::Shape("invalid dimensions".to_string());
    assert!(err.to_string().contains("invalid dimensions"));

    let err = InferError::Io("model file not found".to_string());
    assert!(err.to_string().contains("model file not found"));

    let err = InferError::Runtime("task panicked".to_string());
    assert!(err.to_string().contains("task panicked"));
}

#[test]
fn test_infer_error_is_error_trait() {
    use std::error::Error;
    let err = InferError::Candle("test".to_string());
    let _: &dyn Error = &err;
}
