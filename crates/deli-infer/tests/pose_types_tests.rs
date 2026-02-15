#![cfg(feature = "onnx")]

use deli_infer::pose::{Keypoint, KeypointIndex, PoseDetection, COCO_KEYPOINT_COUNT};
use deli_math::{Rect, Vec2};

#[test]
fn test_coco_keypoint_count() {
    assert_eq!(COCO_KEYPOINT_COUNT, 17);
}

#[test]
fn test_keypoint_creation() {
    let position = Vec2::new(100.0, 200.0);
    let confidence = 0.95;
    let keypoint = Keypoint { position, confidence };

    assert_eq!(keypoint.position.x, 100.0);
    assert_eq!(keypoint.position.y, 200.0);
    assert_eq!(keypoint.confidence, 0.95);
}

#[test]
fn test_keypoint_index_enum_count() {
    // Test that we have exactly 17 keypoint indices (0-16)
    assert_eq!(usize::from(KeypointIndex::Nose), 0);
    assert_eq!(usize::from(KeypointIndex::LeftEye), 1);
    assert_eq!(usize::from(KeypointIndex::RightEye), 2);
    assert_eq!(usize::from(KeypointIndex::LeftEar), 3);
    assert_eq!(usize::from(KeypointIndex::RightEar), 4);
    assert_eq!(usize::from(KeypointIndex::LeftShoulder), 5);
    assert_eq!(usize::from(KeypointIndex::RightShoulder), 6);
    assert_eq!(usize::from(KeypointIndex::LeftElbow), 7);
    assert_eq!(usize::from(KeypointIndex::RightElbow), 8);
    assert_eq!(usize::from(KeypointIndex::LeftWrist), 9);
    assert_eq!(usize::from(KeypointIndex::RightWrist), 10);
    assert_eq!(usize::from(KeypointIndex::LeftHip), 11);
    assert_eq!(usize::from(KeypointIndex::RightHip), 12);
    assert_eq!(usize::from(KeypointIndex::LeftKnee), 13);
    assert_eq!(usize::from(KeypointIndex::RightKnee), 14);
    assert_eq!(usize::from(KeypointIndex::LeftAnkle), 15);
    assert_eq!(usize::from(KeypointIndex::RightAnkle), 16);
}

#[test]
fn test_keypoint_index_try_from_valid() {
    assert_eq!(KeypointIndex::try_from(0).unwrap(), KeypointIndex::Nose);
    assert_eq!(KeypointIndex::try_from(10).unwrap(), KeypointIndex::RightWrist);
    assert_eq!(KeypointIndex::try_from(16).unwrap(), KeypointIndex::RightAnkle);
}

#[test]
fn test_keypoint_index_try_from_invalid() {
    assert!(KeypointIndex::try_from(17).is_err());
    assert!(KeypointIndex::try_from(100).is_err());
}

#[test]
fn test_pose_detection_creation() {
    let bbox = Rect::new(Vec2::new(50.0, 60.0), Vec2::new(100.0, 150.0));
    let confidence = 0.92;
    let keypoints = [Keypoint {
        position: Vec2::zero(),
        confidence: 0.0,
    }; COCO_KEYPOINT_COUNT];

    let detection = PoseDetection {
        bbox,
        confidence,
        keypoints,
    };

    assert_eq!(detection.bbox.origin.x, 50.0);
    assert_eq!(detection.bbox.origin.y, 60.0);
    assert_eq!(detection.bbox.size.x, 100.0);
    assert_eq!(detection.bbox.size.y, 150.0);
    assert_eq!(detection.confidence, 0.92);
    assert_eq!(detection.keypoints.len(), 17);
}

#[test]
fn test_pose_detection_keypoint_accessor() {
    let bbox = Rect::new(Vec2::zero(), Vec2::new(100.0, 100.0));
    let confidence = 0.8;
    let mut keypoints = [Keypoint {
        position: Vec2::zero(),
        confidence: 0.0,
    }; COCO_KEYPOINT_COUNT];

    // Set nose keypoint to a specific value
    keypoints[0] = Keypoint {
        position: Vec2::new(50.0, 30.0),
        confidence: 0.99,
    };

    let detection = PoseDetection {
        bbox,
        confidence,
        keypoints,
    };

    let nose = detection.keypoint(KeypointIndex::Nose);
    assert_eq!(nose.position.x, 50.0);
    assert_eq!(nose.position.y, 30.0);
    assert_eq!(nose.confidence, 0.99);
}

#[test]
fn test_keypoint_debug() {
    let keypoint = Keypoint {
        position: Vec2::new(10.0, 20.0),
        confidence: 0.5,
    };
    let debug_str = format!("{:?}", keypoint);
    assert!(debug_str.contains("Keypoint"));
}

#[test]
fn test_pose_detection_debug() {
    let detection = PoseDetection {
        bbox: Rect::new(Vec2::zero(), Vec2::new(10.0, 10.0)),
        confidence: 0.7,
        keypoints: [Keypoint {
            position: Vec2::zero(),
            confidence: 0.0,
        }; COCO_KEYPOINT_COUNT],
    };
    let debug_str = format!("{:?}", detection);
    assert!(debug_str.contains("PoseDetection"));
}

#[test]
fn test_keypoint_clone() {
    let original = Keypoint {
        position: Vec2::new(1.0, 2.0),
        confidence: 0.3,
    };
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_pose_detection_clone() {
    let original = PoseDetection {
        bbox: Rect::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0)),
        confidence: 0.85,
        keypoints: [Keypoint {
            position: Vec2::new(5.0, 6.0),
            confidence: 0.75,
        }; COCO_KEYPOINT_COUNT],
    };
    let cloned = original.clone();
    assert_eq!(original, cloned);
}
