#![cfg(feature = "onnx")]

use deli_infer::pose::{iou, postprocess, LetterboxInfo};
use deli_math::{Rect, Tensor, Vec2};

#[test]
fn test_iou_non_overlapping() {
    let a = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
    let b = Rect::new(Vec2::new(20.0, 20.0), Vec2::new(10.0, 10.0));
    assert_eq!(iou(&a, &b), 0.0);
}

#[test]
fn test_iou_identical() {
    let a = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
    let b = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
    assert_eq!(iou(&a, &b), 1.0);
}

#[test]
fn test_iou_partial_overlap() {
    let a = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
    let b = Rect::new(Vec2::new(5.0, 0.0), Vec2::new(10.0, 10.0));
    // Intersection: 5x10 = 50, Union: 100+100-50 = 150, IoU = 1/3
    let result = iou(&a, &b);
    assert!((result - 0.333).abs() < 0.01);
}

#[test]
fn test_iou_zero_area_boxes() {
    let a = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0));
    let b = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
    assert_eq!(iou(&a, &b), 0.0);
}

/// Helper to set value at [0, feature_idx, detection_idx] in a [1, 56, N] tensor
fn set_detection(data: &mut [f32], n: usize, feature_idx: usize, detection_idx: usize, value: f32) {
    data[feature_idx * n + detection_idx] = value;
}

/// Fill a detection in the tensor data buffer
fn fill_detection(
    data: &mut [f32],
    n: usize,
    det_idx: usize,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    conf: f32,
) {
    set_detection(data, n, 0, det_idx, cx);
    set_detection(data, n, 1, det_idx, cy);
    set_detection(data, n, 2, det_idx, w);
    set_detection(data, n, 3, det_idx, h);
    set_detection(data, n, 4, det_idx, conf);
    for i in 0..17 {
        set_detection(data, n, 5 + i * 3, det_idx, 0.0);
        set_detection(data, n, 5 + i * 3 + 1, det_idx, 0.0);
        set_detection(data, n, 5 + i * 3 + 2, det_idx, 0.0);
    }
}

#[test]
fn test_postprocess_invalid_shape_returns_error() {
    // Shape [1, 10, 5] is invalid (should be [1, 56, N])
    let data = vec![0.0; 10 * 5];
    let output = Tensor::new(vec![1, 10, 5], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let result = postprocess(&output, &letterbox, 0.25, 0.45);
    assert!(result.is_err());
}

#[test]
fn test_postprocess_confidence_filtering() {
    let mut data = vec![0.0; 56 * 2];

    // Detection 0: high confidence
    fill_detection(&mut data, 2, 0, 320.0, 320.0, 100.0, 100.0, 0.8);
    // Detection 1: low confidence
    fill_detection(&mut data, 2, 1, 100.0, 100.0, 50.0, 50.0, 0.1);

    let output = Tensor::new(vec![1, 56, 2], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();
    assert_eq!(detections.len(), 1);
    assert!((detections[0].confidence - 0.8).abs() < 0.01);
}

#[test]
fn test_postprocess_nms_suppression() {
    let mut data = vec![0.0; 56 * 2];

    // Detection 0: confidence 0.9
    fill_detection(&mut data, 2, 0, 320.0, 320.0, 100.0, 100.0, 0.9);
    // Detection 1: confidence 0.7, almost same position (high IoU)
    fill_detection(&mut data, 2, 1, 325.0, 325.0, 100.0, 100.0, 0.7);

    let output = Tensor::new(vec![1, 56, 2], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();
    assert_eq!(detections.len(), 1);
    assert!((detections[0].confidence - 0.9).abs() < 0.01);
}

#[test]
fn test_postprocess_nms_identical_confidence_deterministic() {
    // 3 non-overlapping detections with identical confidence
    // Sort should be stable (preserve original order by index)
    let mut data = vec![0.0; 56 * 3];

    // All three at different positions, same confidence
    fill_detection(&mut data, 3, 0, 100.0, 100.0, 50.0, 50.0, 0.8);
    fill_detection(&mut data, 3, 1, 300.0, 300.0, 50.0, 50.0, 0.8);
    fill_detection(&mut data, 3, 2, 500.0, 500.0, 50.0, 50.0, 0.8);

    let output = Tensor::new(vec![1, 56, 3], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();

    // All 3 should survive NMS (non-overlapping)
    assert_eq!(detections.len(), 3);

    // All should have identical confidence
    for det in &detections {
        assert!((det.confidence - 0.8).abs() < 0.01);
    }

    // Run again to verify determinism
    let data2 = {
        let mut d = vec![0.0; 56 * 3];
        fill_detection(&mut d, 3, 0, 100.0, 100.0, 50.0, 50.0, 0.8);
        fill_detection(&mut d, 3, 1, 300.0, 300.0, 50.0, 50.0, 0.8);
        fill_detection(&mut d, 3, 2, 500.0, 500.0, 50.0, 50.0, 0.8);
        d
    };
    let output2 = Tensor::new(vec![1, 56, 3], data2).unwrap();
    let detections2 = postprocess(&output2, &letterbox, 0.25, 0.45).unwrap();

    // Same order both times
    for (a, b) in detections.iter().zip(detections2.iter()) {
        assert_eq!(a.bbox.origin.x, b.bbox.origin.x);
        assert_eq!(a.bbox.origin.y, b.bbox.origin.y);
    }
}

#[test]
fn test_postprocess_coordinate_rescaling_positive() {
    let mut data = vec![0.0; 56];

    // Detection at (400, 400) in model space
    data[0] = 400.0; // cx
    data[1] = 400.0; // cy
    data[2] = 100.0; // w
    data[3] = 100.0; // h
    data[4] = 0.8; // confidence
    data[5] = 400.0; // kp0_x
    data[6] = 400.0; // kp0_y
    data[7] = 0.9; // kp0_vis
    for i in 1..17 {
        data[5 + i * 3] = 400.0;
        data[5 + i * 3 + 1] = 400.0;
        data[5 + i * 3 + 2] = 0.5;
    }

    let output = Tensor::new(vec![1, 56, 1], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 2.0,
        pad_x: 160.0,
        pad_y: 160.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();
    assert_eq!(detections.len(), 1);

    // Model space (400, 400) → subtract pad → (240, 240) → divide by scale → (120, 120)
    let kp0 = &detections[0].keypoints[0];
    assert!((kp0.position.x - 120.0).abs() < 1.0);
    assert!((kp0.position.y - 120.0).abs() < 1.0);
    assert!((kp0.confidence - 0.9).abs() < 0.01);
}

#[test]
fn test_postprocess_empty_input() {
    let output = Tensor::new(vec![1, 56, 0], vec![]).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();
    assert_eq!(detections.len(), 0);
}

#[test]
fn test_postprocess_all_below_threshold() {
    let mut data = vec![0.0; 56 * 3];
    for i in 0..3 {
        fill_detection(&mut data, 3, i, 100.0, 100.0, 50.0, 50.0, 0.1);
    }

    let output = Tensor::new(vec![1, 56, 3], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();
    assert_eq!(detections.len(), 0);
}

#[test]
fn test_postprocess_bbox_conversion() {
    let mut data = vec![0.0; 56];
    // Box at center (100, 100) with size 40x40
    fill_detection(&mut data, 1, 0, 100.0, 100.0, 40.0, 40.0, 0.8);

    let output = Tensor::new(vec![1, 56, 1], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();
    assert_eq!(detections.len(), 1);
    // Expected Rect: origin (80, 80), size (40, 40)
    assert!((detections[0].bbox.origin.x - 80.0).abs() < 0.1);
    assert!((detections[0].bbox.origin.y - 80.0).abs() < 0.1);
    assert!((detections[0].bbox.size.x - 40.0).abs() < 0.1);
    assert!((detections[0].bbox.size.y - 40.0).abs() < 0.1);
}

#[test]
fn test_postprocess_round_trip_coordinates() {
    // Round-trip test: define bbox in original space, convert to model space, postprocess back
    // Original image: 480x640 (H x W)
    // Original bbox: center at (200, 300), size 100x80 in original space
    let orig_cx = 200.0_f32;
    let orig_cy = 300.0_f32;
    let orig_w = 100.0_f32;
    let orig_h = 80.0_f32;

    // Compute letterbox parameters for 480x640 → 640x640
    // scale = min(640/640, 640/480) = min(1.0, 1.333) = 1.0
    let scale = (640.0_f32 / 640.0).min(640.0 / 480.0);
    let new_w = (640.0 * scale) as usize; // 640
    let new_h = (480.0 * scale) as usize; // 480
    let pad_x = ((640 - new_w) / 2) as f32; // 0
    let pad_y = ((640 - new_h) / 2) as f32; // 80

    let letterbox = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };

    // Convert original coords to model space
    let model_cx = orig_cx * scale + pad_x;
    let model_cy = orig_cy * scale + pad_y;
    let model_w = orig_w * scale;
    let model_h = orig_h * scale;

    // Create synthetic output tensor
    let mut data = vec![0.0; 56];
    data[0] = model_cx;
    data[1] = model_cy;
    data[2] = model_w;
    data[3] = model_h;
    data[4] = 0.9; // confidence

    let output = Tensor::new(vec![1, 56, 1], data).unwrap();
    let detections = postprocess(&output, &letterbox, 0.25, 0.45).unwrap();

    assert_eq!(detections.len(), 1);

    // Recovered bbox center should match original within ±1.0 pixel
    let recovered_cx = detections[0].bbox.origin.x + detections[0].bbox.size.x / 2.0;
    let recovered_cy = detections[0].bbox.origin.y + detections[0].bbox.size.y / 2.0;

    assert!(
        (recovered_cx - orig_cx).abs() < 1.0,
        "Round-trip cx failed: {} vs {}",
        recovered_cx,
        orig_cx
    );
    assert!(
        (recovered_cy - orig_cy).abs() < 1.0,
        "Round-trip cy failed: {} vs {}",
        recovered_cy,
        orig_cy
    );
    assert!(
        (detections[0].bbox.size.x - orig_w).abs() < 1.0,
        "Round-trip w failed: {} vs {}",
        detections[0].bbox.size.x,
        orig_w
    );
    assert!(
        (detections[0].bbox.size.y - orig_h).abs() < 1.0,
        "Round-trip h failed: {} vs {}",
        detections[0].bbox.size.y,
        orig_h
    );
}
