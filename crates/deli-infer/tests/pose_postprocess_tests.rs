#![cfg(feature = "onnx")]

use deli_infer::pose::{iou, postprocess, LetterboxInfo};
use deli_base::{Rect, Tensor, Vec2};

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

/// Number of values per detection in YOLO26 end-to-end pose output:
/// 4 (x1,y1,x2,y2) + 1 (score) + 1 (class_id) + 51 (17*3 keypoints) = 57
const DET_SIZE: usize = 57;

/// Fill a detection row in [1, N, 57] tensor data.
/// All coordinates are normalized (0..1). Confidence is absolute.
fn fill_detection(
    data: &mut [f32],
    det_idx: usize,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
) {
    let base = det_idx * DET_SIZE;
    data[base] = x1;
    data[base + 1] = y1;
    data[base + 2] = x2;
    data[base + 3] = y2;
    data[base + 4] = conf;
    data[base + 5] = 0.0; // class_id
    // keypoints default to 0.0 (already zeroed)
}

#[test]
fn test_postprocess_invalid_shape_returns_error() {
    // Shape [1, 10, 5] is invalid (should be [1, N, 57])
    let data = vec![0.0; 10 * 5];
    let output = Tensor::new(vec![1, 10, 5], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let result = postprocess(&output, &letterbox, 0.25);
    assert!(result.is_err());
}

#[test]
fn test_postprocess_confidence_filtering() {
    let n = 2;
    let mut data = vec![0.0; n * DET_SIZE];

    // Detection 0: high confidence, normalized bbox
    fill_detection(&mut data, 0, 0.4, 0.4, 0.6, 0.6, 0.8);
    // Detection 1: low confidence
    fill_detection(&mut data, 1, 0.1, 0.1, 0.2, 0.2, 0.1);

    let output = Tensor::new(vec![1, n, DET_SIZE], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25).unwrap();
    assert_eq!(detections.len(), 1);
    assert!((detections[0].confidence - 0.8).abs() < 0.01);
}

#[test]
fn test_postprocess_sorted_by_confidence() {
    let n = 3;
    let mut data = vec![0.0; n * DET_SIZE];

    // Three non-overlapping detections with different confidences (normalized coords)
    fill_detection(&mut data, 0, 0.1, 0.1, 0.2, 0.2, 0.5);
    fill_detection(&mut data, 1, 0.4, 0.4, 0.5, 0.5, 0.9);
    fill_detection(&mut data, 2, 0.7, 0.7, 0.8, 0.8, 0.7);

    let output = Tensor::new(vec![1, n, DET_SIZE], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25).unwrap();
    assert_eq!(detections.len(), 3);
    assert!((detections[0].confidence - 0.9).abs() < 0.01);
    assert!((detections[1].confidence - 0.7).abs() < 0.01);
    assert!((detections[2].confidence - 0.5).abs() < 0.01);
}

#[test]
fn test_postprocess_denormalization_no_padding() {
    // Model input 640x640, original image 640x640 (no padding, scale=1.0)
    // Normalized coord 0.5 → pixel 320.0 in model space → 320.0 in original
    let n = 1;
    let mut data = vec![0.0; n * DET_SIZE];

    let base = 0;
    // Normalized bbox: center region of image
    data[base] = 0.25;     // x1 = 0.25
    data[base + 1] = 0.25; // y1 = 0.25
    data[base + 2] = 0.75; // x2 = 0.75
    data[base + 3] = 0.75; // y2 = 0.75
    data[base + 4] = 0.8;  // confidence
    data[base + 5] = 0.0;  // class_id
    // kp0 at normalized (0.5, 0.5), vis 0.9
    data[base + 6] = 0.5;
    data[base + 7] = 0.5;
    data[base + 8] = 0.9;

    let output = Tensor::new(vec![1, n, DET_SIZE], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25).unwrap();
    assert_eq!(detections.len(), 1);

    // 0.25 * 640 = 160, 0.75 * 640 = 480 → origin=(160,160), size=(320,320)
    assert!((detections[0].bbox.origin.x - 160.0).abs() < 1.0);
    assert!((detections[0].bbox.origin.y - 160.0).abs() < 1.0);
    assert!((detections[0].bbox.size.x - 320.0).abs() < 1.0);
    assert!((detections[0].bbox.size.y - 320.0).abs() < 1.0);

    // kp0: 0.5 * 640 = 320
    let kp0 = &detections[0].keypoints[0];
    assert!((kp0.position.x - 320.0).abs() < 1.0);
    assert!((kp0.position.y - 320.0).abs() < 1.0);
    assert!((kp0.confidence - 0.9).abs() < 0.01);
}

#[test]
fn test_postprocess_denormalization_with_padding() {
    // Original image: 480x640 (H x W) → letterbox to 640x640
    // scale = min(640/640, 640/480) = 1.0
    // new_w = 640, new_h = 480, pad_x = 0, pad_y = 80
    //
    // A keypoint at normalized (0.5, 0.5) means:
    //   pixel in model space = 0.5 * 640 = 320
    //   x: (320 - 0) / 1.0 = 320 in original
    //   y: (320 - 80) / 1.0 = 240 in original
    let n = 1;
    let mut data = vec![0.0; n * DET_SIZE];

    let base = 0;
    data[base] = 0.25;     // x1
    data[base + 1] = 0.25; // y1
    data[base + 2] = 0.75; // x2
    data[base + 3] = 0.75; // y2
    data[base + 4] = 0.8;  // confidence
    data[base + 5] = 0.0;  // class_id
    // kp0 at center
    data[base + 6] = 0.5;
    data[base + 7] = 0.5;
    data[base + 8] = 0.9;

    let output = Tensor::new(vec![1, n, DET_SIZE], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 80.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25).unwrap();
    assert_eq!(detections.len(), 1);

    // kp0: x = (0.5*640 - 0)/1.0 = 320, y = (0.5*640 - 80)/1.0 = 240
    let kp0 = &detections[0].keypoints[0];
    assert!((kp0.position.x - 320.0).abs() < 1.0);
    assert!((kp0.position.y - 240.0).abs() < 1.0);

    // bbox: x1 = (0.25*640 - 0)/1.0 = 160, y1 = (0.25*640 - 80)/1.0 = 80
    //        x2 = (0.75*640 - 0)/1.0 = 480, y2 = (0.75*640 - 80)/1.0 = 400
    //        → origin=(160,80), size=(320,320)
    assert!((detections[0].bbox.origin.x - 160.0).abs() < 1.0);
    assert!((detections[0].bbox.origin.y - 80.0).abs() < 1.0);
    assert!((detections[0].bbox.size.x - 320.0).abs() < 1.0);
    assert!((detections[0].bbox.size.y - 320.0).abs() < 1.0);
}

#[test]
fn test_postprocess_empty_detections() {
    let n = 3;
    let mut data = vec![0.0; n * DET_SIZE];
    for i in 0..n {
        fill_detection(&mut data, i, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let output = Tensor::new(vec![1, n, DET_SIZE], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25).unwrap();
    assert_eq!(detections.len(), 0);
}

#[test]
fn test_postprocess_all_below_threshold() {
    let n = 3;
    let mut data = vec![0.0; n * DET_SIZE];
    for i in 0..n {
        fill_detection(&mut data, i, 0.1, 0.1, 0.2, 0.2, 0.1);
    }

    let output = Tensor::new(vec![1, n, DET_SIZE], data).unwrap();
    let letterbox = LetterboxInfo {
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    let detections = postprocess(&output, &letterbox, 0.25).unwrap();
    assert_eq!(detections.len(), 0);
}

#[test]
fn test_postprocess_round_trip_coordinates() {
    // Round-trip test: define bbox in original space, convert to normalized model output,
    // then postprocess back to original space.
    //
    // Original image: 480x640 (H x W) → letterbox to 640x640
    // scale = 1.0, pad_x = 0, pad_y = 80
    let orig_x1 = 150.0_f32;
    let orig_y1 = 260.0_f32;
    let orig_w = 100.0_f32;
    let orig_h = 80.0_f32;
    let orig_x2 = orig_x1 + orig_w;
    let orig_y2 = orig_y1 + orig_h;

    let scale = (640.0_f32 / 640.0).min(640.0 / 480.0);
    let new_w = (640.0 * scale) as usize;
    let new_h = (480.0 * scale) as usize;
    let pad_x = ((640 - new_w) / 2) as f32;
    let pad_y = ((640 - new_h) / 2) as f32;

    let letterbox = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };

    // Original → model-space pixels → normalized
    let model_x1 = orig_x1 * scale + pad_x;
    let model_y1 = orig_y1 * scale + pad_y;
    let model_x2 = orig_x2 * scale + pad_x;
    let model_y2 = orig_y2 * scale + pad_y;
    let norm_x1 = model_x1 / 640.0;
    let norm_y1 = model_y1 / 640.0;
    let norm_x2 = model_x2 / 640.0;
    let norm_y2 = model_y2 / 640.0;

    let mut data = vec![0.0; DET_SIZE];
    data[0] = norm_x1;
    data[1] = norm_y1;
    data[2] = norm_x2;
    data[3] = norm_y2;
    data[4] = 0.9;
    data[5] = 0.0;

    let output = Tensor::new(vec![1, 1, DET_SIZE], data).unwrap();
    let detections = postprocess(&output, &letterbox, 0.25).unwrap();

    assert_eq!(detections.len(), 1);

    assert!(
        (detections[0].bbox.origin.x - orig_x1).abs() < 1.0,
        "Round-trip x1 failed: {} vs {}",
        detections[0].bbox.origin.x,
        orig_x1
    );
    assert!(
        (detections[0].bbox.origin.y - orig_y1).abs() < 1.0,
        "Round-trip y1 failed: {} vs {}",
        detections[0].bbox.origin.y,
        orig_y1
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
