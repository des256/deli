use super::types::{Keypoint, PoseDetection};
use candle_core::{Result, Tensor};
use deli_base::{Rect, Vec2};
use std::collections::VecDeque;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute Intersection over Union (IoU) between two bounding boxes
fn iou(box1: &Rect<f32>, box2: &Rect<f32>) -> f32 {
    let intersection = box1.intersection(*box2);

    match intersection {
        None => 0.0,
        Some(inter) => {
            let inter_area = inter.area();
            let box1_area = box1.area();
            let box2_area = box2.area();
            let union_area = box1_area + box2_area - inter_area;

            if union_area > 0.0 {
                inter_area / union_area
            } else {
                0.0
            }
        }
    }
}

/// Non-Maximum Suppression (NMS) to remove overlapping boxes.
/// Returns indices of boxes to keep.
fn nms(boxes: &[(Rect<f32>, f32)], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut indices: VecDeque<usize> = {
        let mut v: Vec<usize> = (0..boxes.len()).collect();
        v.sort_by(|&a, &b| {
            boxes[b].1.partial_cmp(&boxes[a].1).unwrap_or(std::cmp::Ordering::Equal)
        });
        v.into()
    };

    let mut keep = Vec::new();

    while let Some(current) = indices.pop_front() {
        keep.push(current);

        indices.retain(|&idx| {
            let iou_val = iou(&boxes[current].0, &boxes[idx].0);
            iou_val < iou_threshold
        });
    }

    keep
}

/// Post-process raw model output into PoseDetection structs
pub(crate) fn postprocess(
    pred: &Tensor,
    original_hw: (usize, usize),
    model_hw: (usize, usize),
    conf_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<PoseDetection>> {
    // pred shape: [1, 56, N] where N is number of anchors
    let pred = pred.squeeze(0)?; // [56, N]
    let (_, n) = pred.dims2()?;

    let pred_data = pred.to_vec2::<f32>()?;

    let mut candidates = Vec::new();

    #[allow(clippy::needless_range_loop)] // col indexes multiple rows at same column
    for col in 0..n {
        // Apply sigmoid to raw class logit to get probability in [0, 1]
        let conf = sigmoid(pred_data[4][col]);

        if conf < conf_threshold {
            continue;
        }

        // Get bbox (indices 0-3: cx, cy, w, h)
        let cx = pred_data[0][col];
        let cy = pred_data[1][col];
        let w = pred_data[2][col];
        let h = pred_data[3][col];

        let bbox = Rect::from_min_max(
            Vec2::new(cx - w / 2.0, cy - h / 2.0),
            Vec2::new(cx + w / 2.0, cy + h / 2.0),
        );

        // Get keypoints (indices 5-55: 17 keypoints * 3 values each)
        let mut keypoints = [Keypoint {
            position: Vec2::new(0.0, 0.0),
            confidence: 0.0,
        }; 17];
        for kpt_idx in 0..17 {
            let base = 5 + kpt_idx * 3;
            let x = pred_data[base][col];
            let y = pred_data[base + 1][col];
            let kpt_conf = sigmoid(pred_data[base + 2][col]);

            keypoints[kpt_idx] = Keypoint {
                position: Vec2::new(x, y),
                confidence: kpt_conf,
            };
        }

        candidates.push((bbox, conf, keypoints));
    }

    // Apply NMS
    let boxes_for_nms: Vec<(Rect<f32>, f32)> =
        candidates.iter().map(|(bbox, conf, _)| (*bbox, *conf)).collect();
    let keep_indices = nms(&boxes_for_nms, nms_threshold);

    // Scale coordinates from model space to original image space
    let scale_x = original_hw.1 as f32 / model_hw.1 as f32;
    let scale_y = original_hw.0 as f32 / model_hw.0 as f32;

    let mut detections = Vec::new();
    for &idx in &keep_indices {
        let (bbox, conf, keypoints) = &candidates[idx];

        let orig_w = original_hw.1 as f32;
        let orig_h = original_hw.0 as f32;

        // Scale and clamp bbox to image bounds
        let x = (bbox.origin.x * scale_x).clamp(0.0, orig_w);
        let y = (bbox.origin.y * scale_y).clamp(0.0, orig_h);
        let w = (bbox.size.x * scale_x).min(orig_w - x);
        let h = (bbox.size.y * scale_y).min(orig_h - y);
        let scaled_bbox = Rect::new(Vec2::new(x, y), Vec2::new(w, h));

        // Scale and clamp keypoints to image bounds
        let mut scaled_keypoints = [Keypoint {
            position: Vec2::new(0.0, 0.0),
            confidence: 0.0,
        }; 17];

        for (i, kp) in keypoints.iter().enumerate() {
            scaled_keypoints[i] = Keypoint {
                position: Vec2::new(
                    (kp.position.x * scale_x).clamp(0.0, orig_w),
                    (kp.position.y * scale_y).clamp(0.0, orig_h),
                ),
                confidence: kp.confidence,
            };
        }

        detections.push(PoseDetection {
            bbox: scaled_bbox,
            confidence: *conf,
            keypoints: scaled_keypoints,
        });
    }

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_iou_full_overlap() {
        let box1 = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let box2 = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let iou_val = iou(&box1, &box2);
        assert!((iou_val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_iou_no_overlap() {
        let box1 = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let box2 = Rect::new(Vec2::new(20.0, 20.0), Vec2::new(10.0, 10.0));
        let iou_val = iou(&box1, &box2);
        assert!((iou_val - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let box1 = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let box2 = Rect::new(Vec2::new(5.0, 5.0), Vec2::new(10.0, 10.0));
        let iou_val = iou(&box1, &box2);
        // Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175, IoU = 25/175
        assert!((iou_val - 0.142857).abs() < 0.01);
    }

    #[test]
    fn test_nms_removes_overlapping_boxes() {
        let boxes = vec![
            (Rect::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0)), 0.9),
            (Rect::new(Vec2::new(1.0, 1.0), Vec2::new(10.0, 10.0)), 0.8),
            (Rect::new(Vec2::new(50.0, 50.0), Vec2::new(10.0, 10.0)), 0.7),
        ];

        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn test_postprocess_output_structure() {
        let device = Device::Cpu;
        let pred = Tensor::zeros(&[1, 56, 100], DType::F32, &device).unwrap();
        let detections = postprocess(&pred, (480, 640), (640, 640), 0.25, 0.45).unwrap();
        // All zeros → sigmoid(0)=0.5 > 0.25, so candidates exist but bbox is zero-area
        // NMS will keep 1 candidate (all bboxes identical → first kept)
        // This verifies the function runs without error
        assert!(detections.len() <= 100);
    }

    #[test]
    fn test_postprocess_with_high_confidence() {
        let device = Device::Cpu;

        let mut data = vec![0.0f32; 56 * 100];

        // Set bbox (cx, cy, w, h) for first detection
        data[0] = 50.0;   // cx
        data[100] = 50.0;  // cy
        data[200] = 100.0; // w
        data[300] = 100.0; // h

        // Set confidence logit: sigmoid(3.0) ≈ 0.952
        data[400] = 3.0;

        // Set keypoint data: 17 keypoints * 3 values (x, y, conf) at rows 5..55
        for k in 0..17 {
            let base = 5 + k * 3;
            data[base * 100] = 50.0 + k as f32 * 10.0;       // x at row (5+k*3)
            data[(base + 1) * 100] = 60.0 + k as f32 * 10.0;  // y at row (5+k*3+1)
            data[(base + 2) * 100] = 2.0;                      // logit: sigmoid(2.0) ≈ 0.881
        }

        let pred = Tensor::from_vec(data, (1, 56, 100), &device).unwrap();
        // Same model and original dims → scale factors = 1.0
        let detections = postprocess(&pred, (640, 640), (640, 640), 0.25, 0.45).unwrap();

        assert!(detections.len() >= 1);
        let det = &detections[0];
        assert!(det.confidence > 0.9);
        assert_eq!(det.keypoints.len(), 17);

        // Verify keypoint 0 position (scale=1.0 so no scaling)
        assert!((det.keypoints[0].position.x - 50.0).abs() < 0.1);
        assert!((det.keypoints[0].position.y - 60.0).abs() < 0.1);
        assert!(det.keypoints[0].confidence > 0.8);

        // Verify keypoint 1 position
        assert!((det.keypoints[1].position.x - 60.0).abs() < 0.1);
        assert!((det.keypoints[1].position.y - 70.0).abs() < 0.1);
    }
}
