use crate::InferError;
use deli_math::{Rect, Tensor, Vec2};

use super::types::{Keypoint, LetterboxInfo, PoseDetection};

/// Compute Intersection over Union (IoU) between two bounding boxes
///
/// Returns 0.0 for non-overlapping boxes or zero-area boxes (no division by zero).
pub fn iou(a: &Rect<f32>, b: &Rect<f32>) -> f32 {
    // Check for zero-area boxes
    if a.size.x <= 0.0 || a.size.y <= 0.0 || b.size.x <= 0.0 || b.size.y <= 0.0 {
        return 0.0;
    }

    // Compute intersection area
    let intersection = a.intersection(*b);
    let intersection_area = match intersection {
        Some(rect) => rect.size.x * rect.size.y,
        None => 0.0,
    };

    // Compute union area
    let area_a = a.size.x * a.size.y;
    let area_b = b.size.x * b.size.y;
    let union_area = area_a + area_b - intersection_area;

    // Avoid division by zero
    if union_area <= 0.0 {
        return 0.0;
    }

    intersection_area / union_area
}

/// Post-process YOLO pose model output
///
/// Takes raw model output tensor [1, 56, N], applies confidence filtering, NMS,
/// and coordinate rescaling to produce final pose detections.
///
/// # Arguments
/// * `output` - Raw model output tensor with shape [1, 56, N]
/// * `letterbox` - Letterbox parameters for coordinate rescaling
/// * `conf_threshold` - Minimum confidence threshold (default: 0.25)
/// * `iou_threshold` - IoU threshold for NMS (default: 0.45)
///
/// # Returns
/// Vector of `PoseDetection` sorted by confidence descending, or `InferError::ShapeMismatch`
/// if the output tensor has an unexpected shape.
pub fn postprocess(
    output: &Tensor<f32>,
    letterbox: &LetterboxInfo,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<PoseDetection>, InferError> {
    // Validate output shape
    if output.shape.len() != 3 || output.shape[0] != 1 || output.shape[1] != 56 {
        return Err(InferError::ShapeMismatch {
            expected: format!("[1, 56, N]"),
            got: format!("{:?}", output.shape),
        });
    }

    let n = output.shape[2];
    if n == 0 {
        return Ok(Vec::new());
    }

    // Transpose from [1, 56, N] to [N, 56] for per-detection iteration
    // In the flat data, element at [0, row, col] is at index: row * N + col
    let mut candidates = Vec::new();

    for i in 0..n {
        // Extract detection data (column i in the transposed view)
        let cx = output.data[0 * n + i];
        let cy = output.data[1 * n + i];
        let w = output.data[2 * n + i];
        let h = output.data[3 * n + i];
        let confidence = output.data[4 * n + i];

        // Filter by confidence threshold
        if confidence < conf_threshold {
            continue;
        }

        // Extract keypoints (17 keypoints x 3 values each)
        let mut keypoints = Vec::with_capacity(17);
        for kp_idx in 0..17 {
            let base = 5 + kp_idx * 3;
            let x = output.data[base * n + i];
            let y = output.data[(base + 1) * n + i];
            let vis = output.data[(base + 2) * n + i];

            // Rescale keypoint coordinates from model space to original image
            let rescaled_x = (x - letterbox.pad_x) / letterbox.scale;
            let rescaled_y = (y - letterbox.pad_y) / letterbox.scale;

            keypoints.push(Keypoint {
                position: Vec2::new(rescaled_x, rescaled_y),
                confidence: vis,
            });
        }

        // Convert center-based bbox to origin-based Rect
        // Rescale bbox coordinates
        let rescaled_cx = (cx - letterbox.pad_x) / letterbox.scale;
        let rescaled_cy = (cy - letterbox.pad_y) / letterbox.scale;
        let rescaled_w = w / letterbox.scale;
        let rescaled_h = h / letterbox.scale;

        // Convert from center to top-left origin
        let origin_x = rescaled_cx - rescaled_w / 2.0;
        let origin_y = rescaled_cy - rescaled_h / 2.0;

        let bbox = Rect::new(
            Vec2::new(origin_x, origin_y),
            Vec2::new(rescaled_w, rescaled_h),
        );

        candidates.push((
            confidence,
            PoseDetection {
                bbox,
                confidence,
                keypoints: keypoints.try_into().unwrap(),
            },
        ));
    }

    // Sort by confidence descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Apply NMS (greedy algorithm)
    let mut keep = Vec::new();
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(candidates[i].1.clone());

        // Suppress overlapping detections
        for j in (i + 1)..candidates.len() {
            if suppressed[j] {
                continue;
            }

            let iou_val = iou(&candidates[i].1.bbox, &candidates[j].1.bbox);
            if iou_val > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Ok(keep)
}
