use crate::InferError;
use deli_base::{Rect, Tensor, Vec2};

use super::types::{Keypoint, LetterboxInfo, PoseDetection};

/// Number of values per detection in YOLO26 end-to-end pose output:
/// 4 (x1,y1,x2,y2) + 1 (score) + 1 (class_id) + 51 (17*3 keypoints) = 57
const DET_VALUES: usize = 57;

/// Model input size (matches TARGET_SIZE in preprocess.rs).
/// Used to denormalize coordinates from 0..1 to pixel space.
const MODEL_SIZE: f32 = 640.0;

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

/// Post-process YOLO26 end-to-end pose model output
///
/// Takes raw model output tensor [1, N, 57], applies confidence filtering
/// and coordinate rescaling to produce final pose detections. NMS is not needed
/// because the YOLO26 end-to-end model applies NMS internally.
///
/// Each detection row contains 57 values:
/// - [0..4]: bounding box in xyxy format (x1, y1, x2, y2)
/// - [4]: confidence score
/// - [5]: class id
/// - [6..57]: 17 keypoints × 3 values (x, y, visibility)
///
/// # Arguments
/// * `output` - Raw model output tensor with shape [1, N, 57]
/// * `letterbox` - Letterbox parameters for coordinate rescaling
/// * `conf_threshold` - Minimum confidence threshold (default: 0.25)
///
/// # Returns
/// Vector of `PoseDetection` sorted by confidence descending, or `InferError::ShapeMismatch`
/// if the output tensor has an unexpected shape.
pub fn postprocess(
    output: &Tensor<f32>,
    letterbox: &LetterboxInfo,
    conf_threshold: f32,
) -> Result<Vec<PoseDetection>, InferError> {
    // Validate output shape: [1, N, 57]
    if output.shape.len() != 3 || output.shape[0] != 1 || output.shape[2] != DET_VALUES {
        println!(
            "postprocess: shape mismatch — expected [1, N, {}], got {:?}",
            DET_VALUES, output.shape
        );
        return Err(InferError::ShapeMismatch {
            expected: format!("[1, N, {}]", DET_VALUES),
            got: format!("{:?}", output.shape),
        });
    }

    let n = output.shape[1];

    if n == 0 {
        return Ok(Vec::new());
    }

    let mut candidates: Vec<PoseDetection> = Vec::new();

    for i in 0..n {
        let base = i * DET_VALUES;

        // Read xyxy bbox — coordinates are normalized (0..1), denormalize to model pixels
        let x1 = output.data[base] * MODEL_SIZE;
        let y1 = output.data[base + 1] * MODEL_SIZE;
        let x2 = output.data[base + 2] * MODEL_SIZE;
        let y2 = output.data[base + 3] * MODEL_SIZE;
        let confidence = output.data[base + 4];
        let class_id = output.data[base + 5];

        // Filter by confidence threshold
        if confidence < conf_threshold {
            continue;
        }

        // Extract keypoints (17 keypoints × 3 values each, starting at offset 6)
        let mut keypoints = Vec::with_capacity(17);
        for kp_idx in 0..17 {
            let kp_base = base + 6 + kp_idx * 3;
            let x = output.data[kp_base] * MODEL_SIZE;
            let y = output.data[kp_base + 1] * MODEL_SIZE;
            let vis = output.data[kp_base + 2];

            // Rescale keypoint coordinates from model space to original image
            let rescaled_x = (x - letterbox.pad_x) / letterbox.scale;
            let rescaled_y = (y - letterbox.pad_y) / letterbox.scale;

            keypoints.push(Keypoint {
                position: Vec2::new(rescaled_x, rescaled_y),
                confidence: vis,
            });
        }

        // Convert xyxy bbox to Rect (origin + size), rescaling from model space
        let rescaled_x1 = (x1 - letterbox.pad_x) / letterbox.scale;
        let rescaled_y1 = (y1 - letterbox.pad_y) / letterbox.scale;
        let rescaled_x2 = (x2 - letterbox.pad_x) / letterbox.scale;
        let rescaled_y2 = (y2 - letterbox.pad_y) / letterbox.scale;

        let bbox = Rect::new(
            Vec2::new(rescaled_x1, rescaled_y1),
            Vec2::new(rescaled_x2 - rescaled_x1, rescaled_y2 - rescaled_y1),
        );

        candidates.push(PoseDetection {
            bbox,
            confidence,
            keypoints: keypoints.try_into().unwrap(),
        });
    }

    // Sort by confidence descending
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(candidates)
}
