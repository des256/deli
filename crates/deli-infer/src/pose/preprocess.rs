use crate::InferError;
use deli_math::Tensor;

use super::types::LetterboxInfo;

const TARGET_SIZE: usize = 640;
const PAD_COLOR: f32 = 114.0 / 255.0; // Gray padding in normalized range

/// Preprocess an image for YOLO pose inference
///
/// Takes an image tensor in HWC format (height, width, 3 channels) with pixel values in [0, 255]
/// and returns a preprocessed tensor in NCHW format (1, 3, 640, 640) with values in [0.0, 1.0].
///
/// The preprocessing includes:
/// - Letterbox resize to 640x640 maintaining aspect ratio
/// - HWC -> NCHW transpose
/// - Rescale from [0, 255] to [0.0, 1.0]
///
/// Returns the preprocessed tensor and letterbox info for coordinate rescaling.
pub fn preprocess(image: &Tensor<f32>) -> Result<(Tensor<f32>, LetterboxInfo), InferError> {
    // Validate input shape
    if image.shape.len() != 3 {
        return Err(InferError::ShapeMismatch {
            expected: format!("[H, W, 3]"),
            got: format!("{:?}", image.shape),
        });
    }
    let [h, w, c] = [image.shape[0], image.shape[1], image.shape[2]];
    if c != 3 {
        return Err(InferError::ShapeMismatch {
            expected: format!("3 channels"),
            got: format!("{} channels", c),
        });
    }

    // Compute scale factor (min of width_scale, height_scale)
    let scale = (TARGET_SIZE as f32 / w as f32).min(TARGET_SIZE as f32 / h as f32);

    // Compute new dimensions after scaling
    let new_w = (w as f32 * scale) as usize;
    let new_h = (h as f32 * scale) as usize;

    // Compute padding
    let pad_x = ((TARGET_SIZE - new_w) / 2) as f32;
    let pad_y = ((TARGET_SIZE - new_h) / 2) as f32;

    // Resize image using nearest-neighbor interpolation
    let mut resized_data = vec![0.0; new_h * new_w * 3];
    for out_y in 0..new_h {
        for out_x in 0..new_w {
            // Map output pixel to source pixel using nearest-neighbor
            let src_y = ((out_y as f32 / scale).floor() as usize).min(h - 1);
            let src_x = ((out_x as f32 / scale).floor() as usize).min(w - 1);

            // Copy RGB values
            for ch in 0..3 {
                let src_idx = (src_y * w + src_x) * 3 + ch;
                let dst_idx = (out_y * new_w + out_x) * 3 + ch;
                resized_data[dst_idx] = image.data[src_idx];
            }
        }
    }

    // Apply letterbox (pad to 640x640) and transpose to NCHW format
    // Output shape: [1, 3, 640, 640]
    let mut nchw_data = vec![PAD_COLOR; 1 * 3 * TARGET_SIZE * TARGET_SIZE];

    let pad_x_int = pad_x as usize;
    let pad_y_int = pad_y as usize;

    for ch in 0..3 {
        for y in 0..new_h {
            for x in 0..new_w {
                let src_idx = (y * new_w + x) * 3 + ch;
                let dst_y = y + pad_y_int;
                let dst_x = x + pad_x_int;

                // NCHW layout: channel, height, width
                let dst_idx = ch * (TARGET_SIZE * TARGET_SIZE) + dst_y * TARGET_SIZE + dst_x;

                // Rescale from [0, 255] to [0.0, 1.0]
                nchw_data[dst_idx] = resized_data[src_idx] / 255.0;
            }
        }
    }

    let preprocessed = Tensor::new(vec![1, 3, TARGET_SIZE, TARGET_SIZE], nchw_data)
        .map_err(|e| InferError::BackendError(format!("failed to create tensor: {}", e)))?;

    let letterbox = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };

    Ok((preprocessed, letterbox))
}
