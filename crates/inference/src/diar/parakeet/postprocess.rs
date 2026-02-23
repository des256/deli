use crate::diar::parakeet::{DiarizationConfig, SpeakerSegment};

const NUM_SPEAKERS: usize = 4;
const FRAME_DURATION: f32 = 0.08;

/// Apply median filter per speaker over time axis.
pub(crate) fn median_filter(preds: &[f32], num_frames: usize, window: usize) -> Vec<f32> {
    let mut result = vec![0.0; num_frames * NUM_SPEAKERS];

    for spk_idx in 0..NUM_SPEAKERS {
        // Extract this speaker's original predictions (read-only)
        let original: Vec<f32> = (0..num_frames)
            .map(|frame_idx| preds[frame_idx * NUM_SPEAKERS + spk_idx])
            .collect();

        // Apply median filter reading from original, writing to result
        for frame_idx in 0..num_frames {
            let half_window = window / 2;
            let start = frame_idx.saturating_sub(half_window);
            let end = (frame_idx + half_window + 1).min(num_frames);

            let mut window_vals: Vec<f32> = original[start..end].to_vec();
            window_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = window_vals[window_vals.len() / 2];
            result[frame_idx * NUM_SPEAKERS + spk_idx] = median;
        }
    }

    result
}

/// Binarize predictions using hysteresis thresholding to produce speaker segments.
pub(crate) fn binarize(
    preds: &[f32],
    num_frames: usize,
    config: &DiarizationConfig,
) -> Vec<SpeakerSegment> {
    let mut segments = Vec::new();

    for spk_idx in 0..NUM_SPEAKERS {
        // Extract this speaker's predictions
        let spk_preds: Vec<f32> = (0..num_frames)
            .map(|frame_idx| preds[frame_idx * NUM_SPEAKERS + spk_idx])
            .collect();

        // Hysteresis binarization
        let mut is_active = false;
        let mut segment_start: Option<usize> = None;

        for frame_idx in 0..num_frames {
            let prob = spk_preds[frame_idx];

            if !is_active && prob >= config.onset {
                // Start new segment
                is_active = true;
                segment_start = Some(frame_idx);
            } else if is_active && prob < config.offset {
                // End segment
                is_active = false;
                if let Some(start) = segment_start {
                    let start_time = start as f32 * FRAME_DURATION;
                    let end_time = frame_idx as f32 * FRAME_DURATION;

                    // Apply min duration filter (0.1s)
                    if end_time - start_time >= 0.1 {
                        segments.push(SpeakerSegment {
                            start: start_time,
                            end: end_time,
                            speaker_id: spk_idx,
                        });
                    }
                }
                segment_start = None;
            }
        }

        // Close any remaining segment at the end
        if is_active {
            if let Some(start) = segment_start {
                let start_time = start as f32 * FRAME_DURATION;
                let end_time = num_frames as f32 * FRAME_DURATION;

                if end_time - start_time >= 0.1 {
                    segments.push(SpeakerSegment {
                        start: start_time,
                        end: end_time,
                        speaker_id: spk_idx,
                    });
                }
            }
        }
    }

    // Sort segments by start time (NaN-safe)
    segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

    segments
}
