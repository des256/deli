const NUM_SPEAKERS: usize = 4;
const PRED_SCORE_THRESHOLD: f32 = 0.25;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;

/// Compute log-likelihood ratio scores for each speaker per frame.
///
/// Returns scores [num_frames * NUM_SPEAKERS] flattened.
pub(crate) fn get_log_pred_scores(preds: &[f32], num_frames: usize) -> Vec<f32> {
    let mut scores = Vec::with_capacity(num_frames * NUM_SPEAKERS);

    for frame_idx in 0..num_frames {
        let offset = frame_idx * NUM_SPEAKERS;
        let frame_preds = &preds[offset..offset + NUM_SPEAKERS];

        // Compute sum of all speaker probs
        let total: f32 = frame_preds.iter().sum();
        let remaining = (1.0 - total).max(1e-10);

        // Log-likelihood ratio: log(p_speaker / (1 - sum(all_speakers)))
        for &p in frame_preds {
            let score = (p.max(1e-10) / remaining).ln();
            scores.push(score);
        }
    }

    scores
}

/// Set non-positive scores to NEG_INFINITY.
pub(crate) fn disable_low_scores(scores: &mut [f32], num_frames: usize) {
    for frame_idx in 0..num_frames {
        for spk_idx in 0..NUM_SPEAKERS {
            let idx = frame_idx * NUM_SPEAKERS + spk_idx;
            if scores[idx] <= PRED_SCORE_THRESHOLD {
                scores[idx] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Boost top-K scores per speaker.
pub(crate) fn boost_topk_scores(scores: &mut [f32], num_frames: usize) {
    for spk_idx in 0..NUM_SPEAKERS {
        // Extract scores for this speaker across all frames
        let mut spk_scores: Vec<(usize, f32)> = (0..num_frames)
            .map(|frame_idx| {
                let idx = frame_idx * NUM_SPEAKERS + spk_idx;
                (frame_idx, scores[idx])
            })
            .filter(|(_, score)| score.is_finite())
            .collect();

        if spk_scores.is_empty() {
            continue;
        }

        // Sort by score descending
        spk_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_pos = spk_scores.len();
        let strong_k = ((n_pos as f32 * STRONG_BOOST_RATE) as usize).max(1);
        let weak_k = ((n_pos as f32 * WEAK_BOOST_RATE) as usize).max(1);

        // Boost strong frames (top strong_k)
        for i in 0..strong_k.min(spk_scores.len()) {
            let frame_idx = spk_scores[i].0;
            let idx = frame_idx * NUM_SPEAKERS + spk_idx;
            scores[idx] += 10.0; // Additive boost
        }

        // Boost weak frames (next weak_k)
        for i in strong_k..weak_k.min(spk_scores.len()) {
            let frame_idx = spk_scores[i].0;
            let idx = frame_idx * NUM_SPEAKERS + spk_idx;
            scores[idx] += 5.0; // Smaller boost
        }
    }
}

/// Select top-K frame indices per speaker based on quality scores.
///
/// Returns selected frame indices (not necessarily sorted).
pub(crate) fn get_topk_indices(
    scores: &[f32],
    num_frames: usize,
    k_per_speaker: usize,
) -> Vec<usize> {
    let mut selected = std::collections::HashSet::new();

    for spk_idx in 0..NUM_SPEAKERS {
        // Extract (frame_idx, score) for this speaker
        let mut spk_scores: Vec<(usize, f32)> = (0..num_frames)
            .map(|frame_idx| {
                let idx = frame_idx * NUM_SPEAKERS + spk_idx;
                (frame_idx, scores[idx])
            })
            .filter(|(_, score)| score.is_finite())
            .collect();

        // Sort by score descending
        spk_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top k_per_speaker frames
        for i in 0..k_per_speaker.min(spk_scores.len()) {
            selected.insert(spk_scores[i].0);
        }
    }

    selected.into_iter().collect()
}
