use crate::diar::parakeet::postprocess;
use crate::diar::parakeet::DiarizationConfig;

const NUM_SPEAKERS: usize = 4;

#[test]
fn test_median_filter_smooths_spike() {
    // Create predictions with a single spike in speaker 0
    // 7 frames, 4 speakers: speaker 0 has [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    let num_frames = 7;
    let mut preds = vec![0.0; num_frames * NUM_SPEAKERS];

    // Set speaker 0 predictions
    for frame in 0..num_frames {
        preds[frame * NUM_SPEAKERS + 0] = if frame == 3 { 0.9 } else { 0.1 };
    }

    let filtered = postprocess::median_filter(&preds, num_frames, 3);

    // With window=3, the spike at frame 3 should be smoothed out
    // Window for frame 3: [0.1, 0.9, 0.1] → median = 0.1
    let spk0_frame3 = filtered[3 * NUM_SPEAKERS + 0];
    assert!(
        (spk0_frame3 - 0.1).abs() < 0.01,
        "Spike should be smoothed to ~0.1, got {}",
        spk0_frame3
    );
}

#[test]
fn test_median_filter_preserves_sustained_signal() {
    // Speaker 0 high for a sustained period: [0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1]
    let num_frames = 7;
    let mut preds = vec![0.0; num_frames * NUM_SPEAKERS];

    for frame in 0..num_frames {
        let val = if frame >= 1 && frame <= 5 { 0.8 } else { 0.1 };
        preds[frame * NUM_SPEAKERS + 0] = val;
    }

    let filtered = postprocess::median_filter(&preds, num_frames, 3);

    // Middle frames should still be high
    let spk0_frame3 = filtered[3 * NUM_SPEAKERS + 0];
    assert!(
        spk0_frame3 > 0.7,
        "Sustained signal should be preserved, got {}",
        spk0_frame3
    );
}

#[test]
fn test_median_filter_no_mutation_corruption() {
    // Regression test for the in-place mutation bug.
    // Pattern: [high, low, low, low, low, low] for speaker 0
    // A buggy in-place filter would propagate the low value backwards.
    let num_frames = 6;
    let mut preds = vec![0.0; num_frames * NUM_SPEAKERS];

    preds[0 * NUM_SPEAKERS + 0] = 0.9;
    for frame in 1..num_frames {
        preds[frame * NUM_SPEAKERS + 0] = 0.1;
    }

    let filtered = postprocess::median_filter(&preds, num_frames, 3);

    // Frame 0 window (padded): [0.9, 0.1] → median = 0.9 (or 0.1 depending on sort)
    // The key thing: no values outside [0.1, 0.9] should appear (mutation would create
    // intermediate values or propagate incorrectly)
    for frame in 0..num_frames {
        let val = filtered[frame * NUM_SPEAKERS + 0];
        assert!(
            (val - 0.1).abs() < 0.01 || (val - 0.9).abs() < 0.01,
            "Frame {} has unexpected value {}, possible mutation corruption",
            frame,
            val
        );
    }
}

#[test]
fn test_binarize_produces_segment_with_hysteresis() {
    // Speaker 0: rises above onset, stays above offset, then drops below offset
    // onset=0.641, offset=0.561 (callhome defaults)
    // Probabilities: [0.0, 0.0, 0.7, 0.7, 0.6, 0.5, 0.0, 0.0] for speaker 0
    // Frame duration = 0.08s
    let num_frames = 8;
    let config = DiarizationConfig::callhome();
    let mut preds = vec![0.0; num_frames * NUM_SPEAKERS];

    let spk0_vals = [0.0, 0.0, 0.7, 0.7, 0.6, 0.5, 0.0, 0.0];
    for (frame, &val) in spk0_vals.iter().enumerate() {
        preds[frame * NUM_SPEAKERS + 0] = val;
    }

    let segments = postprocess::binarize(&preds, num_frames, &config);

    // Should produce 1 segment for speaker 0
    // Onset at frame 2 (0.7 >= 0.641)
    // Stays active at frame 3 (0.7 >= offset), frame 4 (0.6 >= 0.561)
    // Offset at frame 5 (0.5 < 0.561)
    // Duration: frame 2 to frame 5 = 3 frames * 0.08s = 0.24s (> 0.1s min)
    assert_eq!(segments.len(), 1, "Expected 1 segment, got {:?}", segments);

    let seg = &segments[0];
    assert_eq!(seg.speaker_id, 0);
    assert!((seg.start - 0.16).abs() < 0.01, "Start should be frame 2 * 0.08 = 0.16, got {}", seg.start);
    assert!((seg.end - 0.40).abs() < 0.01, "End should be frame 5 * 0.08 = 0.40, got {}", seg.end);
}

#[test]
fn test_binarize_min_duration_filter() {
    // Very short segment that should be filtered out (< 0.1s)
    // onset at frame 3, offset at frame 4 → 1 frame = 0.08s < 0.1s
    let num_frames = 8;
    let config = DiarizationConfig::callhome();
    let mut preds = vec![0.0; num_frames * NUM_SPEAKERS];

    preds[3 * NUM_SPEAKERS + 0] = 0.7; // Above onset
    // Frame 4: 0.0, below offset → segment ends
    // Duration: 0.08s < 0.1s → filtered out

    let segments = postprocess::binarize(&preds, num_frames, &config);
    assert_eq!(segments.len(), 0, "Short segment should be filtered out");
}

#[test]
fn test_binarize_multiple_speakers() {
    // Speaker 0 active frames 1-4, Speaker 2 active frames 3-6
    let num_frames = 8;
    let config = DiarizationConfig::callhome();
    let mut preds = vec![0.0; num_frames * NUM_SPEAKERS];

    for frame in 1..=4 {
        preds[frame * NUM_SPEAKERS + 0] = 0.8;
    }
    for frame in 3..=6 {
        preds[frame * NUM_SPEAKERS + 2] = 0.8;
    }

    let segments = postprocess::binarize(&preds, num_frames, &config);

    // Should have 2 segments (one per speaker)
    assert_eq!(segments.len(), 2, "Expected 2 segments, got {:?}", segments);

    let spk_ids: Vec<usize> = segments.iter().map(|s| s.speaker_id).collect();
    assert!(spk_ids.contains(&0));
    assert!(spk_ids.contains(&2));
}
