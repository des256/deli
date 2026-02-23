use crate::diar::parakeet::compression;

const NUM_SPEAKERS: usize = 4;

#[test]
fn test_get_log_pred_scores_basic() {
    // 3 frames, 4 speakers
    // Frame 0: speaker 0 dominates (0.8), Frame 1: speaker 1 dominates, Frame 2: low activity
    let preds = vec![
        0.8, 0.05, 0.05, 0.05, // frame 0
        0.05, 0.8, 0.05, 0.05, // frame 1
        0.1, 0.1, 0.1, 0.1,   // frame 2
    ];

    let scores = compression::get_log_pred_scores(&preds, 3);
    assert_eq!(scores.len(), 12); // 3 frames * 4 speakers

    // Speaker 0 in frame 0 should have highest score for speaker 0
    let spk0_frame0 = scores[0 * NUM_SPEAKERS + 0];
    let spk0_frame1 = scores[1 * NUM_SPEAKERS + 0];
    assert!(spk0_frame0 > spk0_frame1, "Speaker 0 should score higher in frame 0");

    // Speaker 1 in frame 1 should have highest score for speaker 1
    let spk1_frame0 = scores[0 * NUM_SPEAKERS + 1];
    let spk1_frame1 = scores[1 * NUM_SPEAKERS + 1];
    assert!(spk1_frame1 > spk1_frame0, "Speaker 1 should score higher in frame 1");
}

#[test]
fn test_disable_low_scores() {
    let mut scores = vec![0.5, -0.1, 0.3, 0.1]; // 1 frame, 4 speakers
    compression::disable_low_scores(&mut scores, 1);

    // PRED_SCORE_THRESHOLD = 0.25
    // 0.5 > 0.25 → kept
    // -0.1 <= 0.25 → NEG_INFINITY
    // 0.3 > 0.25 → kept
    // 0.1 <= 0.25 → NEG_INFINITY
    assert!(scores[0].is_finite());
    assert!(scores[1] == f32::NEG_INFINITY);
    assert!(scores[2].is_finite());
    assert!(scores[3] == f32::NEG_INFINITY);
}

#[test]
fn test_boost_topk_scores_boosts_highest() {
    // 4 frames, 4 speakers — only speaker 0 has finite scores
    let mut scores = vec![
        3.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, // frame 0
        2.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, // frame 1
        1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, // frame 2
        0.5, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, // frame 3
    ];

    let original_top = scores[0]; // frame 0 speaker 0 = 3.0
    compression::boost_topk_scores(&mut scores, 4);

    // Top frame should be boosted (strong boost = +10)
    assert!(scores[0] > original_top, "Top score should be boosted");
}

#[test]
fn test_get_topk_indices_selects_correct_frames() {
    // 6 frames, 4 speakers. Speaker 0 dominates frames 0,1; Speaker 1 dominates frames 2,3
    let mut scores = vec![f32::NEG_INFINITY; 6 * NUM_SPEAKERS];

    // Speaker 0 high in frames 0 and 1
    scores[0 * NUM_SPEAKERS + 0] = 10.0;
    scores[1 * NUM_SPEAKERS + 0] = 9.0;

    // Speaker 1 high in frames 2 and 3
    scores[2 * NUM_SPEAKERS + 1] = 10.0;
    scores[3 * NUM_SPEAKERS + 1] = 9.0;

    let indices = compression::get_topk_indices(&scores, 6, 2);

    // Should select frames 0, 1 (for speaker 0) and 2, 3 (for speaker 1)
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(indices.contains(&2));
    assert!(indices.contains(&3));
}
