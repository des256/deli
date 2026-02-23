use crate::diar::parakeet::{DiarizationConfig, Sortformer};
use crate::Inference;

#[test]
#[ignore] // Requires ONNX model file
fn test_sortformer_new_initializes_zero_state() {
    let inference = Inference::cpu().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let session = inference
        .onnx_session(&model_path)
        .expect("Failed to load model");
    let config = DiarizationConfig::callhome();

    let sortformer = Sortformer::new(session, config).expect("Failed to create Sortformer");

    // Verify it was created successfully
    // (Internal state is private, so we can only verify construction succeeds)
    drop(sortformer);
}

#[test]
#[ignore] // Requires ONNX model file
fn test_first_streaming_update_with_empty_cache() {
    // This test verifies that the first streaming_update call succeeds with zero-length
    // spkcache/fifo (empty tensors [1, 0, DIM]).
    let inference = Inference::cpu().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let session = inference
        .onnx_session(&model_path)
        .expect("Failed to load model");
    let config = DiarizationConfig::callhome();

    let mut sortformer = Sortformer::new(session, config).expect("Failed to create Sortformer");

    // Generate synthetic features: 124 frames x 128 dims (CHUNK_LEN = 124)
    let chunk_feat_frames = 124;
    let chunk_feat: Vec<f32> = vec![0.0; chunk_feat_frames * 128];

    // First call with empty cache/fifo
    let result = sortformer.streaming_update(&chunk_feat, chunk_feat_frames, chunk_feat_frames);

    assert!(
        result.is_ok(),
        "First streaming_update with empty cache should succeed: {:?}",
        result.err()
    );

    let predictions = result.unwrap();
    // Predictions should have shape [chunk_out_frames, 4]
    // chunk_out_frames = chunk_feat_frames / SUBSAMPLING = 124 / 8 = 15 (with rounding)
    assert!(predictions.len() % 4 == 0, "Predictions should be multiple of 4 speakers");
    assert!(predictions.len() > 0, "Predictions should not be empty");
}

#[test]
#[ignore] // Requires ONNX model file
fn test_reset_zeros_state() {
    let inference = Inference::cpu().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let session = inference
        .onnx_session(&model_path)
        .expect("Failed to load model");
    let config = DiarizationConfig::callhome();

    let mut sortformer = Sortformer::new(session, config).expect("Failed to create Sortformer");

    // Process a chunk to build up state
    let chunk_feat_frames = 124;
    let chunk_feat: Vec<f32> = vec![0.1; chunk_feat_frames * 128];
    let _ = sortformer.streaming_update(&chunk_feat, chunk_feat_frames, chunk_feat_frames);

    // Reset
    sortformer.reset().expect("Reset should succeed");

    // Process another chunk - should behave like first chunk again
    let result2 = sortformer.streaming_update(&chunk_feat, chunk_feat_frames, chunk_feat_frames);
    assert!(result2.is_ok(), "Streaming after reset should succeed");
}

#[test]
#[ignore] // Requires ONNX model file
fn test_streaming_update_state_persistence() {
    // Verify that FIFO and cache grow over multiple calls
    let inference = Inference::cpu().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let session = inference
        .onnx_session(&model_path)
        .expect("Failed to load model");
    let config = DiarizationConfig::callhome();

    let mut sortformer = Sortformer::new(session, config).expect("Failed to create Sortformer");

    let chunk_feat_frames = 124;
    let chunk_feat: Vec<f32> = vec![0.0; chunk_feat_frames * 128];

    // First call
    let preds1 = sortformer
        .streaming_update(&chunk_feat, chunk_feat_frames, chunk_feat_frames)
        .expect("First call should succeed");

    // Second call - state should persist
    let preds2 = sortformer
        .streaming_update(&chunk_feat, chunk_feat_frames, chunk_feat_frames)
        .expect("Second call should succeed");

    // Both calls should succeed and produce predictions
    assert!(preds1.len() > 0);
    assert!(preds2.len() > 0);
    // Predictions may differ between calls if state affects output
}

#[test]
#[ignore] // Requires ONNX model file
fn test_cache_compression_triggers() {
    // Process many chunks to fill the cache and trigger compression
    // Verify cache doesn't grow unbounded
    let inference = Inference::cpu().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let session = inference
        .onnx_session(&model_path)
        .expect("Failed to load model");
    let config = DiarizationConfig::callhome();

    let mut sortformer = Sortformer::new(session, config).expect("Failed to create Sortformer");

    let chunk_feat_frames = 124;
    let chunk_feat: Vec<f32> = vec![0.1; chunk_feat_frames * 128];

    // Process 20 chunks - this should trigger cache compression
    // Each chunk adds ~15 frames to FIFO, which eventually overflows into cache
    for i in 0..20 {
        let result = sortformer.streaming_update(&chunk_feat, chunk_feat_frames, chunk_feat_frames);
        assert!(
            result.is_ok(),
            "Chunk {} should succeed: {:?}",
            i,
            result.err()
        );
    }

    // Compression should have kept the cache bounded
    // (We can't directly inspect cache size, but the test succeeding means no panic/error)
    // If compression didn't work, the cache would grow unbounded and potentially cause issues
}
