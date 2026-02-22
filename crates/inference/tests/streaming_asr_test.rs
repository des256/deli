// Re-export the functions from the streaming module for testing
// In production, users will access via inference::asr::streaming::*
mod streaming_imports {
    pub use inference::asr::streaming::{compute_features, load_tokens, StreamingAsr};
}
use streaming_imports::*;
use inference::InferError;

#[test]
fn test_load_tokens_valid_file() {
    let tokens_path = "tests/fixtures/test_tokens.txt";

    // Create a test tokens file
    std::fs::create_dir_all("tests/fixtures").unwrap();
    std::fs::write(
        tokens_path,
        "<blk> 0\n<sos/eos> 1\n<unk> 2\nS 3\n▁THE 4\n",
    )
    .unwrap();

    let result = load_tokens(tokens_path);
    assert!(result.is_ok());

    let tokens = result.unwrap();
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[0], "<blk>");
    assert_eq!(tokens[1], "<sos/eos>");
    assert_eq!(tokens[2], "<unk>");
    assert_eq!(tokens[3], "S");
    assert_eq!(tokens[4], "▁THE");

    // Clean up
    std::fs::remove_file(tokens_path).ok();
}

#[test]
fn test_load_tokens_missing_file() {
    let result = load_tokens("nonexistent_tokens.txt");
    assert!(result.is_err());
}

#[test]
fn test_load_tokens_malformed_file() {
    let tokens_path = "tests/fixtures/malformed_tokens.txt";

    std::fs::create_dir_all("tests/fixtures").unwrap();
    std::fs::write(tokens_path, "invalid line without ID\n").unwrap();

    let result = load_tokens(tokens_path);
    assert!(result.is_err());

    std::fs::remove_file(tokens_path).ok();
}

#[test]
fn test_compute_features_dimensions() {
    // 16kHz PCM audio - 1 second of silence (16000 samples)
    let pcm_data: Vec<i16> = vec![0; 16000];
    let sample_rate = 16000;

    let result = compute_features(&pcm_data, sample_rate);
    assert!(result.is_ok());

    let features = result.unwrap();

    // num_frames = (num_samples - window_size) / hop_size + 1
    // = (16000 - 400) / 160 + 1 = 98.5 rounded down = 98
    // Total length = num_frames * 80 bins
    let expected_num_frames = (pcm_data.len() - 400) / 160 + 1;
    assert_eq!(features.len(), expected_num_frames * 80);
}

#[test]
fn test_compute_features_invalid_sample_rate() {
    let pcm_data: Vec<i16> = vec![0; 8000];
    let sample_rate = 8000;

    let result = compute_features(&pcm_data, sample_rate);
    assert!(result.is_err());

    match result {
        Err(InferError::Runtime(msg)) => {
            assert!(msg.contains("16000"));
            assert!(msg.contains("8000"));
        }
        _ => panic!("Expected Runtime error with sample rate message"),
    }
}

#[test]
fn test_streaming_asr_is_send() {
    // This test verifies that StreamingAsr can be sent across threads
    fn assert_send<T: Send>() {}
    assert_send::<StreamingAsr>();
}

// Note: Full decode_chunk tests require actual ONNX model files
// These will be added when the user provides the model files
