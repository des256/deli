use deli_infer::{InferError, Inference};

#[test]
fn test_inference_cpu() {
    let inference = Inference::cpu();
    assert!(inference.device().is_cpu());
}

#[cfg(feature = "cuda")]
#[test]
fn test_inference_cuda() {
    // Note: This test may fail if CUDA is not available on the system
    // That's expected behavior - it tests the API, not the availability
    match Inference::cuda(0) {
        Ok(inference) => {
            assert!(inference.device().is_cuda());
        }
        Err(_) => {
            // CUDA not available, skip test
            eprintln!("CUDA not available, skipping cuda test");
        }
    }
}

#[test]
fn test_use_pose_detector_signature() {
    let inference = Inference::cpu();
    // Verify method exists and returns error for non-existent file
    let result = inference.use_pose_detector("fake_path.safetensors");
    assert!(result.is_err()); // Should error because file doesn't exist
}

#[test]
fn test_use_qwen3_signature() {
    let inference = Inference::cpu();
    // Verify method exists and returns error for non-existent files
    let result = inference.use_qwen3("fake_model.gguf", "fake_tokenizer.json");
    assert!(result.is_err()); // Should error because files don't exist
}

#[test]
fn test_onnx_session_nonexistent_file() {
    let inference = Inference::cpu();
    let result = inference.onnx_session("nonexistent.onnx");
    assert!(result.is_err());
    match result {
        Err(InferError::Io(msg)) | Err(InferError::Onnx(msg)) => {
            // Should contain "not found" or similar
            assert!(
                msg.to_lowercase().contains("not found") ||
                msg.to_lowercase().contains("no such file") ||
                msg.to_lowercase().contains("does not exist"),
                "Expected error message about file not found, got: {}", msg
            );
        }
        _ => panic!("Expected InferError::Onnx or InferError::Io variant"),
    }
}

#[test]
fn test_infererror_onnx_display() {
    let err = InferError::Onnx("test onnx error".to_string());
    let display_str = format!("{}", err);
    assert!(display_str.contains("onnx"));
    assert!(display_str.contains("test onnx error"));
}

#[test]
fn test_onnx_session_multiple_calls_no_panic() {
    let inference = Inference::cpu();
    // Calling onnx_session multiple times should not panic (OnceLock ensures single init)
    let _result1 = inference.onnx_session("fake1.onnx");
    let _result2 = inference.onnx_session("fake2.onnx");
    // If we got here without panicking, the test passes
}
