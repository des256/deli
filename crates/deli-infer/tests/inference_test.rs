use deli_infer::Inference;

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
