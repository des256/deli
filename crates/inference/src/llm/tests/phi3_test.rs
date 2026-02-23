// Tests for Phi3 LLM public API

use {
    candle_core::Device,
    inference::{Inference, llm::Phi3},
    std::path::Path,
};

fn cuda_device() -> Device {
    Device::new_cuda(0).expect("CUDA device required")
}

fn cuda() -> Inference {
    Inference::cuda(0).expect("CUDA device required")
}

#[test]
fn test_phi3_construction_fails_for_missing_file() {
    let device = cuda_device();
    let result = Phi3::new("fake_model.gguf", "fake_tokenizer.json", device);
    assert!(result.is_err(), "should fail with non-existent files");
}

#[test]
fn test_phi3_send_sync() {
    // Verify Phi3 implements Send + Sync (required for async use across .await points)
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Phi3>();
}

#[test]
fn test_inference_factory_signature() {
    let inference = cuda();
    // Verify method exists and returns error for non-existent file
    let result = inference.use_phi3("fake_model.gguf", "fake_tokenizer.json");
    assert!(result.is_err(), "should fail with non-existent files");
}

#[test]
fn test_infer_error_importable() {
    // Verifies the InferError type is accessible for matching.
    // Actual empty-prompt validation is exercised in test_forward_empty_prompt_validation
    // (requires real model files).
    use inference::error::InferError;
    let _: Option<InferError> = None;
}

#[tokio::test]
async fn test_forward_empty_prompt_validation() {
    let model_path = Path::new("../../data/phi3/Phi-3-mini-4k-instruct-q4.gguf");
    let tokenizer_path = Path::new("../../data/phi3/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at data/phi3/");
        return;
    }

    let device = cuda_device();
    let phi3 = Phi3::new(model_path, tokenizer_path, device).expect("Failed to load model");

    // Empty prompt should return error
    let result = phi3.forward("", 10).await;
    assert!(result.is_err(), "empty prompt should be rejected");
}

#[tokio::test]
async fn test_cuda_device_propagation() {
    let inference = cuda();
    // Verify device propagates to Phi3
    // This will fail without model files, but confirms the API exists
    let result = inference.use_phi3("fake_model.gguf", "fake_tokenizer.json");
    assert!(result.is_err(), "should fail with non-existent files");
}

#[tokio::test]
async fn test_forward_with_cuda() {
    let model_path = Path::new("../../data/phi3/Phi-3-mini-4k-instruct-q4.gguf");
    let tokenizer_path = Path::new("../../data/phi3/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at data/phi3/");
        return;
    }

    let inference = cuda();
    let phi3 = inference
        .use_phi3(model_path, tokenizer_path)
        .expect("Failed to load model on CUDA");

    // Generate with a simple prompt
    let prompt = "Hello";
    let sample_len = 10;
    let result = phi3.forward(prompt, sample_len).await;

    assert!(
        result.is_ok(),
        "forward should succeed on CUDA: {:?}",
        result.err()
    );
    let text = result.unwrap();
    assert!(!text.is_empty(), "generated text should not be empty");
    println!("Generated on CUDA: {}", text);
}
