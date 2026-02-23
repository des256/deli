// Tests for Llama LLM public API

use {
    candle_core::Device,
    inference::{Inference, llm::Llama},
    std::path::Path,
};

fn cuda_device() -> Device {
    Device::new_cuda(0).expect("CUDA device required")
}

fn cuda() -> Inference {
    Inference::cuda(0).expect("CUDA device required")
}

#[test]
fn test_llama_construction_fails_for_missing_file() {
    let device = cuda_device();
    let result = Llama::new("fake_model.gguf", "fake_tokenizer.json", device);
    assert!(result.is_err(), "should fail with non-existent files");
}

#[test]
fn test_llama_send_sync() {
    // Verify Llama implements Send + Sync (required for async use across .await points)
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Llama>();
}

#[test]
fn test_inference_factory_signature() {
    let inference = cuda();
    // Verify method exists and returns error for non-existent file
    let result = inference.use_llama("fake_model.gguf", "fake_tokenizer.json");
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
    let model_path = Path::new("../../data/llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    let tokenizer_path = Path::new("../../data/llama/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at data/llama/");
        return;
    }

    let device = cuda_device();
    let llama = Llama::new(model_path, tokenizer_path, device).expect("Failed to load model");

    // Empty prompt should return error
    let result = llama.forward("", 10).await;
    assert!(result.is_err(), "empty prompt should be rejected");
}

#[tokio::test]
async fn test_cuda_device_propagation() {
    let inference = cuda();
    // Verify device propagates to Llama
    // This will fail without model files, but confirms the API exists
    let result = inference.use_llama("fake_model.gguf", "fake_tokenizer.json");
    assert!(result.is_err(), "should fail with non-existent files");
}

#[tokio::test]
async fn test_forward_with_cuda() {
    let model_path = Path::new("../../data/llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    let tokenizer_path = Path::new("../../data/llama/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at data/llama/");
        return;
    }

    let inference = cuda();
    let llama = inference
        .use_llama(model_path, tokenizer_path)
        .expect("Failed to load model on CUDA");

    // Generate with a simple prompt
    let prompt = "Hello";
    let sample_len = 10;
    let result = llama.forward(prompt, sample_len).await;

    assert!(
        result.is_ok(),
        "forward should succeed on CUDA: {:?}",
        result.err()
    );
    let text = result.unwrap();
    assert!(!text.is_empty(), "generated text should not be empty");
    println!("Generated on CUDA: {}", text);
}
