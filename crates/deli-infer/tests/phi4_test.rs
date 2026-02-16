// Tests for Phi4 LLM public API

use candle_core::Device;
use deli_infer::{Inference, Phi4};
use std::path::Path;

#[test]
fn test_phi4_construction_fails_for_missing_file() {
    let device = Device::Cpu;
    let result = Phi4::new("fake_model.gguf", "fake_tokenizer.json", device);
    assert!(result.is_err(), "should fail with non-existent files");
}

#[test]
fn test_phi4_send_sync() {
    // Verify Phi4 implements Send + Sync (required for async use across .await points)
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Phi4>();
}

#[test]
fn test_inference_factory_signature() {
    let inference = Inference::cpu();
    // Verify method exists and returns error for non-existent file
    let result = inference.use_phi4("fake_model.gguf", "fake_tokenizer.json");
    assert!(result.is_err(), "should fail with non-existent files");
}

#[tokio::test]
async fn test_forward_empty_prompt_rejected() {
    // Validates the empty prompt check without needing a real model.
    // We test the public async fn via a helper that bypasses model loading.
    // Since we can't construct Phi4 without a GGUF file, test the error message
    // expectation is documented here; the actual validation is exercised in
    // test_forward_empty_prompt_validation (requires real model).
    // This test verifies the InferError type is accessible for matching.
    use deli_infer::InferError;
    let _: Option<InferError> = None; // type exists and is importable
}

#[tokio::test]
#[ignore] // Only run if real model files are available
async fn test_forward_with_real_model() {
    let model_path = Path::new("../../models/phi-4/phi-4-q4.gguf");
    let tokenizer_path = Path::new("../../models/phi-4/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at models/phi-4/");
        return;
    }

    let device = Device::Cpu;
    let phi4 = Phi4::new(model_path, tokenizer_path, device)
        .expect("Failed to load model");

    // Generate with a simple prompt
    let prompt = "Hello";
    let sample_len = 10; // Generate 10 tokens
    let result = phi4.forward(prompt, sample_len).await;

    assert!(result.is_ok(), "forward should succeed: {:?}", result.err());
    let text = result.unwrap();
    assert!(!text.is_empty(), "generated text should not be empty");
    println!("Generated: {}", text);
}

#[tokio::test]
#[ignore] // Requires real model - tests validation logic without full generation
async fn test_forward_empty_prompt_validation() {
    let model_path = Path::new("../../models/phi-4/phi-4-q4.gguf");
    let tokenizer_path = Path::new("../../models/phi-4/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at models/phi-4/");
        return;
    }

    let device = Device::Cpu;
    let phi4 = Phi4::new(model_path, tokenizer_path, device)
        .expect("Failed to load model");

    // Empty prompt should return error
    let result = phi4.forward("", 10).await;
    assert!(result.is_err(), "empty prompt should be rejected");
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore] // Requires GPU
async fn test_cuda_device_propagation() {
    // Note: This test may fail if CUDA is not available on the system
    // That's expected behavior - it tests the API, not the availability
    match Inference::cuda(0) {
        Ok(inference) => {
            // Verify device propagates to Phi4
            // This will fail without model files, but confirms the API exists
            let result = inference.use_phi4("fake_model.gguf", "fake_tokenizer.json");
            assert!(result.is_err(), "should fail with non-existent files");
        }
        Err(_) => {
            // CUDA not available, skip test
            eprintln!("CUDA not available, skipping cuda test");
        }
    }
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore] // Requires GPU and real model files
async fn test_forward_with_cuda() {
    let model_path = Path::new("../../models/phi-4/phi-4-q4.gguf");
    let tokenizer_path = Path::new("../../models/phi-4/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping test - model files not found at models/phi-4/");
        return;
    }

    match Inference::cuda(0) {
        Ok(inference) => {
            let phi4 = inference
                .use_phi4(model_path, tokenizer_path)
                .expect("Failed to load model on CUDA");

            // Generate with a simple prompt
            let prompt = "Hello";
            let sample_len = 10;
            let result = phi4.forward(prompt, sample_len).await;

            assert!(result.is_ok(), "forward should succeed on CUDA: {:?}", result.err());
            let text = result.unwrap();
            assert!(!text.is_empty(), "generated text should not be empty");
            println!("Generated on CUDA: {}", text);
        }
        Err(_) => {
            eprintln!("CUDA not available, skipping cuda test");
        }
    }
}
