use crate::{Inference, llm::{Llama32Handle, LlmInput, LlmOutput}, base::Epoch};

#[test]
fn test_llama32_handle_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Llama32Handle<u64>>();
}

#[test]
#[ignore] // Requires ONNX model
fn test_llama32_model_io_verification() {
    // Verify the ONNX model has the expected input/output structure
    let inference = Inference::new().unwrap();
    let onnx = std::sync::Arc::new(onnx::Onnx::new(24).unwrap());
    let session = onnx
        .create_session(
            &onnx::Executor::Cpu,
            &onnx::OptimizationLevel::EnableAll,
            4,
            "data/llama32/model_int8.onnx",
        )
        .unwrap();

    // Verify input count: 3 base inputs + 32 KV cache (16 layers * 2) = 35 total
    let input_count = session.input_count().unwrap();
    assert_eq!(
        input_count, 35,
        "Expected 35 inputs (3 base + 32 KV cache for 16 layers), got {}",
        input_count
    );

    // Verify output count: 1 logits + 32 KV cache = 33 total
    let output_count = session.output_count().unwrap();
    assert_eq!(
        output_count, 33,
        "Expected 33 outputs (1 logits + 32 KV cache), got {}",
        output_count
    );

    // Collect input names and verify key inputs are present
    let mut input_names: Vec<String> = (0..input_count)
        .map(|i| session.input_name(i).unwrap())
        .collect();
    input_names.sort();

    assert!(
        input_names.contains(&"input_ids".to_string()),
        "input_ids not found in inputs: {:?}",
        input_names
    );
    assert!(
        input_names.contains(&"attention_mask".to_string()),
        "attention_mask not found in inputs: {:?}",
        input_names
    );
    assert!(
        input_names.contains(&"position_ids".to_string()),
        "position_ids not found in inputs: {:?}",
        input_names
    );

    // Collect output names and verify logits is present
    let output_names: Vec<String> = (0..output_count)
        .map(|i| session.output_name(i).unwrap())
        .collect();

    assert!(
        output_names.contains(&"logits".to_string()),
        "logits not found in outputs: {:?}",
        output_names
    );
}

#[tokio::test]
#[ignore] // Requires ONNX model
async fn test_llama32_integration() {
    // Integration test for handle/listener generation
    let inference = Inference::new().unwrap();
    let epoch = Epoch::new();
    let (handle, mut listener) = inference.use_llama32::<u64>(&onnx::Executor::Cpu, epoch).unwrap();

    // Send prompt
    handle.send(LlmInput {
        payload: 0,
        prompt: "Hello".to_string(),
    }).unwrap();

    // Collect generated tokens
    let mut tokens = Vec::new();
    let mut token_count = 0;
    loop {
        match listener.recv().await {
            Some(stamped) => {
                match stamped.inner {
                    LlmOutput::Token { token, .. } => {
                        eprint!("{}", token); // Print for visibility
                        tokens.push(token);
                        token_count += 1;
                        if token_count >= 20 {
                            break;
                        }
                    }
                    LlmOutput::Eos { .. } => {
                        break;
                    }
                }
            }
            None => break,
        }
    }
    eprintln!(); // Newline after generation

    // Verify at least 3 tokens were generated
    assert!(
        tokens.len() >= 3,
        "Expected at least 3 tokens, got {}",
        tokens.len()
    );

    // Verify concatenated output is non-empty and >= 5 chars
    let output = tokens.join("");
    assert!(!output.is_empty(), "Generated output is empty");
    assert!(
        output.len() >= 5,
        "Generated output is too short: {} chars, expected >= 5",
        output.len()
    );
}
