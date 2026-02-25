use crate::{Inference, llm::Gemma3};

#[test]
fn test_gemma3_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Gemma3>();
}

#[test]
#[ignore] // Requires ONNX model
fn test_gemma3_model_io_verification() {
    // Verify the ONNX model has the expected input/output structure
    let inference = Inference::cpu().unwrap();
    let session = inference
        .onnx_session("../../data/gemma3/model_int8.onnx")
        .unwrap();

    // Gemma 3 1B int8: 2 base inputs + 52 KV cache (26 layers * 2) = 54 total
    // Base inputs: input_ids, position_ids (no attention_mask)
    let input_count = session.input_count().unwrap();
    assert_eq!(
        input_count, 54,
        "Expected 54 inputs (2 base + 52 KV cache for 26 layers), got {}",
        input_count
    );

    // Verify output count: 1 logits + 52 KV cache = 53 total
    let output_count = session.output_count().unwrap();
    assert_eq!(
        output_count, 53,
        "Expected 53 outputs (1 logits + 52 KV cache), got {}",
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
        input_names.contains(&"position_ids".to_string()),
        "position_ids not found in inputs: {:?}",
        input_names
    );

    // Gemma 3 1B int8 does NOT have attention_mask
    assert!(
        !input_names.contains(&"attention_mask".to_string()),
        "attention_mask should not be present in Gemma 3 1B int8 inputs"
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
async fn test_gemma3_integration() {
    // Integration test for forward/recv generation
    let inference = Inference::cpu().unwrap();
    let mut model = inference.use_gemma3().unwrap().with_max_tokens(20);

    // Start generation
    model.forward("Hello").unwrap();

    // Collect generated tokens
    let mut tokens = Vec::new();
    loop {
        match model.recv().await {
            Some(Ok(token)) => {
                eprint!("{}", token); // Print for visibility
                tokens.push(token);
            }
            Some(Err(e)) => {
                panic!("Generation error: {}", e);
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

    // Verify recv() returns None after generation completes
    assert!(
        model.recv().await.is_none(),
        "recv() should return None after generation completes"
    );
}
