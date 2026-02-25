use crate::{Inference, llm::Smollm3};

#[test]
fn test_smollm3_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Smollm3>();
}

#[test]
#[ignore] // Requires ONNX model
fn test_smollm3_model_io_verification() {
    // Verify the ONNX model has the expected input/output structure
    let inference = Inference::cpu().unwrap();
    let session = inference
        .onnx_session("../../data/smollm3/model_int8.onnx")
        .unwrap();

    // Verify input count: 3 base inputs + 72 KV cache = 75 total
    let input_count = session.input_count().unwrap();
    assert_eq!(
        input_count, 75,
        "Expected 75 inputs (3 base + 72 KV cache), got {}",
        input_count
    );

    // Verify output count: 1 logits + 72 KV cache = 73 total
    let output_count = session.output_count().unwrap();
    assert_eq!(
        output_count, 73,
        "Expected 73 outputs (1 logits + 72 KV cache), got {}",
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
async fn test_smollm3_integration() {
    // Integration test for forward/recv generation
    let inference = Inference::cpu().unwrap();
    let mut model = inference.use_smollm3().unwrap().with_max_tokens(20);

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
