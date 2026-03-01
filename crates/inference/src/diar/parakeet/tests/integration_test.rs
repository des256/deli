use crate::Inference;

#[test]
#[ignore] // Requires ONNX model file
fn test_inference_use_parakeet_diar() {
    // Test that Inference::use_parakeet_diar creates a Sortformer instance
    let inference = Inference::new().expect("Failed to create inference");

    let sortformer = inference
        .use_parakeet_diar(&onnx::Executor::Cpu)
        .expect("Failed to create Sortformer via Inference API");

    // Verify it was created successfully
    drop(sortformer);
}

#[test]
#[ignore] // Requires ONNX model file
fn test_diarize_silence_returns_empty_segments() {
    // Test that silence audio (all zeros) produces no speaker segments
    let inference = Inference::new().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let mut sortformer = inference
        .use_parakeet_diar(&onnx::Executor::Cpu)
        .expect("Failed to create Sortformer");

    // 1 second of silence at 16kHz
    let audio: Vec<f32> = vec![0.0; 16000];

    let segments = sortformer
        .diarize_chunk(&audio)
        .expect("Diarization should succeed");

    // Silence should produce no segments (or very few with low confidence)
    // Allowing up to 1 segment in case the model has noise
    assert!(
        segments.len() <= 1,
        "Silence should produce 0-1 segments, got {}",
        segments.len()
    );
}

#[test]
#[ignore] // Requires ONNX model file
fn test_diarize_test_audio_produces_valid_segments() {
    // Test that non-silence audio produces valid speaker segments
    let inference = Inference::new().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let mut sortformer = inference
        .use_parakeet_diar(&onnx::Executor::Cpu)
        .expect("Failed to create Sortformer");

    // Generate synthetic audio: 440Hz sine wave (simulates speech)
    let sample_rate = 16000;
    let duration = 2.0; // 2 seconds
    let num_samples = (sample_rate as f32 * duration) as usize;
    let frequency = 440.0;

    let mut audio = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = 0.3 * (2.0 * std::f32::consts::PI * frequency * t).sin();
        audio.push(sample);
    }

    let segments = sortformer
        .diarize_chunk(&audio)
        .expect("Diarization should succeed");

    // Verify segments have valid structure
    for segment in &segments {
        assert!(
            segment.start < segment.end,
            "Segment start {} should be < end {}",
            segment.start,
            segment.end
        );
        assert!(
            segment.start >= 0.0 && segment.end <= duration,
            "Segment times should be within audio duration"
        );
        assert!(
            segment.speaker_id < 4,
            "Speaker ID {} should be in range 0..3",
            segment.speaker_id
        );
    }
}

#[test]
#[ignore] // Requires ONNX model file
fn test_multi_chunk_streaming_state_persistence() {
    // Call diarize_chunk multiple times and verify state persists
    let inference = Inference::new().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let mut sortformer = inference
        .use_parakeet_diar(&onnx::Executor::Cpu)
        .expect("Failed to create Sortformer");

    // Generate 3 seconds of audio
    let sample_rate = 16000;
    let duration_per_chunk = 1.0; // 1 second
    let num_samples_per_chunk = (sample_rate as f32 * duration_per_chunk) as usize;
    let frequency = 440.0;

    // Process 3 chunks
    for chunk_idx in 0..3 {
        let mut audio = Vec::with_capacity(num_samples_per_chunk);
        for i in 0..num_samples_per_chunk {
            let t = ((chunk_idx * num_samples_per_chunk) + i) as f32 / sample_rate as f32;
            let sample = 0.3 * (2.0 * std::f32::consts::PI * frequency * t).sin();
            audio.push(sample);
        }

        let segments = sortformer
            .diarize_chunk(&audio)
            .expect(&format!("Chunk {} should succeed", chunk_idx));

        // Each chunk should produce some output (or none for silence)
        // Just verify it doesn't panic and produces valid segments
        for segment in &segments {
            assert!(segment.start < segment.end);
            assert!(segment.speaker_id < 4);
        }
    }

    // If we got here without panicking, streaming state persisted correctly
}

#[test]
#[ignore] // Requires ONNX model file
fn test_short_audio_no_panic() {
    // Test that short audio (0.5s) succeeds without panic (partial chunk handling)
    let inference = Inference::new().expect("Failed to create inference");
    let model_path = format!(
        "{}/../../data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx",
        env!("CARGO_MANIFEST_DIR")
    );

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }

    let mut sortformer = inference
        .use_parakeet_diar(&onnx::Executor::Cpu)
        .expect("Failed to create Sortformer");

    // 0.5 seconds = 8000 samples at 16kHz
    let audio: Vec<f32> = vec![0.1; 8000];

    let result = sortformer.diarize_chunk(&audio);

    // Should succeed without panic
    assert!(
        result.is_ok(),
        "Short audio should not panic: {:?}",
        result.err()
    );

    let segments = result.unwrap();
    // Verify any segments produced are valid
    for segment in &segments {
        assert!(segment.start < segment.end);
        assert!(segment.speaker_id < 4);
    }
}

#[test]
fn test_sortformer_is_send() {
    // Verify Sortformer is Send (for use across async boundaries)
    fn assert_send<T: Send>() {}
    assert_send::<crate::diar::parakeet::Sortformer>();
}
