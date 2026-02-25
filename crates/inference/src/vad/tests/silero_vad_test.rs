use crate::{Inference, vad::SileroVad};

#[test]
fn test_silero_vad_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<SileroVad>();
}

#[test]
#[ignore] // Requires ONNX model
fn test_silero_vad_model_io_names() {
    // Risk mitigation: verify Silero VAD v5 I/O names match expectations
    let inference = Inference::cpu().unwrap();
    let session = inference
        .onnx_session("../../data/silero/silero_vad.onnx")
        .unwrap();

    // Verify input names
    let input_count = session.input_count().unwrap();
    assert_eq!(input_count, 3, "Expected 3 inputs (input, state, sr)");
    let mut input_names: Vec<String> = (0..input_count)
        .map(|i| session.input_name(i).unwrap())
        .collect();
    input_names.sort();
    assert_eq!(input_names, vec!["input", "sr", "state"]);

    // Verify output names
    let output_count = session.output_count().unwrap();
    assert_eq!(output_count, 2, "Expected 2 outputs (output, stateN)");
    let mut output_names: Vec<String> = (0..output_count)
        .map(|i| session.output_name(i).unwrap())
        .collect();
    output_names.sort();
    assert_eq!(output_names, vec!["output", "stateN"]);
}

#[test]
#[ignore] // Requires ONNX model
fn test_silero_vad_integration() {
    let inference = Inference::cpu().unwrap();
    let mut vad = inference.use_silero_vad().unwrap();

    // Test 1: Silence should produce low probability
    let silence: Vec<f32> = vec![0.0; 512];
    let prob_silence = vad.process(&silence).unwrap();
    assert!(
        prob_silence < 0.3,
        "Silence should produce low probability, got {}",
        prob_silence
    );
    assert!(
        prob_silence >= 0.0 && prob_silence <= 1.0 && !prob_silence.is_nan(),
        "Probability must be in [0,1] and not NaN"
    );

    // Test 2: Generate sine wave (440Hz at 16kHz = ~36 samples per cycle)
    let mut sine_wave = Vec::with_capacity(512);
    for i in 0..512 {
        let t = i as f32 / 16000.0;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        sine_wave.push(sample);
    }

    // Process multiple frames of sine wave
    for _ in 0..5 {
        let prob = vad.process(&sine_wave).unwrap();
        assert!(
            prob >= 0.0 && prob <= 1.0 && !prob.is_nan() && !prob.is_infinite(),
            "Probability must be valid float in [0,1], got {}",
            prob
        );
    }

    // Test 3: Reset and verify state returns to initial
    vad.reset().unwrap();
    let prob_after_reset = vad.process(&silence).unwrap();
    assert!(
        (prob_after_reset - prob_silence).abs() < 0.1,
        "After reset, silence probability should be similar to initial: {} vs {}",
        prob_after_reset,
        prob_silence
    );
}

#[test]
#[ignore] // Requires ONNX model
fn test_silero_vad_multiple_frames() {
    let inference = Inference::cpu().unwrap();
    let mut vad = inference.use_silero_vad().unwrap();

    let silence: Vec<f32> = vec![0.0; 512];

    // Process 100 frames
    for i in 0..100 {
        let prob = vad.process(&silence).unwrap();
        assert!(
            prob >= 0.0 && prob <= 1.0 && !prob.is_nan() && !prob.is_infinite(),
            "Frame {}: probability must be valid, got {}",
            i,
            prob
        );
    }
}
