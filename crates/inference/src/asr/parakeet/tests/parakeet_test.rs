use crate::asr::parakeet::features::compute_features;
use crate::asr::parakeet::tokens::load_tokens;

#[test]
fn test_load_tokens_from_sentencepiece_model() {
    let model_path = format!(
        "{}/../../data/parakeet/tokenizer.model",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: tokenizer.model not found at {}", model_path);
        return;
    }

    let tokens = load_tokens(&model_path).expect("Failed to load tokenizer.model");
    assert_eq!(tokens.len(), 1024, "Should have 1024 tokens");
    assert_eq!(tokens[0], "<unk>", "First token should be <unk>");
}

#[test]
fn test_load_tokens_missing_file() {
    let result = load_tokens("nonexistent_file.model");
    assert!(result.is_err(), "Should error on missing file");
}

#[test]
fn test_compute_features_dimensions() {
    let pcm_data: Vec<i16> = vec![0; 16000];
    let sample_rate = 16000;

    let result = compute_features(&pcm_data, sample_rate);
    assert!(result.is_ok());

    let (features, num_frames) = result.unwrap();
    let expected_num_frames = (pcm_data.len() - 400) / 160 + 1;
    assert_eq!(num_frames, expected_num_frames);
    assert_eq!(features.len(), 128 * expected_num_frames);
}

#[test]
fn test_compute_features_invalid_sample_rate() {
    let pcm_data: Vec<i16> = vec![0; 8000];
    let result = compute_features(&pcm_data, 8000);
    assert!(result.is_err());
}

#[test]
fn test_compute_features_too_short() {
    let pcm_data: Vec<i16> = vec![0; 300];
    let result = compute_features(&pcm_data, 16000);
    assert!(result.is_err());
}

#[test]
fn test_compute_features_normalization() {
    let mut pcm_data: Vec<i16> = Vec::with_capacity(16000);
    for i in 0..16000 {
        let sample =
            (10000.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()) as i16;
        pcm_data.push(sample);
    }

    let (features, num_frames) = compute_features(&pcm_data, 16000).unwrap();

    for bin_idx in [0, 32, 64, 96, 127] {
        let mut bin_values = Vec::new();
        for frame_idx in 0..num_frames {
            bin_values.push(features[bin_idx * num_frames + frame_idx]);
        }
        let mean: f32 = bin_values.iter().sum::<f32>() / bin_values.len() as f32;
        let variance: f32 =
            bin_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / bin_values.len() as f32;
        let std = variance.sqrt();
        assert!(mean.abs() < 1e-4, "Bin {} mean {} should be ~0", bin_idx, mean);
        assert!((std - 1.0).abs() < 1e-4, "Bin {} std {} should be ~1", bin_idx, std);
    }
}

#[test]
#[ignore] // Integration test - requires model files
fn test_asrcore_basic_decode() {
    use crate::Inference;

    let inference = Inference::cpu().expect("Failed to create inference");
    let vocab_path = format!("{}/../../data/parakeet/tokenizer.model", env!("CARGO_MANIFEST_DIR"));
    let encoder_path = format!("{}/../../data/parakeet/encoder.int8.onnx", env!("CARGO_MANIFEST_DIR"));
    let decoder_path = format!("{}/../../data/parakeet/decoder_joint.int8.onnx", env!("CARGO_MANIFEST_DIR"));

    let encoder = inference.onnx_session(&encoder_path).expect("Failed to load encoder");
    let decoder_joint = inference.onnx_session(&decoder_path).expect("Failed to load decoder_joint");
    let tokens = load_tokens(&vocab_path).expect("Failed to load tokens");

    let mut core = crate::asr::parakeet::asrcore::AsrCore::new(encoder, decoder_joint, tokens)
        .expect("Failed to create AsrCore");

    let pcm_data: Vec<i16> = vec![0; 16000];
    let (features, num_frames) = compute_features(&pcm_data, 16000).expect("Failed to compute features");

    let result = core.decode_chunk(&features, num_frames);
    assert!(result.is_ok(), "decode_chunk should succeed: {:?}", result.err());
}

#[test]
#[ignore] // Integration test - requires model files and tokio runtime
fn test_parakeet_send_recv() {
    use crate::asr::parakeet::parakeet::Parakeet;
    use crate::Inference;
    use audio::{AudioData, AudioSample};
    use base::Tensor;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let inference = Inference::cpu().expect("Failed to create inference");
        let vocab_path = format!("{}/../../data/parakeet/tokenizer.model", env!("CARGO_MANIFEST_DIR"));
        let encoder_path = format!("{}/../../data/parakeet/encoder.int8.onnx", env!("CARGO_MANIFEST_DIR"));
        let decoder_path = format!("{}/../../data/parakeet/decoder_joint.int8.onnx", env!("CARGO_MANIFEST_DIR"));

        let encoder = inference.onnx_session(&encoder_path).expect("Failed to load encoder");
        let decoder_joint = inference.onnx_session(&decoder_path).expect("Failed to load decoder_joint");

        let mut parakeet = Parakeet::new(encoder, decoder_joint, &vocab_path)
            .expect("Failed to create Parakeet");

        let pcm_data: Vec<i16> = vec![0; 32000];
        let sample = AudioSample {
            sample_rate: 16000,
            data: AudioData::Pcm(Tensor { data: pcm_data, shape: vec![32000] }),
        };

        parakeet.send(sample).await.expect("Failed to send audio");
        parakeet.close().await.expect("Failed to close");

        let mut count = 0;
        while let Some(result) = parakeet.recv().await {
            result.expect("Transcription error");
            count += 1;
        }
        assert!(count > 0, "Should receive at least one transcription");
    });
}

#[test]
fn test_parakeet_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<crate::asr::parakeet::Parakeet>();
}
