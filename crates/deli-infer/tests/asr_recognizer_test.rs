// Tests for ASR SpeechRecognizer public API

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use deli_base::Tensor as BaseTensor;
use deli_infer::asr::{Config, Whisper};
use deli_infer::{Inference, SpeechRecognizer};
use std::path::Path;

/// Build a SpeechRecognizer from VarMap random weights (no model files needed).
/// Uses small max_target_positions so decode loops terminate quickly with random weights.
fn build_test_recognizer() -> SpeechRecognizer {
    let mut config = Config::tiny_en();
    config.max_target_positions = 10; // Fast termination with random weights
    let device = Device::Cpu;

    // Build model with VarMap
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = Whisper::load(vb, config.clone()).expect("build model");

    // Save model to temp file
    let temp_dir = std::env::temp_dir().join("deli_infer_test");
    std::fs::create_dir_all(&temp_dir).expect("create temp dir");
    let model_path = temp_dir.join("test_model.safetensors");
    varmap.save(&model_path).expect("save varmap");

    // Create tokenizer JSON with Whisper special tokens
    let tokenizer_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 50258, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50359, "content": "<|transcribe|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50363, "content": "<|notimestamps|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50257, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50362, "content": "<|nocaptions|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "model": {
            "type": "BPE",
            "vocab": {"hello": 0, "world": 1, " ": 2},
            "merges": []
        }
    }"#;

    let tokenizer_path = temp_dir.join("test_tokenizer.json");
    std::fs::write(&tokenizer_path, tokenizer_json).expect("write tokenizer");

    SpeechRecognizer::new_with_config(model_path, tokenizer_path, config, device)
        .expect("create recognizer")
}

#[tokio::test]
async fn test_sample_rate_validation() {
    let recognizer = build_test_recognizer();

    // Wrong sample rate should be rejected
    let samples: Vec<i16> = vec![0; 16000];
    let audio = BaseTensor::new(vec![16000], samples).expect("create tensor");
    let result = recognizer.transcribe(&audio, 44100).await;

    assert!(result.is_err(), "should reject wrong sample rate");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("16000"),
        "error should mention required sample rate, got: {err_msg}"
    );
}

#[tokio::test]
async fn test_empty_audio_validation() {
    let recognizer = build_test_recognizer();

    // Empty audio should be rejected
    let audio = BaseTensor::new(vec![0], vec![]).expect("create tensor");
    let result = recognizer.transcribe(&audio, 16000).await;

    assert!(result.is_err(), "should reject empty audio");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.to_lowercase().contains("empty"),
        "error should mention empty audio, got: {err_msg}"
    );
}

#[tokio::test]
async fn test_non_1d_tensor_validation() {
    let recognizer = build_test_recognizer();

    // 2D tensor should be rejected
    let samples: Vec<i16> = vec![0; 32000];
    let audio = BaseTensor::new(vec![2, 16000], samples).expect("create tensor");
    let result = recognizer.transcribe(&audio, 16000).await;

    assert!(result.is_err(), "should reject non-1D tensor");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("1D") || err_msg.contains("shape"),
        "error should mention shape requirement, got: {err_msg}"
    );
}

#[tokio::test]
async fn test_transcribe_happy_path() {
    let recognizer = build_test_recognizer();

    // 1-second of silence at 16kHz
    let samples: Vec<i16> = vec![0; 16000];
    let audio = BaseTensor::new(vec![16000], samples).expect("create tensor");

    // Full pipeline: audio → mel → model → tokens → text
    // With random weights the text is meaningless, but the pipeline should not error
    let result = recognizer.transcribe(&audio, 16000).await;
    assert!(
        result.is_ok(),
        "transcribe should succeed with valid input: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_inference_factory_signature() {
    let inference = Inference::cpu();

    // Try to create - will fail since files don't exist, but that's expected
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("nonexistent_model.safetensors");
    let tokenizer_path = temp_dir.join("nonexistent_tokenizer.json");
    let config_path = temp_dir.join("nonexistent_config.json");

    let result = inference.use_speech_recognizer(&model_path, &tokenizer_path, &config_path);
    assert!(result.is_err(), "should fail with non-existent files");
}

#[tokio::test]
#[ignore] // Only run if real model files are available
async fn test_with_real_model() {
    let model_path = Path::new("models/whisper-tiny.en/model.safetensors");
    let tokenizer_path = Path::new("models/whisper-tiny.en/tokenizer.json");
    let config_path = Path::new("models/whisper-tiny.en/config.json");

    if !model_path.exists() || !tokenizer_path.exists() || !config_path.exists() {
        println!("Skipping test - model files not found at models/whisper-tiny.en/");
        return;
    }

    let inference = Inference::cpu();
    let recognizer = inference
        .use_speech_recognizer(model_path, tokenizer_path, config_path)
        .expect("create recognizer");

    // Test sample rate validation
    let samples: Vec<i16> = vec![0; 44100];
    let audio = BaseTensor::new(vec![44100], samples).expect("create tensor");
    let result = recognizer.transcribe(&audio, 44100).await;
    assert!(result.is_err(), "should reject wrong sample rate");

    // Test with correct sample rate - 1-second silence
    let samples: Vec<i16> = vec![0; 16000];
    let audio = BaseTensor::new(vec![16000], samples).expect("create tensor");
    let text = recognizer
        .transcribe(&audio, 16000)
        .await
        .expect("transcribe");
    println!("Transcription of silence: {:?}", text);

    // Generate 1-second 440Hz sine wave
    let num_samples = 16000;
    let samples: Vec<i16> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / 16000.0;
            (0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 32767.0) as i16
        })
        .collect();

    let audio = BaseTensor::new(vec![num_samples], samples).expect("create tensor");
    let text = recognizer
        .transcribe(&audio, 16000)
        .await
        .expect("transcribe");
    println!("Transcription of 440Hz sine wave: {:?}", text);
    assert!(!text.is_empty(), "should produce some transcription");
}

#[test]
fn test_public_api_exports() {
    // Verify SpeechRecognizer is publicly exported
    let _: Option<SpeechRecognizer> = None;
    let _inference = Inference::cpu();
}
