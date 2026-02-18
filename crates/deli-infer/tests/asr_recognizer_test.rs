// Tests for ASR Whisper public API (Sink<AudioSample> + Stream<Transcription>)

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use deli_audio::{AudioData, AudioSample};
use deli_base::Tensor as BaseTensor;
use deli_infer::asr::{Config, Transcription, WhisperModel};
use deli_infer::{Inference, Whisper};
use futures_util::{SinkExt, StreamExt};
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Build a Whisper recognizer from VarMap random weights (no model files needed).
/// Uses small max_target_positions so decode loops terminate quickly with random weights.
fn build_test_whisper() -> Whisper {
    let mut config = Config::tiny_en();
    config.max_target_positions = 10; // Fast termination with random weights
    let device = Device::Cpu;

    // Build model with VarMap
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = WhisperModel::load(vb, config.clone()).expect("build model");

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

    Whisper::new_with_config(model_path, tokenizer_path, config, device)
        .expect("create whisper")
}

fn make_audio_sample(samples: Vec<i16>, sample_rate: usize) -> AudioSample {
    let tensor = BaseTensor::new(vec![samples.len()], samples).expect("create tensor");
    AudioSample {
        data: AudioData::Pcm(tensor),
        sample_rate,
    }
}

#[tokio::test]
async fn test_sample_rate_validation() {
    let mut whisper = build_test_whisper();

    // Wrong sample rate should be rejected
    let sample = make_audio_sample(vec![0; 16000], 44100);
    let result = whisper.send(sample).await;

    assert!(result.is_err(), "should reject wrong sample rate");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("16000"),
        "error should mention required sample rate, got: {err_msg}"
    );
}

#[tokio::test]
async fn test_sink_stream_happy_path() {
    let mut whisper = build_test_whisper()
        .with_window_samples(16000); // 1-second window

    // Send 1 second of silence at 16kHz
    let sample = make_audio_sample(vec![0; 16000], SAMPLE_RATE);
    whisper.send(sample).await.expect("send audio");
    whisper.close().await.expect("close sink");

    // Should produce a transcription result
    let result = whisper.next().await;
    assert!(result.is_some(), "should produce a transcription");
    let transcription = result.unwrap().expect("transcription should succeed");
    match transcription {
        Transcription::Final { text, .. } => {
            // With random weights the text is meaningless, but the pipeline should complete
            let _ = text;
        }
        _ => panic!("expected Transcription::Final"),
    }
}

#[tokio::test]
async fn test_close_flushes_remaining_buffer() {
    let mut whisper = build_test_whisper()
        .with_window_samples(32000); // 2-second window

    // Send only 1 second (less than window), then close
    let sample = make_audio_sample(vec![0; 16000], SAMPLE_RATE);
    whisper.send(sample).await.expect("send audio");
    whisper.close().await.expect("close sink");

    // Closing should flush the remaining buffer
    let result = whisper.next().await;
    assert!(result.is_some(), "close should flush remaining audio");
    result.unwrap().expect("flushed transcription should succeed");

    // Stream should now be complete
    let end = whisper.next().await;
    assert!(end.is_none(), "stream should end after flush");
}

#[tokio::test]
async fn test_stream_terminates_on_close_empty() {
    let mut whisper = build_test_whisper();

    // Close without sending any audio
    whisper.close().await.expect("close sink");

    // Stream should immediately return None (no audio to transcribe)
    let result = whisper.next().await;
    assert!(result.is_none(), "empty stream should return None");
}

#[tokio::test]
async fn test_multiple_windows() {
    let mut whisper = build_test_whisper()
        .with_window_samples(8000); // 0.5-second window

    // Send 1 second total (2 windows worth)
    let sample = make_audio_sample(vec![0; 16000], SAMPLE_RATE);
    whisper.send(sample).await.expect("send audio");
    whisper.close().await.expect("close sink");

    // Should produce 2 transcriptions
    let r1 = whisper.next().await;
    assert!(r1.is_some(), "should produce first transcription");
    r1.unwrap().expect("first transcription ok");

    let r2 = whisper.next().await;
    assert!(r2.is_some(), "should produce second transcription");
    r2.unwrap().expect("second transcription ok");

    // Stream should end
    let end = whisper.next().await;
    assert!(end.is_none(), "stream should end");
}

#[tokio::test]
async fn test_inference_factory_signature() {
    let inference = Inference::cpu();

    // Try to create - will fail since files don't exist, but that's expected
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("nonexistent_model.safetensors");
    let tokenizer_path = temp_dir.join("nonexistent_tokenizer.json");
    let config_path = temp_dir.join("nonexistent_config.json");

    let result = inference.use_whisper(&model_path, &tokenizer_path, &config_path);
    assert!(result.is_err(), "should fail with non-existent files");
}

#[tokio::test]
async fn test_implements_sink_and_stream() {
    fn assert_sink<T: futures_sink::Sink<AudioSample>>() {}
    fn assert_stream<T: futures_core::Stream>() {}
    assert_sink::<Whisper>();
    assert_stream::<Whisper>();
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
    let mut whisper = inference
        .use_whisper(model_path, tokenizer_path, config_path)
        .expect("create whisper")
        .with_window_samples(16000); // 1-second window

    // Test sample rate validation
    let sample = make_audio_sample(vec![0; 44100], 44100);
    let result = whisper.send(sample).await;
    assert!(result.is_err(), "should reject wrong sample rate");

    // Test with correct sample rate - 1-second silence
    let sample = make_audio_sample(vec![0; 16000], SAMPLE_RATE);
    whisper.send(sample).await.expect("send audio");
    whisper.close().await.expect("close sink");

    let result = whisper.next().await;
    assert!(result.is_some(), "should produce transcription");
    match result.unwrap().expect("transcription ok") {
        Transcription::Final { text, .. } => println!("Transcription of silence: {:?}", text),
        _ => panic!("expected Final"),
    }
}

#[test]
fn test_public_api_exports() {
    // Verify Whisper is publicly exported
    let _: Option<Whisper> = None;
    let _inference = Inference::cpu();
}
