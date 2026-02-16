// Tests for ASR token decoder (greedy decoding loop)

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use deli_infer::asr::{Config, TokenDecoder, Whisper};
use tokenizers::Tokenizer;

fn build_test_tokenizer() -> Tokenizer {
    let tokenizer_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 50258, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50359, "content": "<|transcribe|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50363, "content": "<|notimestamps|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50257, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 50362, "content": "<|nospeech|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
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
    serde_json::from_str(tokenizer_json).expect("parse tokenizer")
}

/// Build a decoder with small max_target_positions for fast tests.
/// Random weights never produce EOT, so we limit the max tokens to keep tests quick.
fn build_test_decoder() -> (TokenDecoder, Config) {
    let device = Device::Cpu;
    let mut config = Config::tiny_en();
    // Use small max_target_positions so decode loop terminates quickly (max_tokens = 10/2 = 5)
    config.max_target_positions = 10;
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Whisper::load(vb.pp("model"), config.clone()).expect("load model");
    let tokenizer = build_test_tokenizer();
    let decoder = TokenDecoder::new(model, tokenizer, &device, &config).expect("create decoder");
    (decoder, config)
}

#[test]
fn test_token_id_lookup() {
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("<|endoftext|>".to_string(), 50257);
    vocab.insert("<|startoftranscript|>".to_string(), 50258);
    vocab.insert("<|transcribe|>".to_string(), 50359);
    vocab.insert("<|notimestamps|>".to_string(), 50363);

    let vocab_json = serde_json::to_string(&vocab).expect("serialize vocab");
    let tokenizer_json = format!(
        r#"{{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {{"id": 50258, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 50359, "content": "<|transcribe|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 50363, "content": "<|notimestamps|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 50257, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {{
            "type": "BPE",
            "vocab": {},
            "merges": []
        }}
    }}"#,
        vocab_json
    );

    let tokenizer: Tokenizer = serde_json::from_str(&tokenizer_json).expect("parse tokenizer");

    let sot = deli_infer::asr::token_id(&tokenizer, "<|startoftranscript|>").expect("SOT token");
    assert_eq!(sot, 50258);

    let transcribe =
        deli_infer::asr::token_id(&tokenizer, "<|transcribe|>").expect("transcribe token");
    assert_eq!(transcribe, 50359);

    let eot = deli_infer::asr::token_id(&tokenizer, "<|endoftext|>").expect("EOT token");
    assert_eq!(eot, 50257);

    let no_timestamps =
        deli_infer::asr::token_id(&tokenizer, "<|notimestamps|>").expect("no_timestamps token");
    assert_eq!(no_timestamps, 50363);
}

#[test]
fn test_token_id_missing_token() {
    let tokenizer_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": {},
            "merges": []
        }
    }"#;

    let tokenizer: Tokenizer = serde_json::from_str(tokenizer_json).expect("parse tokenizer");
    let result = deli_infer::asr::token_id(&tokenizer, "<|missing|>");
    assert!(result.is_err());
}

#[test]
fn test_decoder_construction() {
    let (_decoder, _config) = build_test_decoder();
}

#[test]
fn test_decoder_decode_terminates() {
    let (mut decoder, _config) = build_test_decoder();
    let device = Device::Cpu;

    // Create mel spectrogram input: [1, 80, 3000] (30s of audio)
    let mel = Tensor::zeros((1, 80, 3000), DType::F32, &device).expect("create mel");

    // decode() should terminate (via EOT or max token limit) and return Ok
    let result = decoder.decode(&mel);
    assert!(result.is_ok(), "decode should not error: {:?}", result.err());

    let decoding_result = result.unwrap();
    // With random weights, text may be empty or garbage, but should be valid String
    let _ = decoding_result.text.as_str();
    // no_speech_prob should be a valid probability
    assert!(
        (0.0..=1.0).contains(&decoding_result.no_speech_prob),
        "no_speech_prob should be in [0, 1], got {}",
        decoding_result.no_speech_prob
    );
}

#[test]
fn test_decoder_run_single_chunk() {
    let (mut decoder, _config) = build_test_decoder();
    let device = Device::Cpu;

    // Single 30s chunk: [1, 80, 3000]
    let mel = Tensor::zeros((1, 80, 3000), DType::F32, &device).expect("create mel");

    let result = decoder.run(&mel);
    assert!(result.is_ok(), "run should not error: {:?}", result.err());
}

#[test]
fn test_decoder_run_multi_chunk() {
    let (mut decoder, _config) = build_test_decoder();
    let device = Device::Cpu;

    // 45s of audio: [1, 80, 4500] → should process as 2 chunks (3000 + 1500)
    let mel = Tensor::zeros((1, 80, 4500), DType::F32, &device).expect("create mel");

    let result = decoder.run(&mel);
    assert!(
        result.is_ok(),
        "run with multi-chunk should not error: {:?}",
        result.err()
    );
}

#[test]
fn test_decoder_multi_chunk_preparation() {
    let device = Device::Cpu;

    let mel =
        Tensor::zeros((1, 80, 4500), candle_core::DType::F32, &device).expect("mel tensor");

    let (batch, bins, frames) = mel.dims3().expect("dims3");
    assert_eq!(batch, 1);
    assert_eq!(bins, 80);
    assert_eq!(frames, 4500);

    const N_FRAMES: usize = 3000;
    let chunk1_frames = N_FRAMES.min(frames);
    let chunk2_start = chunk1_frames;
    let chunk2_frames = (frames - chunk2_start).min(N_FRAMES);

    assert_eq!(chunk1_frames, 3000);
    assert_eq!(chunk2_start, 3000);
    assert_eq!(chunk2_frames, 1500);
}
