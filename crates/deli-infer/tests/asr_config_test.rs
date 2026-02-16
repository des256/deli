use deli_infer::asr::Config;

#[test]
fn test_config_tiny_en() {
    let config = Config::tiny_en();

    // Verify key parameters for tiny.en model
    assert_eq!(config.num_mel_bins, 80);
    assert_eq!(config.d_model, 384);
    assert_eq!(config.encoder_layers, 4);
    assert_eq!(config.decoder_layers, 4);
    assert_eq!(config.encoder_attention_heads, 6);
    assert_eq!(config.decoder_attention_heads, 6);
    assert_eq!(config.max_source_positions, 1500);
    assert_eq!(config.max_target_positions, 448);
    assert_eq!(config.vocab_size, 51865);

    // Verify suppress_tokens is a non-empty vec
    assert!(!config.suppress_tokens.is_empty());
}

#[test]
fn test_config_deserialize() {
    let json = r#"{
        "num_mel_bins": 80,
        "max_source_positions": 1500,
        "d_model": 384,
        "encoder_attention_heads": 6,
        "encoder_layers": 4,
        "vocab_size": 51865,
        "max_target_positions": 448,
        "decoder_attention_heads": 6,
        "decoder_layers": 4,
        "suppress_tokens": [1, 2, 7, 8, 9, 10, 14, 25]
    }"#;

    let config: Config = serde_json::from_str(json).unwrap();

    assert_eq!(config.num_mel_bins, 80);
    assert_eq!(config.d_model, 384);
    assert_eq!(config.suppress_tokens.len(), 8);
}

#[test]
fn test_constants_match_whisper_spec() {
    use deli_infer::asr::config::*;

    assert_eq!(SAMPLE_RATE, 16000);
    assert_eq!(N_FFT, 400);
    assert_eq!(HOP_LENGTH, 160);
    assert_eq!(CHUNK_LENGTH, 30);
    assert_eq!(N_SAMPLES, 480000);  // 30 seconds at 16kHz
    assert_eq!(N_FRAMES, 3000);     // N_SAMPLES / HOP_LENGTH = 3000
    assert!((NO_SPEECH_THRESHOLD - 0.6).abs() < 1e-6);
    assert!((LOGPROB_THRESHOLD - (-1.0)).abs() < 1e-6);
    assert!((COMPRESSION_RATIO_THRESHOLD - 2.4).abs() < 1e-6);
}
