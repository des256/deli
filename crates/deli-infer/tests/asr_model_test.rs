use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use deli_infer::asr::{Config, WhisperModel};

#[test]
fn test_whisper_encoder_forward() {
    let config = Config::tiny_en();
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Create model with random weights
    let model = WhisperModel::load(vb, config.clone()).expect("Failed to load model");

    // Test encoder: input [1, 80, 3000] → output [1, 1500, 384]
    let mel_input = Tensor::zeros(&[1, 80, 3000], DType::F32, &device).unwrap();
    let encoder_output = model.encoder.forward(&mel_input, true).unwrap();

    assert_eq!(encoder_output.dims(), &[1, 1500, 384]);
}

#[test]
fn test_whisper_decoder_forward() {
    let config = Config::tiny_en();
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Create model with random weights
    let mut model = WhisperModel::load(vb, config.clone()).expect("Failed to load model");

    // Encoder output: [1, 1500, 384]
    let encoder_output = Tensor::zeros(&[1, 1500, 384], DType::F32, &device).unwrap();

    // Token input: [1, 4] (4 tokens)
    let tokens = Tensor::zeros(&[1, 4], DType::U32, &device).unwrap();

    // Decoder forward
    let logits = model.decoder.forward(&tokens, &encoder_output).unwrap();

    // Expected output: [1, 4, 51865] (batch, seq_len, vocab_size)
    assert_eq!(logits.dims(), &[1, 4, 51865]);
}

#[test]
fn test_whisper_full_forward() {
    let config = Config::tiny_en();
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Create model
    let mut model = WhisperModel::load(vb, config.clone()).expect("Failed to load model");

    // Mel input: [1, 80, 3000]
    let mel = Tensor::zeros(&[1, 80, 3000], DType::F32, &device).unwrap();

    // Tokens: [1, 4]
    let tokens = Tensor::zeros(&[1, 4], DType::U32, &device).unwrap();

    // Full forward pass
    let logits = model.forward(&mel, &tokens).unwrap();

    // Expected: [1, 4, 51865]
    assert_eq!(logits.dims(), &[1, 4, 51865]);
}

#[test]
#[ignore] // Only runs if real model files exist
fn test_whisper_load_real_model() {
    use std::path::Path;

    let model_path = Path::new("models/whisper-tiny.en/model.safetensors");
    if !model_path.exists() {
        // Skip test if model files not present
        return;
    }

    let config = Config::tiny_en();
    let device = Device::Cpu;

    let weights = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
    }
    .expect("Failed to load safetensors");

    let _model =
        WhisperModel::load(weights, config).expect("Failed to load model from real weights");
}
