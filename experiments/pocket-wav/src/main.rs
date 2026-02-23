use {
    audio::AudioData,
    base::*,
    futures_util::{SinkExt, StreamExt},
    inference::Inference,
    std::path::PathBuf,
};

const SENTENCE: &str = "To be, or not to be, equals, minus one.";
const SAMPLE_RATE: u32 = 24000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <output.wav>", args[0]);
        std::process::exit(1);
    }

    let output_path = &args[1];

    // Model paths
    let data = PathBuf::from("data/pocket");
    let text_conditioner = data.join("text_conditioner.onnx");
    let flow_main = data.join("flow_lm_main_int8.onnx");
    let flow_step = data.join("flow_lm_flow_int8.onnx");
    let mimi_encoder = data.join("mimi_encoder.onnx");
    let mimi_decoder = data.join("mimi_decoder_int8.onnx");
    let tokenizer = data.join("tokenizer.json");
    let voice = data.join("voice.wav");

    // Validate model files exist
    for path in [&text_conditioner, &flow_main, &flow_step, &mimi_encoder, &mimi_decoder, &tokenizer, &voice] {
        if !path.exists() {
            eprintln!("Missing: {}", path.display());
            std::process::exit(1);
        }
    }

    // Initialize inference and load Pocket TTS
    log_info!("Initializing Pocket TTS...");
    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;
    let mut tts = inference.use_pocket_tts(
        &text_conditioner,
        &flow_main,
        &flow_step,
        &mimi_encoder,
        &mimi_decoder,
        &tokenizer,
        &voice,
    )?;
    log_info!("Pocket TTS loaded");

    // Synthesize speech
    log_info!("Synthesizing: \"{}\"", SENTENCE);
    tts.send(SENTENCE.to_string()).await?;
    tts.close().await?;

    let sample = tts.next().await.expect("stream should yield audio")?;
    let AudioData::Pcm(tensor) = sample.data;
    log_info!("Generated {} samples", tensor.data.len());

    // Write WAV file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &s in &tensor.data {
        writer.write_sample(s)?;
    }
    writer.finalize()?;

    println!("Wrote {} samples to {}", tensor.data.len(), output_path);
    Ok(())
}
