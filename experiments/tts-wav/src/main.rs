use deli_base::log;
use deli_infer::Inference;
use std::path::PathBuf;

const SENTENCE: &str = "To be, or not to be, equals, minus one.";
const SAMPLE_RATE: u32 = 24000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <output.wav>", args[0]);
        std::process::exit(1);
    }

    let output_path = &args[1];

    // Model paths
    let model_path = PathBuf::from("models/kokoro/kokoro-v1.0.onnx");
    let voice_path = PathBuf::from("models/kokoro/bf_emma.npy");
    let espeak_data_path = "/usr/lib/x86_64-linux-gnu/espeak-ng-data";

    // Validate model files exist
    if !model_path.exists() || !voice_path.exists() {
        eprintln!("Model files missing. Expected:");
        eprintln!("  - models/kokoro/kokoro-v1.0.onnx");
        eprintln!("  - models/kokoro/bf_emma.npy");
        std::process::exit(1);
    }

    // Initialize inference and load Kokoro model
    log::info!("Initializing Kokoro TTS...");
    let inference = Inference::cpu();
    let kokoro = inference.use_kokoro(&model_path, &voice_path, Some(espeak_data_path))?;
    log::info!("Kokoro model loaded");

    // Synthesize speech
    log::info!("Synthesizing: \"{}\"", SENTENCE);
    let tensor = kokoro.speak(SENTENCE).await?;
    log::info!("Generated {} samples", tensor.data.len());

    // Write WAV file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &sample in &tensor.data {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    println!("Wrote {} samples to {}", tensor.data.len(), output_path);
    Ok(())
}
