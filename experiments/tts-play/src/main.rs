use deli_audio::{AudioOut, AudioSample};
use deli_base::log;
use deli_infer::Inference;
use futures_util::SinkExt;
use std::path::PathBuf;
use std::time::Duration;

const SENTENCE: &str = "To be, or not to be, equals, minus one.";
const SAMPLE_RATE: u32 = 24000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

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

    // Create AudioOut and play samples
    let mut audio_out = AudioOut::new(None, SAMPLE_RATE);
    let num_samples = tensor.data.len();
    log::info!("Sending {} samples to AudioOut", num_samples);
    audio_out.send(AudioSample::Pcm(tensor)).await?;

    // Wait for playback to complete (AudioOut doesn't flush on drop)
    let duration_secs = num_samples as f64 / SAMPLE_RATE as f64;
    let duration_ms = (duration_secs * 1000.0) as u64 + 500;
    log::info!("Waiting {:.2}s for playback to complete...", duration_secs);
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;

    log::info!("Playback complete");
    Ok(())
}
