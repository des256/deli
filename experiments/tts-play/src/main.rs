use audio::{AudioData, AudioOut, AudioOutConfig};
use base::log;
use futures_util::{SinkExt, StreamExt};
use inference::Inference;
use std::path::PathBuf;
use std::time::Duration;

const SENTENCE: &str = "To be, or not to be, equals, minus one.";
const SAMPLE_RATE: usize = 24000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Model paths
    let model_path = PathBuf::from("data/kokoro/kokoro-v1.0.onnx");
    //let voice_path = PathBuf::from("data/kokoro/af_kore.npy");  // good general voice
    //let voice_path = PathBuf::from("data/kokoro/af_jessica.npy");  // annoying, but useful
    let voice_path = PathBuf::from("data/kokoro/af_nicole.npy"); // this could be the best one
    //let voice_path = PathBuf::from("data/kokoro/af_river.npy");  // ok, but dry
    //let voice_path = PathBuf::from("data/kokoro/af_sarah.npy");  // robotic
    //let voice_path = PathBuf::from("data/kokoro/af_sky.npy"); // slow but clear
    //let voice_path = PathBuf::from("data/kokoro/bf_emma.npy"); // dry, british
    //let voice_path = PathBuf::from("data/kokoro/bf_lily.npy"); // more colorful, british
    //let voice_path = PathBuf::from("data/kokoro/ff_siwis.npy"); // french
    //let voice_path = PathBuf::from("data/kokoro/hf_beta.npy"); // indian
    //let voice_path = PathBuf::from("data/kokoro/jf_gongitsune.npy"); // japanese, dramatic
    //let voice_path = PathBuf::from("data/kokoro/jf_tebukuro.npy"); // japanese, high pitched
    //let voice_path = PathBuf::from("data/kokoro/zf_xiaoni.npy"); // chinese, chinese intonation
    //let voice_path = PathBuf::from("data/kokoro/zf_xiaoyi.npy"); // chinese, high pitched
    let espeak_data_path = "/usr/lib/x86_64-linux-gnu/espeak-ng-data";

    // Validate model files exist
    if !model_path.exists() || !voice_path.exists() {
        eprintln!("Model files missing. Expected:");
        eprintln!("  - data/kokoro/kokoro-v1.0.onnx");
        eprintln!("  - data/kokoro/<voice>.npy");
        std::process::exit(1);
    }

    // Initialize inference and load Kokoro model
    log::info!("Initializing Kokoro TTS...");
    let inference = Inference::cpu();
    let mut kokoro = inference.use_kokoro(&model_path, &voice_path, Some(espeak_data_path))?;
    log::info!("Kokoro model loaded");

    // Synthesize speech
    log::info!("Synthesizing: \"{}\"", SENTENCE);
    kokoro.send(SENTENCE.to_string()).await?;
    kokoro.close().await?;

    let sample = kokoro.next().await.expect("stream should yield audio")?;
    let AudioData::Pcm(ref tensor) = sample.data;
    log::info!("Generated {} samples", tensor.data.len());

    // Create AudioOut and play samples
    let mut audioout = AudioOut::open().await;
    audioout
        .select(AudioOutConfig {
            sample_rate: SAMPLE_RATE,
            ..Default::default()
        })
        .await;
    let num_samples = tensor.data.len();
    log::info!("Sending {} samples to AudioOut", num_samples);
    audioout.play(sample).await;

    // Wait for playback to complete (AudioOut doesn't flush on drop)
    let duration_secs = num_samples as f64 / SAMPLE_RATE as f64;
    let duration_ms = (duration_secs * 1000.0) as u64 + 500;
    log::info!("Waiting {:.2}s for playback to complete...", duration_secs);
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;

    log::info!("Playback complete");
    Ok(())
}
