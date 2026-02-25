use {
    audio::{AudioData, AudioOut, AudioOutConfig},
    base::*,
    futures_util::{SinkExt, StreamExt},
    inference::Inference,
    std::time::Duration,
};

const SENTENCE: &str = "To be, or not to be, equals, minus one.";
const SAMPLE_RATE: usize = 24000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Initialize inference and load Kokoro model
    log_info!("Initializing Kokoro TTS...");
    let inference = Inference::cpu()?;
    let mut kokoro = inference.use_kokoro()?;
    log_info!("Kokoro model loaded");

    // Synthesize speech
    log_info!("Synthesizing: \"{}\"", SENTENCE);
    kokoro.send(SENTENCE.to_string()).await?;
    kokoro.close().await?;

    let sample = kokoro.next().await.expect("stream should yield audio")?;
    let AudioData::Pcm(ref tensor) = sample.data;
    log_info!("Generated {} samples", tensor.data.len());

    // Create AudioOut and play samples
    let mut audioout = AudioOut::open(None).await;
    audioout
        .select(AudioOutConfig {
            sample_rate: SAMPLE_RATE,
            ..Default::default()
        })
        .await;
    let num_samples = tensor.data.len();
    log_info!("Sending {} samples to AudioOut", num_samples);
    audioout.play(sample).await;

    // Wait for playback to complete (AudioOut doesn't flush on drop)
    let duration_secs = num_samples as f64 / SAMPLE_RATE as f64;
    let duration_ms = (duration_secs * 1000.0) as u64 + 500;
    log_info!("Waiting {:.2}s for playback to complete...", duration_secs);
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;

    log_info!("Playback complete");
    Ok(())
}
