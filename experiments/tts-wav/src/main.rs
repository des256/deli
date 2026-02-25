use {
    audio::AudioData,
    base::*,
    futures_util::{SinkExt, StreamExt},
    inference::Inference,
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

    // Initialize inference and load Kokoro model
    log_info!("Initializing Kokoro TTS...");
    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;
    let mut kokoro = inference.use_kokoro()?;
    log_info!("Kokoro model loaded");

    // Synthesize speech
    log_info!("Synthesizing: \"{}\"", SENTENCE);
    kokoro.send(SENTENCE.to_string()).await?;
    kokoro.close().await?;

    let sample = kokoro.next().await.expect("stream should yield audio")?;
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
