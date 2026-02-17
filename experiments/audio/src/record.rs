use deli_audio::AudioIn;
use deli_base::log;
use hound;

const SAMPLE_RATE: u32 = 16000;
const CHUNK_FRAMES: u32 = 1600;  // 100ms chunks
const DURATION_SECONDS: u32 = 5;
const TOTAL_SAMPLES: usize = (SAMPLE_RATE * DURATION_SECONDS) as usize;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <output-wav-path>", args[0]);
        std::process::exit(1);
    }
    let output_path = &args[1];

    log::info!("Recording 5 seconds...");

    // Create AudioIn and capture samples
    let mut audio_in = AudioIn::new(None, SAMPLE_RATE, CHUNK_FRAMES);
    let mut all_samples: Vec<i16> = Vec::with_capacity(TOTAL_SAMPLES);

    while all_samples.len() < TOTAL_SAMPLES {
        let chunk = audio_in.recv().await?;
        all_samples.extend(chunk);
    }

    // Truncate to exactly 80000 samples
    all_samples.truncate(TOTAL_SAMPLES);

    log::info!("Recorded {} samples", all_samples.len());

    // Write to WAV file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for sample in &all_samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    log::info!("Saved to {}", output_path);
    Ok(())
}
