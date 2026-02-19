use audio::{AudioData, AudioOut, AudioSample};
use base::{Tensor, log};
use hound;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav-file>", args[0]);
        std::process::exit(1);
    }
    let wav_path = &args[1];

    // Open and read WAV file
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();

    log::info!(
        "WAV format: {} Hz, {} channel(s), {} bits per sample",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample
    );

    if spec.bits_per_sample != 16 {
        log::info!("Converting {}-bit to 16-bit", spec.bits_per_sample);
    }

    // Read all samples as i16
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;

    // Convert to mono if needed
    let mono_samples: Vec<i16> = match spec.channels {
        1 => samples,
        2 => {
            log::info!("Downmixing stereo to mono");
            samples
                .chunks_exact(2)
                .map(|pair| ((pair[0] as i32 + pair[1] as i32) / 2) as i16)
                .collect()
        }
        _ => {
            eprintln!(
                "Error: {} channels not supported (only mono and stereo)",
                spec.channels
            );
            std::process::exit(1);
        }
    };

    let sample_count = mono_samples.len();
    let duration_secs = sample_count as f64 / spec.sample_rate as f64;

    log::info!(
        "Playing {} ({} Hz, {:.2}s)...",
        wav_path,
        spec.sample_rate,
        duration_secs
    );

    // Create AudioOut and send samples
    let audioout = AudioOut::open().await;
    let tensor = Tensor::new(vec![mono_samples.len()], mono_samples).unwrap();
    audioout
        .play(AudioSample {
            data: AudioData::Pcm(tensor),
            sample_rate: spec.sample_rate as usize,
        })
        .await;

    // Wait for playback to complete
    let duration_ms = (duration_secs * 1000.0) as u64 + 500;
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;

    log::info!("Playback complete");
    Ok(())
}
