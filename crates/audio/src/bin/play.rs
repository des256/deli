use {audio::*, base::*, hound, std::time::Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_stdout_logger();

    // get parameters
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        log_fatal!("Usage: {} <wav-file>", args[0]);
    }
    let wav_path = &args[1];

    // load WAV file
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    let mono_samples: Vec<i16> = match spec.channels {
        1 => samples,
        2 => {
            log_info!("Downmixing stereo to mono");
            samples
                .chunks_exact(2)
                .map(|pair| ((pair[0] as i32 + pair[1] as i32) / 2) as i16)
                .collect()
        }
        _ => {
            log_fatal!(
                "{} channels not supported (only mono and stereo)",
                spec.channels
            );
        }
    };
    let sample_count = mono_samples.len();
    let duration_secs = sample_count as f64 / spec.sample_rate as f64;

    // open audio output
    let audioout = AudioOutHandle::open(Some(AudioOutConfig {
        sample_rate: spec.sample_rate as usize,
        ..Default::default()
    }))
    .await;

    // play sample
    if let Err(error) = audioout.send(mono_samples) {
        log_error!("AudioOut send failed: {}", error);
    }

    // wait for playback to complete
    let duration_ms = (duration_secs * 1000.0) as u64 + 500;
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;

    Ok(())
}
