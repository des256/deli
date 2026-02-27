use {audio::*, base::*, hound};

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

    // open audio output
    let (audioout_handle, mut audioout_listener) = create_audioout(Some(AudioOutConfig {
        sample_rate: spec.sample_rate as usize,
        ..Default::default()
    }));

    // send chunk to audio output
    if let Err(error) = audioout_handle.send(AudioOutChunk {
        id: 0,
        data: mono_samples,
    }) {
        log_error!("AudioOut send failed: {}", error);
    }

    // wait for playback to complete
    while let Some(status) = audioout_listener.recv().await {
        match status {
            AudioOutStatus::Started(id) => {
                log_info!("playback started: {}", id);
            }
            AudioOutStatus::Finished { id, index } => {
                log_info!("playback finished: {} ({} samples)", id, index);
                break;
            }
            AudioOutStatus::Canceled { id, index } => {
                log_info!("playback canceled: {} ({} samples)", id, index);
                break;
            }
        }
    }

    Ok(())
}
