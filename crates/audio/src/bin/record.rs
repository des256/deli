use {audio::*, base::*, hound};

const SAMPLE_RATE: usize = 16000;
const DURATION_SECONDS: usize = 5;
const TOTAL_SAMPLES: usize = SAMPLE_RATE * DURATION_SECONDS;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_stdout_logger();

    let mut audioin = AudioIn::open(None).await;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        log_fatal!("Usage: {} <output-wav-path>", args[0]);
    }
    let output_path = &args[1];

    log_info!("Recording 5 seconds...");
    let mut all_samples: Vec<i16> = Vec::new();
    while all_samples.len() < TOTAL_SAMPLES {
        match audioin.capture().await {
            Ok(AudioSample { data, .. }) => match data {
                AudioData::Pcm(tensor) => {
                    all_samples.extend(&tensor.data);
                }
            },
            Err(error) => {
                log_error!("Audio capture ended unexpectedly: {}", error);
                break;
            }
        }
    }
    log_info!("Recorded {} samples", all_samples.len());

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for sample in &all_samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    log_info!("Saved to {}", output_path);
    Ok(())
}
