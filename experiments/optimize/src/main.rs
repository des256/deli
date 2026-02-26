use {base::*, inference::*, std::time::Duration};

const SENTENCE: &str = "After days in the shade, sunlight now cut through the rough surface, sending shimmering rays dancing across the rocky bed of the river, and illuminating the patches of bright green algae that carpeted the rocks of deeper, slower pools.";
const SAMPLE_RATE: u32 = 24000;

const POCKET_VOICE_PATH: &str = "data/pocket/voices/hannah.bin";

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    let inference = Inference::new()?;

    let mut tts = inference.use_pocket(&onnx::Executor::Cpu, &POCKET_VOICE_PATH)?;

    tts.send(SENTENCE.to_string()).await?;

    log_info!("Running...");
    let mut samples: Vec<i16> = Vec::new();
    while let Ok(Some(sample)) = tokio::time::timeout(Duration::from_secs(1), tts.recv()).await {
        samples.extend_from_slice(&sample);
    }
    log_info!("Done.");

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = match hound::WavWriter::create("verify.wav", spec) {
        Ok(writer) => writer,
        Err(e) => return Err(InferError::Runtime(e.to_string())),
    };
    for &s in &samples {
        writer
            .write_sample(s)
            .map_err(|e| InferError::Runtime(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| InferError::Runtime(e.to_string()))?;
    Ok(())
}
