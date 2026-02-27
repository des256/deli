use {base::*, inference::*, std::time::Duration};

const SENTENCE: &str = "The key issue, with rookworst, is that it is a delicious deli meat, made of willing, pork volunteers, slaughtered with love, prepared with care. - Have you had your rookworst today?";
const SAMPLE_RATE: u32 = 24000;

const POCKET_VOICE_PATH: &str = "data/pocket/voices/hannah.bin";

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <output.wav>", args[0]);
        std::process::exit(1);
    }
    let output_path = &args[1];

    let inference = Inference::new()?;

    let epoch = Epoch::new();
    let (tts_handle, mut tts_listener) =
        inference.use_pocket::<()>(&onnx::Executor::Cpu, &POCKET_VOICE_PATH, epoch)?;

    tts_handle
        .send(TtsInput {
            payload: (),
            text: SENTENCE.to_string(),
        })
        .map_err(|e| InferError::Runtime(e.to_string()))?;

    let mut samples: Vec<i16> = Vec::new();
    while let Ok(Some(stamped)) =
        tokio::time::timeout(Duration::from_secs(1), tts_listener.recv()).await
    {
        log_info!("adding {} samples", stamped.inner.data.len());
        samples.extend_from_slice(&stamped.inner.data);
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = match hound::WavWriter::create(output_path, spec) {
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
    println!("Wrote {} samples to {}", samples.len(), output_path);
    Ok(())
}
