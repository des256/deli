use {base::*, inference::*, onnx::*};

const SAMPLE_RATE: usize = 16000;
const CHUNK_SIZE: usize = 16000;

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav-file> [model-dir]", args[0]);
        std::process::exit(1);
    }

    let wav_path = &args[1];

    // Read WAV file
    let reader =
        hound::WavReader::open(wav_path).map_err(|e| InferError::Runtime(e.to_string()))?;
    let spec = reader.spec();
    log_info!(
        "WAV: {} Hz, {} ch, {} bit",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample
    );

    // Read all samples as i32 (hound normalizes per bit depth) then scale to i16 range
    let all_samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| (v as f32 / max_val * i16::MAX as f32) as i16))
                .collect::<Result<_, _>>()
                .map_err(|e| InferError::Runtime(e.to_string()))?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map(|v| (v * i16::MAX as f32) as i16))
            .collect::<Result<_, _>>()
            .map_err(|e| InferError::Runtime(e.to_string()))?,
    };

    // Convert to mono if stereo
    let mono_samples: Vec<i16> = if spec.channels > 1 {
        all_samples
            .chunks(spec.channels as usize)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / frame.len() as i32) as i16
            })
            .collect()
    } else {
        all_samples
    };

    // Resample to 16kHz if needed
    let samples = if spec.sample_rate != SAMPLE_RATE as u32 {
        log_info!(
            "Resampling from {} Hz to {} Hz",
            spec.sample_rate,
            SAMPLE_RATE
        );
        resample(&mono_samples, spec.sample_rate, SAMPLE_RATE as u32)
    } else {
        mono_samples
    };

    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    log_info!("Audio: {:.1}s, {} samples", duration_secs, samples.len());

    // Initialize inference
    let inference = Inference::new()?;

    // load ASR model
    let (asr_handle, mut asr_listener) = inference.use_parakeet::<()>(&Executor::Cpu)?;

    // spawn off task to feed ASR
    tokio::spawn({
        async move {
            for chunk in samples.chunks(CHUNK_SIZE) {
                if let Err(error) = asr_handle.send(AsrInput {
                    payload: (),
                    audio: chunk.to_vec(),
                }) {
                    log_error!("file -> asr: {}", error);
                    break;
                }
            }
            if let Err(error) = asr_handle.send(AsrInput {
                payload: (),
                audio: vec![0i16; 2 * CHUNK_SIZE],
            }) {
                log_error!("file -> asr: {}", error);
            }
        }
    });

    // Collect transcription results
    while let Some(output) = asr_listener.recv().await {
        println!("{}", output.text);
    }

    Ok(())
}

/// Linear interpolation resampling from `from_rate` to `to_rate`.
fn resample(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < samples.len() {
            let a = samples[idx] as f64;
            let b = samples[idx + 1] as f64;
            (a + (b - a) * frac) as i16
        } else {
            samples[idx]
        };

        output.push(sample);
    }

    output
}
