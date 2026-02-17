use deli_base::{log, Tensor};
use deli_infer::Inference;
use std::path::PathBuf;
use std::time::Instant;

const WHISPER_SAMPLE_RATE: u32 = 16000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav-file> [model-dir]", args[0]);
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let model_dir = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("models/whisper-tiny.en");

    // Read WAV file
    let reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    log::info!(
        "WAV: {} Hz, {} ch, {} bit",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample
    );

    // Read all samples as i16
    let all_samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => reader.into_samples::<i16>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map(|v| (v * i16::MAX as f32) as i16))
            .collect::<Result<_, _>>()?,
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
    let samples = if spec.sample_rate != WHISPER_SAMPLE_RATE {
        log::info!(
            "Resampling from {} Hz to {} Hz",
            spec.sample_rate,
            WHISPER_SAMPLE_RATE
        );
        resample(&mono_samples, spec.sample_rate, WHISPER_SAMPLE_RATE)
    } else {
        mono_samples
    };

    let duration_secs = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    log::info!("Audio: {:.1}s, {} samples", duration_secs, samples.len());

    // Validate model directory
    let model_path = PathBuf::from(model_dir).join("model.safetensors");
    let tokenizer_path = PathBuf::from(model_dir).join("tokenizer.json");
    let config_path = PathBuf::from(model_dir).join("config.json");

    if !model_path.exists() || !tokenizer_path.exists() || !config_path.exists() {
        eprintln!("Model directory incomplete. Expected files:");
        eprintln!("  - model.safetensors");
        eprintln!("  - tokenizer.json");
        eprintln!("  - config.json");
        eprintln!("In directory: {}", model_dir);
        std::process::exit(1);
    }

    // Initialize CUDA and load model
    log::info!("Initializing CUDA...");
    let inference = Inference::cuda(0)?;
    log::info!("CUDA initialized");

    let recognizer =
        inference.use_speech_recognizer(&model_path, &tokenizer_path, &config_path)?;
    log::info!("Whisper model loaded");

    // Transcribe
    let tensor = Tensor::new(vec![samples.len()], samples)?;
    let start = Instant::now();
    let text = recognizer.transcribe(&tensor, WHISPER_SAMPLE_RATE).await?;
    let elapsed = start.elapsed();

    let text = text.trim();
    if !text.is_empty() {
        println!("{}", text);
    } else {
        log::info!("No speech detected");
    }

    log::info!("Transcription took {:?}", elapsed);
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
