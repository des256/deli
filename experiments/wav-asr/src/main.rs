use {audio::{AudioData, AudioSample}, base::*, futures_util::{SinkExt, StreamExt}, inference::{Inference, asr::Transcription}, std::path::PathBuf};
use std::time::Instant;

const WHISPER_SAMPLE_RATE: usize = 16000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav-file> [model-dir]", args[0]);
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let model_dir = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("data/whisper/tiny.en");

    // Read WAV file
    let reader = hound::WavReader::open(wav_path)?;
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
                .collect::<Result<_, _>>()?
        }
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
    let samples = if spec.sample_rate != WHISPER_SAMPLE_RATE as u32 {
        log_info!(
            "Resampling from {} Hz to {} Hz",
            spec.sample_rate,
            WHISPER_SAMPLE_RATE
        );
        resample(&mono_samples, spec.sample_rate, WHISPER_SAMPLE_RATE as u32)
    } else {
        mono_samples
    };

    let duration_secs = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    log_info!("Audio: {:.1}s, {} samples", duration_secs, samples.len());

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
    log_info!("Initializing CUDA...");
    let inference = Inference::cuda(0)?;
    log_info!("CUDA initialized");

    let num_samples = samples.len();
    let mut whisper = inference
        .use_whisper(&model_path, &tokenizer_path, &config_path)?
        .with_window_samples(num_samples); // Process entire file as one window
    log_info!("Whisper model loaded");

    // Send audio and close
    let tensor = Tensor::new(vec![num_samples], samples)?;
    let sample = AudioSample {
        data: AudioData::Pcm(tensor),
        sample_rate: WHISPER_SAMPLE_RATE,
    };

    let start = Instant::now();
    whisper.send(sample).await?;
    whisper.close().await?;

    // Read transcription
    if let Some(result) = whisper.next().await {
        let transcription = result?;
        match transcription {
            Transcription::Final { text, .. } => {
                let text = text.trim();
                if !text.is_empty() {
                    println!("{}", text);
                } else {
                    log_info!("No speech detected");
                }
            }
            _ => log_info!("No speech detected"),
        }
    }

    let elapsed = start.elapsed();
    log_info!("Transcription took {:?}", elapsed);
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
