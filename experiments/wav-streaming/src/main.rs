use {
    audio::{AudioData, AudioSample},
    base::*,
    futures_util::{FutureExt, SinkExt, StreamExt},
    inference::{Inference, asr::Transcription},
    std::{path::PathBuf, time::Instant},
};

const SAMPLE_RATE: usize = 16000;
const CHUNK_DURATION_MS: usize = 100;
const CHUNK_SAMPLES: usize = SAMPLE_RATE * CHUNK_DURATION_MS / 1000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav-file> [model-dir]", args[0]);
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("data/sherpa");
    let dir = PathBuf::from(model_dir);

    // Read WAV file
    let reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    log_info!("WAV: {} Hz, {} ch, {} bit", spec.sample_rate, spec.channels, spec.bits_per_sample);

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
    let samples = if spec.sample_rate != SAMPLE_RATE as u32 {
        log_info!("Resampling from {} Hz to {} Hz", spec.sample_rate, SAMPLE_RATE);
        resample(&mono_samples, spec.sample_rate, SAMPLE_RATE as u32)
    } else {
        mono_samples
    };

    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    log_info!("Audio: {:.1}s, {} samples", duration_secs, samples.len());

    // Load streaming ASR
    let inference = Inference::cpu()?;
    let mut asr = inference.use_streaming_asr(
        dir.join("encoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
        dir.join("decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
        dir.join("joiner-epoch-99-avg-1-chunk-16-left-128.onnx"),
        dir.join("tokens.txt"),
    )?;
    log_info!("Model loaded");

    // Feed audio in chunks, simulating a real-time stream
    let start = Instant::now();
    let mut printed_len = 0;

    for chunk in samples.chunks(CHUNK_SAMPLES) {
        let tensor = Tensor::new(vec![chunk.len()], chunk.to_vec())?;
        let sample = AudioSample {
            data: AudioData::Pcm(tensor),
            sample_rate: SAMPLE_RATE,
        };
        asr.send(sample).await?;

        // Drain any ready transcriptions
        while let Some(result) = asr.next().now_or_never() {
            match result {
                Some(Ok(Transcription::Partial { ref text, .. })) => {
                    if text.len() > printed_len {
                        print!("{}", &text[printed_len..]);
                        printed_len = text.len();
                    }
                }
                Some(Ok(Transcription::Final { ref text, .. })) => {
                    if text.len() > printed_len {
                        print!("{}", &text[printed_len..]);
                    }
                    println!();
                    printed_len = 0;
                }
                _ => break,
            }
        }
    }

    // Close and drain remaining
    asr.close().await?;
    while let Some(result) = asr.next().await {
        match result {
            Ok(Transcription::Partial { ref text, .. }) => {
                if text.len() > printed_len {
                    print!("{}", &text[printed_len..]);
                    printed_len = text.len();
                }
            }
            Ok(Transcription::Final { ref text, .. }) => {
                if text.len() > printed_len {
                    print!("{}", &text[printed_len..]);
                }
                println!();
            }
            Ok(Transcription::Cancelled) => {}
            Err(e) => {
                log_error!("Transcription error: {}", e);
            }
        }
    }

    let elapsed = start.elapsed();
    log_info!("Transcription took {:.2?} for {:.1}s audio ({:.1}x realtime)",
        elapsed, duration_secs, duration_secs / elapsed.as_secs_f64());

    Ok(())
}

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
