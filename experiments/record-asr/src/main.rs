use deli_audio::AudioIn;
use deli_base::log;
use deli_infer::asr::Transcription;
use deli_infer::Inference;
use futures_util::{SinkExt, StreamExt};
use std::path::PathBuf;

const SAMPLE_RATE: usize = 16000;
const CHUNK_FRAMES: usize = 1600; // 100ms chunks
const WINDOW_SAMPLES: usize = 48000; // 3 seconds at 16kHz

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    // Parse CLI arguments
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/whisper-tiny.en".to_string());

    let model_path = PathBuf::from(&model_dir).join("model.safetensors");
    let tokenizer_path = PathBuf::from(&model_dir).join("tokenizer.json");
    let config_path = PathBuf::from(&model_dir).join("config.json");

    // Validate model directory
    if !model_path.exists() || !tokenizer_path.exists() || !config_path.exists() {
        eprintln!("Model directory incomplete. Expected files:");
        eprintln!("  - model.safetensors");
        eprintln!("  - tokenizer.json");
        eprintln!("  - config.json");
        eprintln!("In directory: {}", model_dir);
        std::process::exit(1);
    }

    log::info!("Speech Recognition Experiment");
    log::info!("Model directory: {}", model_dir);

    // Initialize CUDA
    log::info!("Initializing CUDA...");
    let inference = Inference::cuda(0)?;
    log::info!("CUDA initialized");

    // Load Whisper model with 3-second window
    let mut whisper = inference
        .use_whisper(&model_path, &tokenizer_path, &config_path)?
        .with_window_samples(WINDOW_SAMPLES);
    log::info!("Whisper model loaded");

    // Create audio input
    let mut audio_in = AudioIn::new(None, SAMPLE_RATE, CHUNK_FRAMES);

    // Verify audio capture with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(5), audio_in.next()).await {
        Ok(Some(_)) => {
            log::info!("Audio capture started");
        }
        Ok(None) => {
            eprintln!("Audio capture ended unexpectedly.");
            eprintln!("Check that PulseAudio is running and a microphone is connected.");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!(
                "Audio capture timeout. Check that PulseAudio is running and a microphone is connected."
            );
            std::process::exit(1);
        }
    }

    // Main loop with Ctrl+C handling
    loop {
        tokio::select! {
            // Feed audio chunks into Whisper
            chunk = audio_in.next() => {
                match chunk {
                    Some(sample) => {
                        whisper.send(sample).await?;
                    }
                    None => {
                        log::info!("Audio capture ended");
                        break;
                    }
                }
            }

            // Read transcription results from Whisper
            result = whisper.next() => {
                match result {
                    Some(Ok(Transcription::Final { text, .. })) => {
                        let text = text.trim();
                        if !text.is_empty() {
                            println!("{}", text);
                        } else {
                            log::debug!("No speech detected in window");
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        log::error!("Transcription error: {}", e);
                    }
                    None => {
                        log::info!("Whisper stream ended");
                        break;
                    }
                }
            }

            // Handle Ctrl+C
            _ = tokio::signal::ctrl_c() => {
                log::info!("Shutting down...");

                // Close sink to flush remaining audio
                whisper.close().await?;

                // Drain remaining transcriptions
                while let Some(result) = whisper.next().await {
                    match result {
                        Ok(Transcription::Final { text, .. }) => {
                            let text = text.trim();
                            if !text.is_empty() {
                                println!("{}", text);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Transcription error: {}", e);
                        }
                    }
                }

                break;
            }
        }
    }

    Ok(())
}
