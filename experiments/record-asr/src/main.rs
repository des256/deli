use audio::AudioIn;
use base::log;
use futures_util::{SinkExt, StreamExt};
use inference::Inference;
use inference::asr::Transcription;
use std::path::PathBuf;

const WINDOW_SAMPLES: usize = 48000; // 3 seconds at 16kHz

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Parse CLI arguments
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/whisper/tiny.en".to_string());

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
    let mut audioin = AudioIn::open(None).await;

    // Main loop with Ctrl+C handling
    loop {
        tokio::select! {
            // Feed audio chunks into Whisper
            chunk = audioin.capture() => {
                match chunk {
                    Ok(sample) => {
                        whisper.send(sample).await?;
                    }
                    Err(error) => {
                        log::error!("Audio capture error: {}", error);
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
