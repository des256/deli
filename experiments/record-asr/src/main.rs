use deli_audio::{AudioIn, AudioSample};
use deli_base::{log, Tensor};
use deli_infer::Inference;
use futures_util::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinHandle;

const SAMPLE_RATE: u32 = 16000;
const CHUNK_FRAMES: u32 = 1600;  // 100ms chunks
const WINDOW_SAMPLES: usize = 48000;  // 3 seconds at 16kHz
const MIN_PARTIAL_SAMPLES: usize = 8000;  // 0.5 seconds

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

    // Load Whisper model
    let recognizer = inference.use_speech_recognizer(&model_path, &tokenizer_path, &config_path)?;
    log::info!("Whisper model loaded");

    // Create audio input
    let mut audio_in = AudioIn::new(None, SAMPLE_RATE, CHUNK_FRAMES);

    // Verify audio capture with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(5), audio_in.next()).await {
        Ok(Some(_)) => {
            // First chunk received successfully, discard it
            log::info!("Audio capture started");
        }
        Ok(None) => {
            eprintln!("Audio capture ended unexpectedly.");
            eprintln!("Check that PulseAudio is running and a microphone is connected.");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("Audio capture timeout. Check that PulseAudio is running and a microphone is connected.");
            std::process::exit(1);
        }
    }

    // Wrap recognizer in Arc for sharing with spawned tasks
    let recognizer = Arc::new(recognizer);
    let mut buffer: Vec<i16> = Vec::new();
    let mut transcription_task: Option<JoinHandle<Result<String, Box<dyn std::error::Error + Send>>>> = None;

    // Main loop with Ctrl+C handling
    loop {
        tokio::select! {
            // Receive audio chunks
            chunk = audio_in.next() => {
                match chunk {
                    Some(AudioSample::Pcm(tensor)) => {
                        buffer.extend(&tensor.data);
                    }
                    None => {
                        log::info!("Audio capture ended");
                        break;
                    }
                }

                // Check if previous transcription completed (non-blocking)
                if let Some(ref task) = transcription_task {
                    if task.is_finished() {
                        let task = transcription_task.take().unwrap();
                        match task.await {
                            Ok(Ok(text)) => {
                                let text = text.trim();
                                if !text.is_empty() {
                                    println!("{}", text);
                                } else {
                                    log::debug!("No speech detected in window");
                                }
                            }
                            Ok(Err(e)) => {
                                log::error!("Transcription error: {}", e);
                            }
                            Err(e) => {
                                log::error!("Task join error: {}", e);
                            }
                        }
                    }
                }

                // When buffer is full and no task running, start new transcription
                if buffer.len() >= WINDOW_SAMPLES && transcription_task.is_none() {
                    let samples = std::mem::take(&mut buffer);
                    let recognizer_clone = Arc::clone(&recognizer);

                    transcription_task = Some(tokio::spawn(async move {
                        let start = Instant::now();
                        let tensor = Tensor::new(vec![samples.len()], samples)
                            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;
                        let result = recognizer_clone.transcribe(&tensor, SAMPLE_RATE).await;
                        let elapsed = start.elapsed();
                        log::debug!("Transcription took {:?}", elapsed);
                        result.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)
                    }));
                }
            }

            // Handle Ctrl+C
            _ = tokio::signal::ctrl_c() => {
                log::info!("Shutting down...");

                // Wait for pending transcription if any
                if let Some(task) = transcription_task.take() {
                    match task.await {
                        Ok(Ok(text)) => {
                            let text = text.trim();
                            if !text.is_empty() {
                                println!("{}", text);
                            }
                        }
                        Ok(Err(e)) => {
                            log::error!("Transcription error: {}", e);
                        }
                        Err(e) => {
                            log::error!("Task join error: {}", e);
                        }
                    }
                }

                // Transcribe partial buffer if it has meaningful content
                if buffer.len() >= MIN_PARTIAL_SAMPLES {
                    log::info!("Transcribing partial buffer before shutdown...");
                    let tensor = Tensor::new(vec![buffer.len()], buffer)?;
                    match recognizer.transcribe(&tensor, SAMPLE_RATE).await {
                        Ok(text) => {
                            let text = text.trim();
                            if !text.is_empty() {
                                println!("{}", text);
                            }
                        }
                        Err(e) => {
                            log::error!("Partial buffer transcription error: {}", e);
                        }
                    }
                }

                break;
            }
        }
    }

    Ok(())
}
