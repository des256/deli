use {
    audio::AudioIn,
    base::*,
    inference::{Inference, asr::Transcription},
    std::io::Write,
    std::path::PathBuf,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Parse CLI arguments
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/parakeet".to_string());

    let dir = PathBuf::from(&model_dir);
    let encoder_path = dir.join("encoder.int8.onnx");
    let decoder_joint_path = dir.join("decoder_joint.int8.onnx");
    let vocab_path = dir.join("tokenizer.model");

    log_info!("Streaming ASR (Parakeet)");
    log_info!("Model directory: {}", model_dir);

    // Load streaming ASR model
    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;
    let mut asr = inference.use_parakeet_asr(&encoder_path, &decoder_joint_path, &vocab_path)?;
    log_info!("Model loaded");

    // Create audio input
    let mut audioin = AudioIn::open(None).await;

    // Track how much text we've already printed so we only print new words
    let mut printed_len = 0;

    // Main loop — biased select ensures ASR results are always consumed first,
    // preventing audio capture from starving the transcription output.
    loop {
        tokio::select! {
            biased;

            result = asr.recv() => {
                match result {
                    Some(Ok(Transcription::Partial { ref text, .. })) => {
                        if text.len() > printed_len {
                            print!("{}", &text[printed_len..]);
                            std::io::stdout().flush().ok();
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
                    Some(Ok(Transcription::Cancelled)) => {}
                    Some(Err(e)) => {
                        log_error!("Transcription error: {}", e);
                    }
                    None => {
                        log_info!("ASR stream ended");
                        break;
                    }
                }
            }

            chunk = audioin.capture() => {
                match chunk {
                    Ok(sample) => {
                        asr.send(sample).await?;
                    }
                    Err(error) => {
                        log_error!("Audio capture error: {}", error);
                        break;
                    }
                }
            }

            _ = tokio::signal::ctrl_c() => {
                log_info!("Shutting down...");

                asr.close().await?;

                while let Some(result) = asr.recv().await {
                    match result {
                        Ok(Transcription::Partial { ref text, .. }) => {
                            if text.len() > printed_len {
                                print!("{}", &text[printed_len..]);
                                std::io::stdout().flush().ok();
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

                break;
            }
        }
    }

    Ok(())
}
