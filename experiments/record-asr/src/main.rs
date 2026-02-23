use {audio::AudioIn, base::*, futures_util::{SinkExt, StreamExt}, inference::{Inference, asr::Transcription}, std::path::PathBuf};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Parse CLI arguments
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/sherpa".to_string());

    let dir = PathBuf::from(&model_dir);
    let encoder_path = dir.join("encoder-epoch-99-avg-1.int8.onnx");
    let decoder_path = dir.join("decoder-epoch-99-avg-1.int8.onnx");
    let joiner_path = dir.join("joiner-epoch-99-avg-1.int8.onnx");
    let tokens_path = dir.join("tokens.txt");

    log_info!("Streaming ASR");
    log_info!("Model directory: {}", model_dir);

    // Load streaming ASR model
    let inference = Inference::cpu()?;
    let mut asr = inference.use_streaming_asr(
        &encoder_path,
        &decoder_path,
        &joiner_path,
        &tokens_path,
    )?;
    log_info!("Model loaded");

    // Create audio input
    let mut audioin = AudioIn::open(None).await;

    // Track how much text we've already printed so we only print new words
    let mut printed_len = 0;

    // Main loop
    loop {
        tokio::select! {
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

            result = asr.next() => {
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

            _ = tokio::signal::ctrl_c() => {
                log_info!("Shutting down...");

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

                break;
            }
        }
    }

    Ok(())
}
