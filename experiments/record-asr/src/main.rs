use {
    audio::AudioIn,
    inference::{Inference, asr::Transcription},
    std::io::Write,
    std::path::PathBuf,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/parakeet".to_string());

    let dir = PathBuf::from(&model_dir);
    let encoder_path = dir.join("encoder.onnx");
    let decoder_joint_path = dir.join("decoder_joint.onnx");
    let vocab_path = dir.join("tokenizer.model");

    for path in [&encoder_path, &decoder_joint_path, &vocab_path] {
        if !path.exists() {
            eprintln!("Missing: {}", path.display());
            std::process::exit(1);
        }
    }

    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;

    let mut asr = inference.use_parakeet()?;
    let mut audioin = AudioIn::open(None).await;
    let mut printed_len = 0;

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
                    Some(Ok(Transcription::Cancelled)) | None => break,
                    Some(Err(e)) => {
                        eprintln!("\nError: {}", e);
                    }
                }
            }

            chunk = audioin.capture() => {
                match chunk {
                    Ok(sample) => asr.send(sample).await?,
                    Err(e) => {
                        eprintln!("\nAudio error: {}", e);
                        break;
                    }
                }
            }

            _ = tokio::signal::ctrl_c() => {
                asr.close().await?;
                while let Some(result) = asr.recv().await {
                    match result {
                        Ok(Transcription::Partial { ref text, .. })
                        | Ok(Transcription::Final { ref text, .. }) => {
                            if text.len() > printed_len {
                                print!("{}", &text[printed_len..]);
                                std::io::stdout().flush().ok();
                                printed_len = text.len();
                            }
                        }
                        _ => {}
                    }
                }
                println!();
                break;
            }
        }
    }

    Ok(())
}
