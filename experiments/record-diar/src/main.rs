use {
    audio::{AudioData, AudioIn},
    base::*,
    inference::{Inference, diar::parakeet::SpeakerSegment},
    std::io::Write,
    std::path::PathBuf,
    tokio::sync::mpsc,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/parakeet".to_string());

    let dir = PathBuf::from(&model_dir);
    let model_path = dir.join("diar_streaming_sortformer_4spk-v2.1.onnx");

    log_info!("Streaming Speaker Diarization (Sortformer v2)");
    log_info!("Model: {}", model_path.display());

    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;
    let diar = inference.use_parakeet_diar(&model_path)?;
    log_info!("Model loaded — listening...");

    // Channels: audio → processing thread, segments → main loop
    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(16);
    let (seg_tx, mut seg_rx) = mpsc::channel::<Vec<SpeakerSegment>>(16);

    // Diarization runs on a dedicated thread so it never blocks audio capture
    std::thread::spawn(move || {
        let mut diar = diar;
        while let Some(chunk) = audio_rx.blocking_recv() {
            match diar.diarize_chunk(&chunk) {
                Ok(segments) => {
                    if seg_tx.blocking_send(segments).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Diarization error: {}", e);
                }
            }
        }
    });

    let mut audioin = AudioIn::open(None).await;
    let mut audio_buf: Vec<f32> = Vec::new();
    let chunk_size: usize = 16000; // ~1 second at 16kHz

    loop {
        tokio::select! {
            biased;

            Some(segments) = seg_rx.recv() => {
                for seg in &segments {
                    print!(
                        "[spk{}] {:.2}s - {:.2}s  ",
                        seg.speaker_id, seg.start, seg.end
                    );
                }
                if !segments.is_empty() {
                    println!();
                    std::io::stdout().flush().ok();
                }
            }

            chunk = audioin.capture() => {
                match chunk {
                    Ok(sample) => {
                        let AudioData::Pcm(ref tensor) = sample.data;
                        for &s in &tensor.data {
                            audio_buf.push(s as f32 / 32768.0);
                        }

                        while audio_buf.len() >= chunk_size {
                            let chunk: Vec<f32> = audio_buf.drain(..chunk_size).collect();
                            if audio_tx.send(chunk).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(error) => {
                        log_error!("Audio capture error: {}", error);
                        break;
                    }
                }
            }

            _ = tokio::signal::ctrl_c() => {
                log_info!("Shutting down...");
                // Send remaining audio
                if !audio_buf.is_empty() {
                    let _ = audio_tx.send(audio_buf.split_off(0)).await;
                }
                drop(audio_tx);
                // Drain remaining results
                while let Some(segments) = seg_rx.recv().await {
                    for seg in &segments {
                        println!("[spk{}] {:.2}s - {:.2}s", seg.speaker_id, seg.start, seg.end);
                    }
                }
                break;
            }
        }
    }

    Ok(())
}
