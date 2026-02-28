use {
    audio::*,
    base::*,
    inference::*,
    std::io::Write,
    tokio::sync::mpsc,
};

const SAMPLE_RATE: usize = 16000;
const CHUNK_SIZE: usize = 16000; // 1 second of audio at 16kHz

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    log_info!("Streaming Speaker Diarization (Sortformer v2)");

    let inference = Inference::new().map_err(|e| InferError::Runtime(e.to_string()))?;
    let diar = inference.use_parakeet_diar(&onnx::Executor::Cpu)?;
    log_info!("Model loaded — listening...");

    let mut audioin_listener = create_audioin(Some(AudioInConfig {
        sample_rate: SAMPLE_RATE,
        chunk_size: CHUNK_SIZE,
        boost: 1,
        ..Default::default()
    }))
    .await;

    // Channels: audio → processing thread, segments → main loop
    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(16);
    let (seg_tx, mut seg_rx) = mpsc::channel::<Vec<diar::parakeet::SpeakerSegment>>(16);

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
                    log_error!("Diarization error: {}", e);
                }
            }
        }
    });

    loop {
        tokio::select! {
            biased;

            Some(segments) = seg_rx.recv() => {
                for seg in &segments {
                    print!(
                        "[spk{}] {:.2}s-{:.2}s  ",
                        seg.speaker_id, seg.start, seg.end
                    );
                }
                if !segments.is_empty() {
                    println!();
                    std::io::stdout().flush().ok();
                }
            }

            Some(audio) = audioin_listener.recv() => {
                // Convert i16 audio to f32 normalized [-1, 1]
                let audio_f32: Vec<f32> = audio.iter().map(|&s| s as f32 / 32768.0).collect();
                if audio_tx.send(audio_f32).await.is_err() {
                    break;
                }
            }

            else => break,
        }
    }

    Ok(())
}
