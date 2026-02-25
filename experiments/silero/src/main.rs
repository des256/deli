use {
    audio::{AudioData, AudioIn, AudioInConfig},
    base::*,
    inference::Inference,
};

const VAD_FRAME_SIZE: usize = 512;
const SPEECH_THRESHOLD: f32 = 0.5;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    log_info!("Silero VAD");

    // Load VAD model
    let inference = Inference::cpu()?;
    let mut vad = inference.use_silero_vad()?;
    log_info!("Model loaded");

    // Open audio input at 16kHz with 512-sample chunks (matching VAD frame size)
    let config = AudioInConfig {
        device_name: None,
        sample_rate: 16000,
        chunk_size: VAD_FRAME_SIZE,
    };
    let mut audioin = AudioIn::open(Some(config)).await;
    log_info!("Listening... (Ctrl+C to stop)");

    let mut is_speech = false;

    loop {
        tokio::select! {
            chunk = audioin.capture() => {
                match chunk {
                    Ok(sample) => {
                        let AudioData::Pcm(ref tensor) = sample.data;
                        let pcm_i16 = &tensor.data;

                        // Convert i16 PCM to f32 normalized [-1, 1]
                        let frame: Vec<f32> = pcm_i16.iter()
                            .map(|&s| s as f32 / 32768.0)
                            .collect();

                        let prob = vad.process(&frame)?;

                        let now_speech = prob >= SPEECH_THRESHOLD;
                        if now_speech != is_speech {
                            if now_speech {
                                println!(">> SPEECH START (p={:.3})", prob);
                            } else {
                                println!("<< SPEECH END   (p={:.3})", prob);
                            }
                            is_speech = now_speech;
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
                break;
            }
        }
    }

    Ok(())
}
