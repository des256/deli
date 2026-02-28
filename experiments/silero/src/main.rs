use {audio::*, base::*, inference::*};

const SAMPLE_RATE: usize = 16000;
const FRAME_SIZE: usize = 512;
const SPEECH_THRESHOLD: f32 = 0.5;

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();
    let mut audioin = create_audioin(None).await;
    let inference = Inference::new()?;
    let mut vad = inference.use_silero(&onnx::Executor::Cpu, SAMPLE_RATE)?;
    let mut buffer: Vec<i16> = Vec::new();
    let mut is_speech = false;
    while let Some(sample) = audioin.recv().await {
        buffer.extend_from_slice(&sample);
        while buffer.len() >= FRAME_SIZE {
            let slice = &buffer[..FRAME_SIZE];
            let probability = vad.process(slice)?;
            let now_speech = probability >= SPEECH_THRESHOLD;
            if now_speech != is_speech {
                if now_speech {
                    log_info!("speech start");
                } else {
                    log_info!("speech end");
                }
                is_speech = now_speech;
            }
            buffer.drain(..FRAME_SIZE);
        }
    }

    Ok(())
}
