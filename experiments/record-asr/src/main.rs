use {audio::*, base::*, inference::*, onnx::*};

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    // initialize inference
    let inference = Inference::new()?;

    // initialize audio input
    let mut audioin = AudioIn::open(None).await;

    // load ASR model
    let mut asr = inference.use_parakeet(&Executor::Cpu)?;

    // spawn off task to feed ASR
    tokio::spawn({
        let asr_audio_tx = asr.audio_tx();
        async move {
            while let Ok(sample) = audioin.capture().await {
                if let Err(error) = asr_audio_tx.send(sample.data).await {
                    log_error!("audio in -> asr: {}", error);
                    break;
                }
            }
        }
    });

    // ASR -> stdout
    while let Some(text) = asr.recv().await {
        println!("{}", text);
    }

    Ok(())
}
