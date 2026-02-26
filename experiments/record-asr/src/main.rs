use {audio::*, inference::*, onnx::*};

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();
    let inference = Inference::new()?;
    let mut audioin = AudioIn::open(None).await;
    let mut asr = inference.use_parakeet(&Executor::Cpu)?;
    tokio::spawn({
        let asr_input_tx = asr.input_tx();
        async move {
            while let Some(audio) = audioin.recv().await {
                if let Err(error) = asr_input_tx.send(AsrInput { audio }).await {
                    panic!("unable to send audio to ASR: {}", error);
                }
            }
        }
    });

    while let Some(output) = asr.recv().await {
        println!("{}", output.text);
    }

    Ok(())
}
