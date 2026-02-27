use {audio::*, inference::*, onnx::*, std::sync::Arc};

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();
    let inference = Inference::new()?;
    let mut audioin_listener = create_audioin(Some(AudioInConfig {
        sample_rate: 16000,
        chunk_size: 8000,
        boost: 4,
        ..Default::default()
    }))
    .await;
    let (asr_handle, mut asr_listener) = inference.use_parakeet::<()>(&Executor::Cpu)?;
    let asr_handle = Arc::new(asr_handle);
    tokio::spawn({
        let asr_handle = Arc::clone(&asr_handle);
        async move {
            while let Some(audio) = audioin_listener.recv().await {
                if let Err(error) = asr_handle.send(AsrInput { payload: (), audio }) {
                    panic!("unable to send audio to ASR: {}", error);
                }
            }
        }
    });

    while let Some(output) = asr_listener.recv().await {
        println!("{}", output.text);
    }

    Ok(())
}
