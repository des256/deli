use {
    crate::{
        error::Result,
        tts::pocket::PocketTts,
        Inference,
    },
    audio::AudioData,
    futures_sink::Sink,
    futures_core::Stream,
};

#[test]
fn test_pocket_tts_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<PocketTts>();
}

#[test]
fn test_implements_sink_and_stream() {
    fn assert_sink<T: Sink<String>>() {}
    fn assert_stream<T: Stream>() {}
    assert_sink::<PocketTts>();
    assert_stream::<PocketTts>();
}

#[test]
#[ignore] // Requires ONNX models
fn test_pocket_tts_integration() -> Result<()> {
    // Create inference context
    let inference = Inference::cpu()?;

    // Create PocketTts via factory method
    let mut tts = inference.use_pocket_tts(
        "data/pocket/text_conditioner.onnx",
        "data/pocket/flow_lm_main_int8.onnx",
        "data/pocket/flow_lm_flow_int8.onnx",
        "data/pocket/mimi_encoder.onnx",
        "data/pocket/mimi_decoder_int8.onnx",
        "data/pocket/tokenizer.json",
        "data/pocket/voice.wav",
    )?;

    // Generate audio
    use futures_util::SinkExt;
    use futures_util::StreamExt;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Send text (convert &str to String)
        tts.send("Hello world".to_string()).await.unwrap();
        tts.close().await.unwrap();

        // Receive audio
        if let Some(result) = tts.next().await {
            let sample = result.unwrap();

            // Verify sample rate
            assert_eq!(sample.sample_rate, 24000);

            // Verify audio has samples
            match sample.data {
                AudioData::Pcm(tensor) => {
                    let len = tensor.shape[0];
                    assert!(len > 0, "Audio should have samples");
                    println!("Generated {} samples", len);

                    // Verify non-zero variance (not silence)
                    let data = &tensor.data;
                    let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / len as f64;
                    let variance: f64 = data
                        .iter()
                        .map(|&x| {
                            let diff = x as f64 - mean;
                            diff * diff
                        })
                        .sum::<f64>()
                        / len as f64;
                    assert!(variance > 0.0, "Audio should not be silence");

                    // Verify duration is reasonable (1-10s for "Hello world")
                    let duration_sec = len as f32 / 24000.0;
                    assert!(duration_sec >= 1.0 && duration_sec <= 10.0,
                            "Duration {} s is unreasonable for 'Hello world'", duration_sec);

                    // Write to file for manual verification
                    std::fs::write("/tmp/pocket_tts_output.raw", unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<i16>(),
                        )
                    })
                    .unwrap();
                    println!("✓ Output written to /tmp/pocket_tts_output.raw");
                }
                _ => panic!("Expected PCM audio data"),
            }
        } else {
            panic!("Expected audio sample");
        }
    });

    Ok(())
}

#[test]
#[ignore] // Requires ONNX models
fn test_pocket_tts_multiple_utterances() -> Result<()> {
    // Create inference context
    let inference = Inference::cpu()?;

    // Create PocketTts via factory method
    let mut tts = inference.use_pocket_tts(
        "data/pocket/text_conditioner.onnx",
        "data/pocket/flow_lm_main_int8.onnx",
        "data/pocket/flow_lm_flow_int8.onnx",
        "data/pocket/mimi_encoder.onnx",
        "data/pocket/mimi_decoder_int8.onnx",
        "data/pocket/tokenizer.json",
        "data/pocket/voice.wav",
    )?;

    // Generate multiple utterances
    use futures_util::SinkExt;
    use futures_util::StreamExt;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Send multiple texts (convert &str to String)
        tts.send("First utterance".to_string()).await.unwrap();
        tts.send("Second utterance".to_string()).await.unwrap();
        tts.send("Third utterance".to_string()).await.unwrap();
        tts.close().await.unwrap();

        // Receive all three
        let mut count = 0;
        while let Some(result) = tts.next().await {
            let sample = result.unwrap();
            match sample.data {
                AudioData::Pcm(tensor) => {
                    let len = tensor.shape[0];
                    assert!(len > 0, "Audio {} should have samples", count + 1);
                    count += 1;
                }
                _ => panic!("Expected PCM audio data"),
            }
        }

        assert_eq!(count, 3, "Should generate 3 audio samples");
        println!("✓ Generated 3 utterances successfully");
    });

    Ok(())
}
