use {
    crate::{Inference, error::Result, tts::pocket::PocketTts},
    audio::AudioData,
    futures_core::Stream,
    futures_sink::Sink,
    std::path::Path,
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
    let mut tts = inference.use_pocket_tts(Path::new("data/pocket/voices/hannah.bin"))?;

    // Generate audio (streaming)
    use futures_util::SinkExt;
    use futures_util::StreamExt;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Send text
        tts.send("Hello world".to_string()).await.unwrap();
        tts.close().await.unwrap();

        // Collect all streamed chunks
        let mut all_data: Vec<i16> = Vec::new();
        let mut chunk_count = 0usize;
        while let Some(result) = tts.next().await {
            let sample = result.unwrap();
            assert_eq!(sample.sample_rate, 24000);
            match sample.data {
                AudioData::Pcm(tensor) => {
                    all_data.extend_from_slice(&tensor.data);
                    chunk_count += 1;
                }
            }
        }

        // Verify audio has samples (last chunk is empty end-of-utterance marker)
        assert!(!all_data.is_empty(), "Audio should have samples");
        assert!(
            chunk_count > 1,
            "Should stream multiple chunks, got {}",
            chunk_count
        );
        println!(
            "Generated {} samples in {} chunks",
            all_data.len(),
            chunk_count
        );

        // Verify non-zero variance (not silence)
        let len = all_data.len();
        let mean: f64 = all_data.iter().map(|&x| x as f64).sum::<f64>() / len as f64;
        let variance: f64 = all_data
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
        assert!(
            duration_sec >= 1.0 && duration_sec <= 10.0,
            "Duration {} s is unreasonable for 'Hello world'",
            duration_sec
        );

        // Write to file for manual verification
        std::fs::write("/tmp/pocket_tts_output.raw", unsafe {
            std::slice::from_raw_parts(
                all_data.as_ptr() as *const u8,
                all_data.len() * std::mem::size_of::<i16>(),
            )
        })
        .unwrap();
        println!("Output written to /tmp/pocket_tts_output.raw");
    });

    Ok(())
}

#[test]
#[ignore] // Requires ONNX models
fn test_pocket_tts_multiple_utterances() -> Result<()> {
    // Create inference context
    let inference = Inference::cpu()?;

    // Create PocketTts via factory method
    let mut tts = inference.use_pocket_tts(Path::new("data/pocket/voices/hannah.bin"))?;

    // Generate multiple utterances (streaming)
    use futures_util::SinkExt;
    use futures_util::StreamExt;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Send multiple texts
        tts.send("First utterance".to_string()).await.unwrap();
        tts.send("Second utterance".to_string()).await.unwrap();
        tts.send("Third utterance".to_string()).await.unwrap();
        tts.close().await.unwrap();

        // Receive all streamed chunks; count utterances via empty markers
        let mut utterance_count = 0;
        let mut current_utterance_samples = 0;
        while let Some(result) = tts.next().await {
            let sample = result.unwrap();
            match sample.data {
                AudioData::Pcm(tensor) => {
                    if tensor.data.is_empty() {
                        // End-of-utterance marker
                        assert!(
                            current_utterance_samples > 0,
                            "Utterance {} should have samples",
                            utterance_count + 1
                        );
                        utterance_count += 1;
                        current_utterance_samples = 0;
                    } else {
                        current_utterance_samples += tensor.data.len();
                    }
                }
            }
        }

        assert_eq!(utterance_count, 3, "Should generate 3 utterances");
        println!("Generated 3 utterances successfully (streaming)");
    });

    Ok(())
}
