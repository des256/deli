use audio::{AudioData, AudioSample};
use futures_core::Stream;
use futures_sink::Sink;
use futures_util::{SinkExt, StreamExt};
use inference::{Inference, Kokoro};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn model_path(relative: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR points to the crate root (crates/deli-infer)
    // Models are at the workspace root: ../../data/kokoro/
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .join("../../data")
        .join(relative)
}

#[test]
fn test_kokoro_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Kokoro>();
}

#[test]
fn test_implements_sink_and_stream() {
    fn assert_sink<T: Sink<String>>() {}
    fn assert_stream<T: Stream<Item = inference::Result<AudioSample>>>() {}
    assert_sink::<Kokoro>();
    assert_stream::<Kokoro>();
}

#[tokio::test]
#[ignore] // Requires model files at data/kokoro/
async fn test_kokoro_integration() {
    let inference = Inference::cuda(0).expect("CUDA device required");

    let mut kokoro = inference
        .use_kokoro(
            model_path("kokoro/kokoro-v1.0.onnx"),
            model_path("kokoro/bf_emma.npy"),
            Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        )
        .expect("Failed to load Kokoro model");

    kokoro.send("Hello world".to_string()).await.unwrap();
    kokoro.close().await.unwrap();

    let sample = kokoro
        .next()
        .await
        .expect("stream should yield an item")
        .unwrap();
    let AudioData::Pcm(tensor) = sample.data;
    assert!(tensor.shape[0] > 0, "Audio tensor should have samples");
    assert!(!tensor.data.is_empty(), "Audio data should not be empty");
    assert_eq!(
        tensor.shape[0],
        tensor.data.len(),
        "Shape should match data length"
    );
    assert_eq!(sample.sample_rate, 24000);

    assert!(
        kokoro.next().await.is_none(),
        "stream should end after close"
    );

    let output_path = "/tmp/kokoro_output.raw";
    let mut file = fs::File::create(output_path).expect("Failed to create output file");
    for s in &tensor.data {
        file.write_all(&s.to_le_bytes())
            .expect("Failed to write sample");
    }
    println!("Audio written to {}", output_path);
    println!("Play with: ffplay -f s16le -ar 24000 -ac 1 {}", output_path);
    println!("Samples: {}", tensor.data.len());
}

#[tokio::test]
#[ignore]
async fn test_kokoro_multiple_utterances() {
    let inference = Inference::cuda(0).expect("CUDA device required");

    let mut kokoro = inference
        .use_kokoro(
            model_path("kokoro/kokoro-v1.0.onnx"),
            model_path("kokoro/bf_emma.npy"),
            Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        )
        .expect("Failed to load Kokoro model");

    let texts = [
        "Hello world",
        "How are you today?",
        "This is a test of the Kokoro TTS system.",
    ];

    for text in &texts {
        kokoro.send(text.to_string()).await.unwrap();
    }
    kokoro.close().await.unwrap();

    for text in &texts {
        let sample = kokoro
            .next()
            .await
            .expect("stream should yield an item")
            .unwrap();
        let AudioData::Pcm(tensor) = sample.data;
        assert!(
            !tensor.data.is_empty(),
            "Audio should not be empty for: {}",
            text
        );
        println!("{}: {} samples", text, tensor.data.len());
    }

    assert!(
        kokoro.next().await.is_none(),
        "stream should end after all items"
    );
}

#[tokio::test]
#[ignore]
async fn test_kokoro_long_text() {
    let inference = Inference::cuda(0).expect("CUDA device required");

    let mut kokoro = inference
        .use_kokoro(
            model_path("kokoro/kokoro-v1.0.onnx"),
            model_path("kokoro/bf_emma.npy"),
            Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        )
        .expect("Failed to load Kokoro model");

    // Test moderate length text (within model limits)
    let moderate_text = "The quick brown fox jumps over the lazy dog. ".repeat(5);
    kokoro.send(moderate_text).await.unwrap();
    kokoro.close().await.unwrap();

    let sample = kokoro
        .next()
        .await
        .expect("stream should yield an item")
        .unwrap();
    let AudioData::Pcm(tensor) = sample.data;
    assert!(
        !tensor.data.is_empty(),
        "Moderate text should produce audio"
    );
    println!("Moderate text samples: {}", tensor.data.len());

    assert!(kokoro.next().await.is_none(), "stream should end");
}
