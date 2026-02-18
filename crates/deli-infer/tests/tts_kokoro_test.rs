use deli_infer::{Inference, Kokoro};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn model_path(relative: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR points to the crate root (crates/deli-infer)
    // Models are at the workspace root: ../../models/kokoro/
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("../../models").join(relative)
}

#[test]
fn test_kokoro_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Kokoro>();
}

#[tokio::test]
#[ignore] // Requires model files at models/kokoro/
async fn test_kokoro_integration() {
    let inference = Inference::cuda(0).expect("CUDA device required");

    let kokoro = inference
        .use_kokoro(
            model_path("kokoro/kokoro-v1.0.onnx"),
            model_path("kokoro/bf_emma.npy"),
            Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        )
        .expect("Failed to load Kokoro model");

    let result = kokoro.speak("Hello world").await;
    assert!(result.is_ok(), "speak() should succeed: {:?}", result.err());

    let tensor = result.unwrap();
    assert!(tensor.shape[0] > 0, "Audio tensor should have samples");
    assert!(tensor.data.len() > 0, "Audio data should not be empty");
    assert_eq!(tensor.shape[0], tensor.data.len(), "Shape should match data length");

    let output_path = "/tmp/kokoro_output.raw";
    let mut file = fs::File::create(output_path).expect("Failed to create output file");
    for sample in &tensor.data {
        file.write_all(&sample.to_le_bytes()).expect("Failed to write sample");
    }
    println!("Audio written to {}", output_path);
    println!("Play with: ffplay -f s16le -ar 24000 -ac 1 {}", output_path);
    println!("Samples: {}", tensor.data.len());
}

#[tokio::test]
#[ignore]
async fn test_kokoro_multiple_utterances() {
    let inference = Inference::cuda(0).expect("CUDA device required");

    let kokoro = inference
        .use_kokoro(
            model_path("kokoro/kokoro-v1.0.onnx"),
            model_path("kokoro/bf_emma.npy"),
            Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        )
        .expect("Failed to load Kokoro model");

    let texts = ["Hello world", "How are you today?", "This is a test of the Kokoro TTS system."];

    for text in &texts {
        let result = kokoro.speak(text).await;
        assert!(result.is_ok(), "speak() should succeed for: {}", text);
        let tensor = result.unwrap();
        assert!(tensor.data.len() > 0, "Audio should not be empty for: {}", text);
        println!("{}: {} samples", text, tensor.data.len());
    }
}

#[tokio::test]
#[ignore]
async fn test_kokoro_long_text() {
    let inference = Inference::cuda(0).expect("CUDA device required");

    let kokoro = inference
        .use_kokoro(
            model_path("kokoro/kokoro-v1.0.onnx"),
            model_path("kokoro/bf_emma.npy"),
            Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        )
        .expect("Failed to load Kokoro model");

    // Test moderate length text (within model limits)
    let moderate_text = "The quick brown fox jumps over the lazy dog. ".repeat(5);
    let result = kokoro.speak(&moderate_text).await;
    assert!(result.is_ok(), "speak() should handle moderate text: {:?}", result.err());
    let tensor = result.unwrap();
    assert!(tensor.data.len() > 0, "Moderate text should produce audio");
    println!("Moderate text samples: {}", tensor.data.len());

    // Very long text may exceed model capacity - verify it returns an error rather than panicking
    let very_long_text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
    let result = kokoro.speak(&very_long_text).await;
    // Either succeeds or returns a graceful error (no panic)
    println!("Very long text result: {:?}", result.is_ok());
}
