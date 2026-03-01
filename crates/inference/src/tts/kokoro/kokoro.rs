use {
    crate::{
        error::{InferError, Result},
        tts::kokoro::{
            phonemize::phonemize,
            vocab::{tokenize, vocab},
        },
    },
    audio::{AudioData, AudioSample},
    base::Tensor,
    futures_core::Stream,
    futures_sink::Sink,
    onnx::{Session, Value},
    std::{
        collections::{HashMap, VecDeque},
        future::Future,
        pin::Pin,
        sync::{Arc, Mutex},
        task::{Context, Poll},
    },
};

const KOKORO_MODEL_PATH: &str = "data/tts/kokoro/kokoro-v1.0.onnx";
const KOKORO_VOICE_PATH: &str = "data/tts/kokoro/bf_nicole.npy";
const KOKORO_ESPEAK_DATA_PATH: &str = "/usr/lib/x86_64-linux-gnu/espeak-ng-data";

const SAMPLE_RATE: usize = 24000;

pub(crate) fn load_voice_style(path: impl AsRef<std::path::Path>) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;

    // Validate magic bytes
    if bytes.len() < 6 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(InferError::Runtime(
            "Invalid NPY file: missing \\x93NUMPY magic bytes".to_string(),
        ));
    }

    // Skip 128-byte header and parse as little-endian f32
    if bytes.len() < 128 {
        return Err(InferError::Runtime(
            "Invalid NPY file: too small for header".to_string(),
        ));
    }

    let style: Vec<f32> = bytes[128..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Validate expected size: 510 × 256 = 130560
    if style.len() != 130560 {
        return Err(InferError::Runtime(format!(
            "Voice file wrong size: expected 130560 f32s, got {}",
            style.len()
        )));
    }

    Ok(style)
}

pub struct Kokoro {
    session: Arc<Session>,
    vocab: Arc<HashMap<char, i64>>,
    style: Arc<Vec<f32>>,
    pending: VecDeque<String>,
    closed: bool,
    inflight: Option<Pin<Box<dyn Future<Output = Result<AudioSample>> + Send>>>,
    stream_waker: Option<std::task::Waker>,
}

impl Kokoro {
    pub fn new(
        onnx: &Arc<onnx::Onnx>,
        executor: onnx::Executor,
        voice_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let session = onnx.create_session(executor, KOKORO_MODEL_PATH)?;
        // Initialize espeak-ng
        super::phonemize::espeak_init(Some(KOKORO_ESPEAK_DATA_PATH))
            .map_err(|e| InferError::Runtime(e))?;

        let vocab = vocab();
        let style = load_voice_style(voice_path)?;

        Ok(Kokoro {
            session: Arc::new(session),
            vocab: Arc::new(vocab),
            style: Arc::new(style),
            pending: VecDeque::new(),
            closed: false,
            inflight: None,
            stream_waker: None,
        })
    }

    // Refactor to: spawn blocking task, create session inside task, get text from channel, feed text to TTS, send audio samples back over other channel
    fn start_synthesis(&mut self, text: String) {
        let session = Arc::clone(&self.session);
        let vocab = Arc::clone(&self.vocab);
        let style = Arc::clone(&self.style);

        self.inflight = Some(Box::pin(async move {
            let tensor = tokio::task::spawn_blocking(move || -> Result<Tensor<i16>> {
                // Phonemize text
                let phonemes = phonemize(&text).map_err(|e| InferError::Runtime(e))?;

                // Tokenize phonemes, pad with 0 at start and end
                let mut token_ids = tokenize(&phonemes, &vocab);
                token_ids.insert(0, 0);
                token_ids.push(0);

                // Clamp style index to prevent OOB
                let style_idx = token_ids.len().min(509);

                // Extract style slice for this token count
                let style_start = style_idx * 256;
                let style_end = style_start + 256;
                let style_slice = &style[style_start..style_end];

                // Build input tensors
                let tokens_shape = [1usize, token_ids.len()];
                let tokens =
                    Value::from_slice::<i64>(&self.session.onnx, &tokens_shape, &token_ids)?;

                let style_shape = [1usize, 256];
                let style_vec = style_slice.to_vec();
                let style_tensor =
                    Value::from_slice::<f32>(&self.session.onnx, &style_shape, &style_vec)?;

                let speed_shape = [1usize];
                let speed_tensor =
                    Value::from_slice::<f32>(&self.session.onnx, &speed_shape, &[1.0f32])?;

                // Run inference with named inputs, extract data within lock scope
                let outputs = session.run(
                    &[
                        ("tokens", &tokens),
                        ("style", &style_tensor),
                        ("speed", &speed_tensor),
                    ],
                    &["audio"],
                )?;

                let output_data = outputs[0]
                    .extract_tensor::<f32>()
                    .map_err(|e| InferError::Runtime(format!("Failed to extract output: {}", e)))?;

                let samples: Vec<i16> = output_data
                    .iter()
                    .map(|&sample| (sample * 32768.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();

                let num_samples = samples.len();
                Tensor::new(vec![num_samples], samples)
                    .map_err(|e| InferError::TensorError(e.to_string()))
            })
            .await
            .map_err(|e| InferError::Runtime(format!("Task join error: {}", e)))??;

            Ok(AudioSample {
                data: AudioData::Pcm(tensor),
                sample_rate: SAMPLE_RATE,
            })
        }));
    }
}

impl Sink<String> for Kokoro {
    type Error = InferError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: String) -> Result<()> {
        let this = self.get_mut();
        this.pending.push_back(item);
        if let Some(waker) = this.stream_waker.take() {
            waker.wake();
        }
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        let this = self.get_mut();
        this.closed = true;
        if let Some(waker) = this.stream_waker.take() {
            waker.wake();
        }
        Poll::Ready(Ok(()))
    }
}

impl Stream for Kokoro {
    type Item = Result<AudioSample>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Poll inflight synthesis
        if let Some(fut) = this.inflight.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    this.inflight = None;
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Dequeue next text and start synthesis
        if let Some(text) = this.pending.pop_front() {
            this.start_synthesis(text);
            if let Some(fut) = this.inflight.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(result) => {
                        this.inflight = None;
                        return Poll::Ready(Some(result));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }

        // Stream complete when closed, pending empty, and no inflight
        if this.closed {
            return Poll::Ready(None);
        }

        // Nothing to do yet — park
        this.stream_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

// Ensure Kokoro is Send (not Sync — inflight future is Send but not Sync)
fn _assert_send() {
    fn assert<T: Send>() {}
    assert::<Kokoro>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_voice_style_missing_magic() {
        let path = "/tmp/test_invalid_magic.npy";
        std::fs::write(path, b"invalid data").unwrap();
        let result = load_voice_style(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid NPY file"),
            "Expected 'Invalid NPY file' error, got: {}",
            err
        );
    }

    #[test]
    fn test_load_voice_style_too_small() {
        let path = "/tmp/test_too_small.npy";
        // Valid magic but file too small for header
        let mut data = b"\x93NUMPY".to_vec();
        data.extend_from_slice(&[0u8; 50]); // not enough for 128-byte header
        std::fs::write(path, &data).unwrap();
        let result = load_voice_style(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("too small for header"),
            "Expected 'too small' error, got: {}",
            err
        );
    }

    #[test]
    fn test_load_voice_style_wrong_size() {
        let path = "/tmp/test_wrong_size.npy";
        // Valid magic + 128-byte header + wrong number of f32s
        let mut data = b"\x93NUMPY".to_vec();
        data.extend_from_slice(&[0u8; 122]); // pad to 128 bytes total
        // Add 4 bytes (1 f32) of data - wrong size
        data.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(path, &data).unwrap();
        let result = load_voice_style(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("wrong size"),
            "Expected 'wrong size' error, got: {}",
            err
        );
    }

    #[test]
    fn test_load_voice_style_file_not_found() {
        let result = load_voice_style("/tmp/nonexistent_voice.npy");
        assert!(result.is_err());
    }
}
