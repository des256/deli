use {
    crate::{
        error::{InferError, Result},
        tts::pocket::core::PocketCore,
    },
    audio::{AudioData, AudioSample},
    base::Tensor,
    futures_core::Stream,
    futures_sink::Sink,
    onnx::Session,
    std::{
        collections::VecDeque,
        future::Future,
        path::Path,
        pin::Pin,
        sync::{Arc, Mutex},
        task::{Context, Poll},
    },
    tokenizers::Tokenizer,
};

const SAMPLE_RATE: usize = 24000;

/// Pocket TTS model for text-to-speech synthesis.
///
/// Implements `Sink<String>` to accept text input and
/// `Stream<Item = Result<AudioSample>>` to produce synthesized audio.
///
/// Each `String` sent via the Sink maps 1:1 to an `AudioSample` yielded
/// from the Stream. Closing the sink signals no more input; the stream
/// ends once all pending texts are synthesized.
pub struct PocketTts {
    core: Arc<Mutex<PocketCore>>,
    tokenizer: Arc<Tokenizer>,
    pending: VecDeque<String>,
    closed: bool,
    inflight: Option<Pin<Box<dyn Future<Output = Result<AudioSample>> + Send>>>,
    stream_waker: Option<std::task::Waker>,
}

impl PocketTts {
    /// Create a new PocketTts instance
    ///
    /// # Arguments
    /// * `text_conditioner` - Text conditioner ONNX session
    /// * `flow_main` - Flow LM main ONNX session
    /// * `flow_step` - Flow LM step ONNX session
    /// * `mimi_encoder` - Mimi encoder ONNX session
    /// * `mimi_decoder` - Mimi decoder ONNX session
    /// * `tokenizer_path` - Path to tokenizer.json
    /// * `voice_audio_path` - Path to voice WAV file for voice cloning
    pub fn new(
        text_conditioner: Session,
        flow_main: Session,
        flow_step: Session,
        mimi_encoder: Session,
        mimi_decoder: Session,
        tokenizer_path: impl AsRef<Path>,
        voice_audio_path: impl AsRef<Path>,
    ) -> Result<Self> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            InferError::Runtime(format!("Failed to load tokenizer: {}", e))
        })?;

        // Create PocketCore
        let mut core = PocketCore::new(
            text_conditioner,
            flow_main,
            flow_step,
            mimi_encoder,
            mimi_decoder,
        )?;

        // Load voice audio, encode, and condition (voice first per reference)
        let voice_audio = Self::load_voice_audio(voice_audio_path)?;
        let voice_latents = core.encode_voice(&voice_audio, SAMPLE_RATE)?;
        let latent_frames = voice_latents.len() / 1024; // 1024 = conditioning dim
        core.condition_voice(&voice_latents, latent_frames)?;

        // Snapshot the voice-conditioned state
        core.snapshot_state()?;

        Ok(PocketTts {
            core: Arc::new(Mutex::new(core)),
            tokenizer: Arc::new(tokenizer),
            pending: VecDeque::new(),
            closed: false,
            inflight: None,
            stream_waker: None,
        })
    }

    /// Load voice audio from WAV file.
    ///
    /// Validates: mono channel, 24kHz sample rate, non-silence.
    fn load_voice_audio(path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(path).map_err(|e| {
            InferError::Runtime(format!("Failed to open voice WAV file: {}", e))
        })?;

        let spec = reader.spec();

        // Validate mono
        if spec.channels != 1 {
            return Err(InferError::Runtime(format!(
                "Voice WAV must be mono, got {} channels",
                spec.channels
            )));
        }

        // Validate sample rate
        if spec.sample_rate != SAMPLE_RATE as u32 {
            return Err(InferError::Runtime(format!(
                "Voice WAV must be {}Hz, got {}Hz",
                SAMPLE_RATE, spec.sample_rate
            )));
        }

        // Convert to f32
        let samples: Result<Vec<f32>> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|val| val as f32 / max_val).map_err(|e| {
                        InferError::Runtime(format!("Failed to read sample: {}", e))
                    }))
                    .collect()
            }
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| {
                    InferError::Runtime(format!("Failed to read sample: {}", e))
                }))
                .collect(),
        };
        let samples = samples?;

        // Validate non-silence
        if !samples.is_empty() {
            let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
            let variance: f64 = samples
                .iter()
                .map(|&x| {
                    let diff = x as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / samples.len() as f64;
            if variance < 1e-6 {
                return Err(InferError::Runtime(
                    "Voice audio appears to be silence or near-silence".to_string(),
                ));
            }
        }

        Ok(samples)
    }

    /// Prepare text for synthesis following Pocket TTS conventions.
    ///
    /// Returns (prepared_text, frames_after_eos).
    fn prepare_text(text: &str) -> (String, usize) {
        // Strip whitespace, replace newlines with spaces
        let mut prepared = text.trim().replace('\n', " ");

        // Capitalize first character
        if let Some(first_char) = prepared.chars().next() {
            if first_char.is_lowercase() {
                prepared = first_char.to_uppercase().collect::<String>() + &prepared[first_char.len_utf8()..];
            }
        }

        // Add period if text ends with alphanumeric
        if let Some(last_char) = prepared.chars().last() {
            if last_char.is_alphanumeric() {
                prepared.push('.');
            }
        }

        // Determine frames_after_eos based on word count
        let word_count = prepared.split_whitespace().count();
        let frames_after_eos = if word_count <= 4 { 3 } else { 1 } + 2;

        // Pad short texts (<5 words) with 8 leading spaces
        if word_count < 5 {
            prepared = format!("        {}", prepared);
        }

        (prepared, frames_after_eos)
    }

    /// Spawn synthesis of the given text as an inflight future
    fn start_synthesis(&mut self, text: String) {
        let core = Arc::clone(&self.core);
        let tokenizer = Arc::clone(&self.tokenizer);

        self.inflight = Some(Box::pin(async move {
            let audio_samples = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
                let mut core_guard = core.lock().map_err(|e| {
                    InferError::Runtime(format!("Core lock poisoned: {}", e))
                })?;

                // Reset state for new utterance
                core_guard.reset_for_utterance()?;

                // Prepare text
                let (prepared, frames_after_eos) = Self::prepare_text(&text);

                // Convert to SentencePiece format: prepend ▁ and replace spaces with ▁.
                // The tokenizer.json wraps a SentencePiece Unigram model but has no
                // pre_tokenizer configured, so we must do this manually.
                let sp_text = format!("\u{2581}{}", prepared.replace(' ', "\u{2581}"));

                // Tokenize
                let encoding = tokenizer.encode(sp_text, false).map_err(|e| {
                    InferError::Runtime(format!("Tokenization failed: {}", e))
                })?;
                let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

                // Run text conditioner
                let embeddings = core_guard.run_text_conditioner(&token_ids)?;

                // Condition text (text second, after voice from snapshot)
                core_guard.condition_text(&embeddings, token_ids.len())?;

                // AR generation loop
                let max_tokens = 1000;
                let mut audio_chunks = Vec::new();
                let mut eos_countdown: Option<usize> = None;

                for _ in 0..max_tokens {
                    // Generate one step
                    let (latent, is_eos) = core_guard.generate_step()?;

                    // Decode to audio
                    let audio_chunk = core_guard.decode_audio(&latent)?;
                    audio_chunks.extend_from_slice(&audio_chunk);

                    // EOS handling with continuation frames
                    if let Some(ref mut remaining) = eos_countdown {
                        if *remaining == 0 {
                            break;
                        }
                        *remaining -= 1;
                    } else if is_eos {
                        eos_countdown = Some(frames_after_eos);
                    }
                }

                Ok(audio_chunks)
            })
            .await
            .map_err(|e| InferError::Runtime(format!("Task join error: {}", e)))??;

            // Convert to i16 PCM
            let i16_samples: Vec<i16> = audio_samples
                .iter()
                .map(|&sample| (sample * 32768.0).clamp(-32768.0, 32767.0) as i16)
                .collect();

            let num_samples = i16_samples.len();
            let tensor = Tensor::new(vec![num_samples], i16_samples)
                .map_err(|e| InferError::TensorError(e.to_string()))?;

            Ok(AudioSample {
                data: AudioData::Pcm(tensor),
                sample_rate: SAMPLE_RATE,
            })
        }));
    }
}

impl Sink<String> for PocketTts {
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

impl Stream for PocketTts {
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

// Ensure PocketTts is Send
fn _assert_send() {
    fn assert<T: Send>() {}
    assert::<PocketTts>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_text_capitalizes_first_char() {
        let (text, _) = PocketTts::prepare_text("hello world this is a test.");
        assert!(text.starts_with('H'));
    }

    #[test]
    fn test_prepare_text_adds_period() {
        let (text, _) = PocketTts::prepare_text("Hello world this is a test");
        assert!(text.ends_with('.'));
    }

    #[test]
    fn test_prepare_text_keeps_existing_punctuation() {
        let (text, _) = PocketTts::prepare_text("Hello world this is a test?");
        assert!(text.ends_with('?'));
        assert!(!text.ends_with("?."));
    }

    #[test]
    fn test_prepare_text_pads_short_text() {
        let (text, _) = PocketTts::prepare_text("hello");
        assert!(text.starts_with("        "));
        assert!(text.contains("Hello."));
    }

    #[test]
    fn test_prepare_text_no_pad_long_text() {
        let (text, _) = PocketTts::prepare_text("one two three four five");
        assert!(!text.starts_with(' '));
    }

    #[test]
    fn test_prepare_text_replaces_newlines() {
        let (text, _) = PocketTts::prepare_text("hello\nworld this is a test");
        assert!(!text.contains('\n'));
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn test_prepare_text_frames_after_eos_short() {
        let (_, frames) = PocketTts::prepare_text("Hi");
        assert_eq!(frames, 5); // 3 + 2 for <= 4 words
    }

    #[test]
    fn test_prepare_text_frames_after_eos_long() {
        let (_, frames) = PocketTts::prepare_text("This is a longer sentence with many words");
        assert_eq!(frames, 3); // 1 + 2 for > 4 words
    }

    #[test]
    fn test_prepare_text_empty_string() {
        let (text, frames) = PocketTts::prepare_text("");
        assert_eq!(frames, 5); // 0 words <= 4, so 3+2
        // Empty string gets short-text padding (0 words < 5)
        assert_eq!(text, "        ");
    }

    #[test]
    fn test_prepare_text_single_word() {
        let (text, _) = PocketTts::prepare_text("test");
        assert!(text.starts_with("        "));
        assert!(text.ends_with('.'));
        assert!(text.contains("Test"));
    }
}

