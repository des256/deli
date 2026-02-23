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
        path::Path,
        pin::Pin,
        sync::{Arc, Mutex},
        task::{Context, Poll},
    },
    tokenizers::Tokenizer,
    tokio::sync::mpsc,
};

const SAMPLE_RATE: usize = 24000;

/// Channel capacity for streaming audio chunks between the synthesis
/// thread and the async stream. Each chunk is ~480 samples (~20ms at
/// 24kHz), so 8 chunks ≈ 160ms of audio lookahead.
const CHUNK_CHANNEL_CAPACITY: usize = 64;

/// Load pre-encoded voice latents from a binary file.
///
/// File format (little-endian):
///   4 bytes  - number of dimensions (u32)
///   N×8 bytes - each dimension (u64)
///   remainder - raw f32 data
fn load_voice_latents(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    use std::io::Read;

    let path = path.as_ref();
    let mut file = std::fs::File::open(path).map_err(|e| {
        InferError::Runtime(format!(
            "Failed to open voice latents file '{}': {}",
            path.display(),
            e
        ))
    })?;

    // Read number of dimensions
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)
        .map_err(|e| InferError::Runtime(format!("Failed to read latents header: {}", e)))?;
    let ndims = u32::from_le_bytes(buf4) as usize;

    // Read dimensions
    let mut total_elements: usize = 1;
    for _ in 0..ndims {
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8).map_err(|e| {
            InferError::Runtime(format!("Failed to read latents dimensions: {}", e))
        })?;
        let dim = u64::from_le_bytes(buf8) as usize;
        total_elements = total_elements
            .checked_mul(dim)
            .ok_or_else(|| InferError::Runtime("Voice latents shape overflow".to_string()))?;
    }

    // Read f32 data
    let mut data = vec![0f32; total_elements];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            total_elements * std::mem::size_of::<f32>(),
        )
    };
    file.read_exact(byte_slice)
        .map_err(|e| InferError::Runtime(format!("Failed to read latents data: {}", e)))?;

    // Handle endianness: file is little-endian, convert if needed
    #[cfg(target_endian = "big")]
    for val in &mut data {
        *val = f32::from_le_bytes(val.to_ne_bytes());
    }

    Ok(data)
}

/// Pocket TTS model for text-to-speech synthesis.
///
/// Implements `Sink<String>` to accept text input and
/// `Stream<Item = Result<AudioSample>>` to produce synthesized audio.
///
/// Audio is streamed incrementally: each AR generation step produces
/// a small audio chunk (~20ms) that is yielded immediately. After all
/// chunks for a text are yielded, an empty `AudioSample` (0 samples)
/// signals end-of-utterance. Closing the sink signals no more input;
/// the stream ends once all pending texts are synthesized.
pub struct PocketTts {
    core: Arc<Mutex<PocketCore>>,
    tokenizer: Arc<Tokenizer>,
    pending: VecDeque<String>,
    closed: bool,
    chunk_rx: Option<mpsc::Receiver<Result<AudioSample>>>,
    stream_waker: Option<std::task::Waker>,
}

impl PocketTts {
    /// Create a new PocketTts instance
    ///
    /// # Arguments
    /// * `text_conditioner` - Text conditioner ONNX session
    /// * `flow_main` - Flow LM main ONNX session
    /// * `flow_step` - Flow LM step ONNX session
    /// * `mimi_decoder` - Mimi decoder ONNX session
    /// * `tokenizer` - Loaded tokenizer
    /// * `voice_latents` - Pre-encoded voice latents from the mimi encoder (flat f32 slice,
    ///   shape [1, T, 1024] where T = voice_latents.len() / 1024)
    pub fn new(
        text_conditioner: Session,
        flow_main: Session,
        flow_step: Session,
        mimi_decoder: Session,
        tokenizer: Tokenizer,
        voice_latents_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let voice_latents = load_voice_latents(voice_latents_path)?;

        // Create PocketCore
        let mut core = PocketCore::new(text_conditioner, flow_main, flow_step, mimi_decoder)?;

        // Condition voice (voice first per reference)
        let latent_frames = voice_latents.len() / 1024; // 1024 = conditioning dim
        core.condition_voice(&voice_latents, latent_frames)?;

        // Snapshot the voice-conditioned state
        core.snapshot_state()?;

        Ok(PocketTts {
            core: Arc::new(Mutex::new(core)),
            tokenizer: Arc::new(tokenizer),
            pending: VecDeque::new(),
            closed: false,
            chunk_rx: None,
            stream_waker: None,
        })
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
                prepared = first_char.to_uppercase().collect::<String>()
                    + &prepared[first_char.len_utf8()..];
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

    /// Split text into sentences for synthesis.
    ///
    /// The flow_main transformer has a 1000-position KV cache. Voice conditioning
    /// uses ~20 positions and text conditioning uses one per token, so long text
    /// must be split into sentences to stay within the limit.
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            // Split on sentence-ending punctuation followed by space or end
            if matches!(ch, '.' | '!' | '?' | ';') {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        // Push remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Synthesize a single sentence: reset → condition → generate → stream.
    fn synthesize_sentence(
        core_guard: &mut PocketCore,
        tokenizer: &Tokenizer,
        sentence: &str,
        tx: &mpsc::Sender<Result<AudioSample>>,
    ) -> Result<()> {
        // Reset state for new sentence
        core_guard.reset_for_utterance()?;

        // Prepare text
        let (prepared, frames_after_eos) = Self::prepare_text(sentence);

        // Convert to SentencePiece format
        let sp_text = format!("\u{2581}{}", prepared.replace(' ', "\u{2581}"));

        // Tokenize
        let encoding = tokenizer
            .encode(sp_text, false)
            .map_err(|e| InferError::Runtime(format!("Tokenization failed: {}", e)))?;
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        // Run text conditioner
        let embeddings = core_guard.run_text_conditioner(&token_ids)?;

        // Condition text (text second, after voice from snapshot)
        core_guard.condition_text(&embeddings, token_ids.len())?;

        // AR generation loop — stream each chunk immediately
        let max_tokens = 1000;
        let mut eos_countdown: Option<usize> = None;

        for _ in 0..max_tokens {
            let (latent, is_eos) = core_guard.generate_step()?;
            let audio_chunk = core_guard.decode_audio(&latent)?;

            let i16_samples: Vec<i16> = audio_chunk
                .iter()
                .map(|&sample| (sample * 32768.0).clamp(-32768.0, 32767.0) as i16)
                .collect();

            let num_samples = i16_samples.len();
            let tensor = Tensor::new(vec![num_samples], i16_samples)
                .map_err(|e| InferError::TensorError(e.to_string()))?;

            let sample = AudioSample {
                data: AudioData::Pcm(tensor),
                sample_rate: SAMPLE_RATE,
            };

            if tx.capacity() == 0 {
                base::log_debug!("TTS chunk channel full (cap {})", CHUNK_CHANNEL_CAPACITY);
            }
            if tx.blocking_send(Ok(sample)).is_err() {
                return Ok(());
            }

            if let Some(ref mut remaining) = eos_countdown {
                if *remaining == 0 {
                    break;
                }
                *remaining -= 1;
            } else if is_eos {
                eos_countdown = Some(frames_after_eos);
            }
        }

        Ok(())
    }

    /// Spawn synthesis of the given text, streaming audio chunks through a channel.
    ///
    /// Long text is automatically split into sentences so each stays within
    /// the flow_main transformer's 1000-position context window.
    fn start_synthesis(&mut self, text: String) {
        let core = Arc::clone(&self.core);
        let tokenizer = Arc::clone(&self.tokenizer);
        let (tx, rx) = mpsc::channel(CHUNK_CHANNEL_CAPACITY);
        self.chunk_rx = Some(rx);

        tokio::task::spawn_blocking(move || {
            let result = (|| -> Result<()> {
                let mut core_guard = core
                    .lock()
                    .map_err(|e| InferError::Runtime(format!("Core lock poisoned: {}", e)))?;

                let sentences = Self::split_sentences(&text);
                for sentence in &sentences {
                    Self::synthesize_sentence(&mut core_guard, &tokenizer, sentence, &tx)?;
                }

                Ok(())
            })();

            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });
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

        // Poll current chunk receiver
        if let Some(rx) = this.chunk_rx.as_mut() {
            match rx.poll_recv(cx) {
                Poll::Ready(Some(chunk)) => return Poll::Ready(Some(chunk)),
                Poll::Ready(None) => {
                    // Channel closed — synthesis done for this text.
                    // Yield empty AudioSample as end-of-utterance marker.
                    this.chunk_rx = None;
                    let marker = AudioSample {
                        data: AudioData::Pcm(Tensor::new(vec![0], vec![]).unwrap()),
                        sample_rate: SAMPLE_RATE,
                    };
                    return Poll::Ready(Some(Ok(marker)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Start synthesis for next pending text
        if let Some(text) = this.pending.pop_front() {
            this.start_synthesis(text);
            // Re-poll the new receiver
            if let Some(rx) = this.chunk_rx.as_mut() {
                match rx.poll_recv(cx) {
                    Poll::Ready(Some(chunk)) => return Poll::Ready(Some(chunk)),
                    Poll::Ready(None) => {
                        this.chunk_rx = None;
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

    #[test]
    fn test_split_sentences_basic() {
        let sentences = PocketTts::split_sentences("Hello world. How are you? I am fine!");
        assert_eq!(
            sentences,
            vec!["Hello world.", "How are you?", "I am fine!"]
        );
    }

    #[test]
    fn test_split_sentences_semicolons() {
        let sentences = PocketTts::split_sentences("First part; second part.");
        assert_eq!(sentences, vec!["First part;", "second part."]);
    }

    #[test]
    fn test_split_sentences_no_punctuation() {
        let sentences = PocketTts::split_sentences("No punctuation here");
        assert_eq!(sentences, vec!["No punctuation here"]);
    }

    #[test]
    fn test_split_sentences_single_sentence() {
        let sentences = PocketTts::split_sentences("Just one sentence.");
        assert_eq!(sentences, vec!["Just one sentence."]);
    }

    #[test]
    fn test_split_sentences_empty() {
        let sentences = PocketTts::split_sentences("");
        assert!(sentences.is_empty());
    }
}
