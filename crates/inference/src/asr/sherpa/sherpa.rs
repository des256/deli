use crate::{
    asr::Transcription,
    error::{InferError, Result},
};
use audio::AudioSample;
use futures_core::Stream;
use futures_sink::Sink;
use std::{
    collections::VecDeque,
    future::Future,
    path::Path,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use super::asrcore::AsrCore;

const SHERPA_ENCODER_PATH: &str = "data/sherpa/encoder-epoch-99-avg-1-chunk-16-left-128";
const SHERPA_DECODER_PATH: &str = "data/sherpa/decoder-epoch-99-avg-1-chunk-16-left-128.onnx";
const SHERPA_JOINER_PATH: &str = "data/sherpa/joiner-epoch-99-avg-1-chunk-16-left-128.onnx";
const SHERPA_TOKENS_PATH: &str = "data/sherpa/tokens.txt";

const REQUIRED_SAMPLE_RATE: usize = 16000;
const HOP_SIZE: usize = 160; // 10ms hop at 16kHz
const WINDOW_SIZE: usize = 400; // 25ms window at 16kHz

/// Streaming ASR using Zipformer-Transducer architecture.
///
/// This implementation uses three ONNX models:
/// - Encoder: Converts mel features to encoder embeddings
/// - Decoder: Converts context tokens to decoder embeddings
/// - Joiner: Combines encoder + decoder embeddings to predict next token
///
/// The encoder is stateful - it maintains hidden states across chunks.
///
/// Implements `Sink<AudioSample>` to accept audio input and
/// `Stream<Item = Result<Transcription>>` to produce transcriptions.
pub struct Sherpa {
    // Core state (shared with async tasks)
    core: Arc<Mutex<AsrCore>>,

    // Streaming state
    sample_buffer: Vec<i16>,
    decoded_text: String,
    chunk_samples: usize,
    pending_chunks: VecDeque<Vec<i16>>,
    closed: bool,
    inflight: Option<Pin<Box<dyn Future<Output = Result<String>> + Send>>>,
    stream_waker: Option<std::task::Waker>,
}

impl Sherpa {
    /// Create a new streaming ASR instance from pre-built ONNX sessions.
    ///
    /// # Arguments
    /// * `encoder` - Encoder ONNX session
    /// * `decoder` - Decoder ONNX session
    /// * `joiner` - Joiner ONNX session
    /// * `tokens_path` - Path to tokens.txt
    ///
    /// # Errors
    /// Returns an error if the token file is invalid or model shapes are unexpected.
    pub fn new<P: AsRef<Path>>(onnx: &Arc<onnx::Onnx>, executor: &onnx::Executor) -> Result<Self> {
        let encoder = onnx.create_session(executor, SHERPA_ENCODER_PATH)?;
        let decoder = onnx.create_session(executor, SHERPA_DECODER_PATH)?;
        let joiner = onnx.create_session(executor, SHERPA_JOINER_PATH)?;

        // Load tokens
        let tokens = super::tokens::load_tokens(SHERPA_TOKENS_PATH)?;

        // Discover encoder state input/output names
        let encoder_state_names = AsrCore::discover_encoder_states(&encoder)?;

        // Initialize encoder states (will be populated with zeros of correct shape)
        let encoder_states = AsrCore::initialize_encoder_states(&encoder, &encoder_state_names)?;

        // Determine context_size from decoder's "y" input shape [batch, context_size]
        let context_size = decoder
            .input_shape(0)
            .map_err(|e| InferError::Runtime(format!("Failed to get decoder input shape: {}", e)))
            .and_then(|shape| {
                if shape.len() >= 2 && shape[1] > 0 {
                    Ok(shape[1] as usize)
                } else {
                    Ok(2) // fallback for dynamic dims
                }
            })?;

        // Determine chunk size from encoder's "x" input shape [batch, num_frames, 80]
        let x_shape = encoder.input_shape(0).map_err(|e| {
            InferError::Runtime(format!("Failed to get encoder x input shape: {}", e))
        })?;
        let required_frames = if x_shape.len() >= 2 && x_shape[1] > 0 {
            x_shape[1] as usize
        } else {
            return Err(InferError::Runtime(
                "Encoder 'x' input has no fixed frame dimension".to_string(),
            ));
        };
        // Convert frames to samples: num_samples = (num_frames - 1) * hop + window
        let chunk_samples = (required_frames - 1) * HOP_SIZE + WINDOW_SIZE;

        // Initialize context with blank tokens (ID 0)
        let context = vec![0i64; context_size];

        let core = AsrCore {
            encoder,
            decoder,
            joiner,
            tokens,
            context,
            encoder_states,
            encoder_state_names,
        };

        Ok(Self {
            core: Arc::new(Mutex::new(core)),
            sample_buffer: Vec::new(),
            decoded_text: String::new(),
            chunk_samples,
            pending_chunks: VecDeque::new(),
            closed: false,
            inflight: None,
            stream_waker: None,
        })
    }

    /// Configure the chunk size in samples.
    ///
    /// Default is 5120 samples (320ms at 16kHz).
    pub fn with_chunk_samples(mut self, samples: usize) -> Self {
        self.chunk_samples = samples;
        self
    }

    /// Start processing an audio chunk.
    fn start_decode(&mut self, chunk: Vec<i16>) {
        let core = Arc::clone(&self.core);

        self.inflight = Some(Box::pin(async move {
            tokio::task::spawn_blocking(move || -> Result<String> {
                // Need at least 400 samples (25ms window at 16kHz) for one frame
                if chunk.len() < 400 {
                    return Ok(String::new());
                }

                // Compute features
                let num_frames = (chunk.len() - 400) / 160 + 1; // 25ms window, 10ms hop
                let features = super::features::compute_features(&chunk, REQUIRED_SAMPLE_RATE)?;

                // Decode chunk
                let mut core_guard = core
                    .lock()
                    .map_err(|e| InferError::Runtime(format!("Core lock poisoned: {}", e)))?;

                core_guard.decode_chunk(&features, num_frames)
            })
            .await
            .map_err(|e| InferError::Runtime(format!("Task join error: {}", e)))?
        }));
    }

    /// Handle a completed decode result, returning the appropriate transcription.
    fn handle_decode_result(
        &mut self,
        result: Result<String>,
    ) -> Poll<Option<Result<Transcription>>> {
        match result {
            Ok(text) => {
                // Append to cumulative text (SentencePiece tokens handle word boundaries)
                self.decoded_text.push_str(&text);

                if self.closed && self.pending_chunks.is_empty() {
                    let final_text = std::mem::take(&mut self.decoded_text);
                    Poll::Ready(Some(Ok(Transcription::Final {
                        text: final_text,
                        confidence: 1.0,
                    })))
                } else {
                    Poll::Ready(Some(Ok(Transcription::Partial {
                        text: self.decoded_text.clone(),
                        confidence: 1.0,
                    })))
                }
            }
            Err(e) => Poll::Ready(Some(Err(e))),
        }
    }
}

// Decode logic is in decode.rs (impl AsrCore)

impl Sink<AudioSample> for Sherpa {
    type Error = InferError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: AudioSample) -> Result<()> {
        let this = self.get_mut();

        // Validate sample rate
        if item.sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(InferError::Runtime(format!(
                "requires {} Hz audio, got {} Hz",
                REQUIRED_SAMPLE_RATE, item.sample_rate
            )));
        }

        // Extract PCM data
        let pcm_data = match &item.data {
            audio::AudioData::Pcm(tensor) => &tensor.data,
        };

        // Append to sample buffer
        this.sample_buffer.extend_from_slice(&pcm_data);

        // Extract complete chunks
        while this.sample_buffer.len() >= this.chunk_samples {
            let chunk: Vec<i16> = this.sample_buffer.drain(..this.chunk_samples).collect();
            this.pending_chunks.push_back(chunk);
        }

        // Wake the stream
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

        // If there are remaining samples, zero-pad to chunk size and enqueue
        if !this.sample_buffer.is_empty() {
            let mut chunk: Vec<i16> = this.sample_buffer.drain(..).collect();
            chunk.resize(this.chunk_samples, 0);
            this.pending_chunks.push_back(chunk);
        }

        if let Some(waker) = this.stream_waker.take() {
            waker.wake();
        }
        Poll::Ready(Ok(()))
    }
}

impl Stream for Sherpa {
    type Item = Result<Transcription>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Poll inflight decode
        if let Some(fut) = this.inflight.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    this.inflight = None;
                    return this.handle_decode_result(result);
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Dequeue next chunk and start decode
        if let Some(chunk) = this.pending_chunks.pop_front() {
            this.start_decode(chunk);
            if let Some(fut) = this.inflight.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(result) => {
                        this.inflight = None;
                        return this.handle_decode_result(result);
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
