use crate::{
    asr::Transcription,
    error::{InferError, Result},
};
use audio::{self, AudioSample};
use base::log_info;
use onnx::Session;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;

use super::asrcore::AsrCore;
use super::features::compute_features;
use super::tokens::load_tokens;

const REQUIRED_SAMPLE_RATE: usize = 16000;

/// Streaming ASR using NVidia Parakeet FastConformer-Transducer.
///
/// Audio samples are sent via `send()` and transcriptions received via `recv()`.
/// Internally, a background `spawn_blocking` task consumes audio chunks and
/// produces transcriptions through channels.
pub struct Parakeet {
    audio_tx: mpsc::UnboundedSender<Vec<i16>>,
    text_rx: mpsc::UnboundedReceiver<Result<Transcription>>,
    sample_buffer: Vec<i16>,
    chunk_samples: usize,
}

impl Parakeet {
    /// Create a new Parakeet ASR instance.
    ///
    /// # Arguments
    /// - `encoder`: ONNX session for encoder model (input: [1, 128, T], output: [1, 1024, T/8])
    /// - `decoder_joint`: ONNX session for decoder-joint model (combines decoder + joiner)
    /// - `vocab_path`: Path to vocab.txt (SentencePiece tokens, one per line as "<token> <id>")
    pub fn new<P: AsRef<Path>>(
        encoder: Session,
        decoder_joint: Session,
        vocab_path: P,
    ) -> Result<Self> {
        let tokens = load_tokens(vocab_path)?;
        let core = AsrCore::new(encoder, decoder_joint, tokens)?;
        let core = Arc::new(Mutex::new(core));

        // Default chunk size: 1 second at 16kHz
        let chunk_samples = REQUIRED_SAMPLE_RATE;

        // Unbounded channels avoid deadlock: the main task can always push audio
        // without blocking, even when the worker is slow to consume. Likewise the
        // worker can always push transcriptions without blocking on a full text
        // channel while the main task is sending audio.
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
        let (text_tx, text_rx) = mpsc::unbounded_channel::<Result<Transcription>>();

        // Spawn the background decode worker
        Self::spawn_worker(core, audio_rx, text_tx);

        Ok(Self {
            audio_tx,
            text_rx,
            sample_buffer: Vec::new(),
            chunk_samples,
        })
    }

    /// Configure chunk size in samples (default: 16000 = 1 second).
    pub fn with_chunk_samples(mut self, chunk_samples: usize) -> Self {
        self.chunk_samples = chunk_samples;
        self
    }

    /// Send an audio sample for transcription.
    ///
    /// Audio is buffered and dispatched in fixed-size chunks to the background
    /// decode worker. Never blocks — the audio channel is unbounded.
    pub async fn send(&mut self, sample: AudioSample) -> Result<()> {
        // Validate sample rate
        if sample.sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(InferError::Runtime(format!(
                "requires {} Hz audio, got {} Hz",
                REQUIRED_SAMPLE_RATE, sample.sample_rate
            )));
        }

        // Extract PCM data
        let pcm_data = match &sample.data {
            audio::AudioData::Pcm(tensor) => &tensor.data,
        };

        // Append to sample buffer
        self.sample_buffer.extend_from_slice(pcm_data);

        // Extract and send complete chunks
        while self.sample_buffer.len() >= self.chunk_samples {
            let chunk: Vec<i16> = self.sample_buffer.drain(..self.chunk_samples).collect();
            self.audio_tx
                .send(chunk)
                .map_err(|_| InferError::Runtime("decode worker gone".to_string()))?;
        }

        Ok(())
    }

    /// Receive the next transcription result.
    ///
    /// Returns `None` when the stream is finished (after `close()` has been
    /// called and all pending audio has been processed).
    pub async fn recv(&mut self) -> Option<Result<Transcription>> {
        self.text_rx.recv().await
    }

    /// Signal that no more audio will be sent.
    ///
    /// Any remaining buffered samples are zero-padded to a full chunk and
    /// flushed. The background worker will finish processing and then the
    /// `recv()` stream will end.
    pub async fn close(&mut self) -> Result<()> {
        // Flush remaining samples (zero-pad to chunk size)
        if !self.sample_buffer.is_empty() {
            let mut chunk: Vec<i16> = self.sample_buffer.drain(..).collect();
            chunk.resize(self.chunk_samples, 0);
            self.audio_tx
                .send(chunk)
                .map_err(|_| InferError::Runtime("decode worker gone".to_string()))?;
        }

        // Drop the sender so the worker's recv loop terminates
        // (replacing with a closed channel)
        let (closed_tx, _) = mpsc::unbounded_channel();
        self.audio_tx = closed_tx;

        Ok(())
    }

    /// Spawn a blocking background task that reads audio chunks from `audio_rx`,
    /// runs the model, and pushes transcription results to `text_tx`.
    fn spawn_worker(
        core: Arc<Mutex<AsrCore>>,
        mut audio_rx: mpsc::UnboundedReceiver<Vec<i16>>,
        text_tx: mpsc::UnboundedSender<Result<Transcription>>,
    ) {
        tokio::task::spawn_blocking(move || {
            let mut decoded_text = String::new();

            // Block on each audio chunk until the sender is dropped
            while let Some(chunk) = audio_rx.blocking_recv() {
                log_info!("got audio chunk ({} samples), processing...", chunk.len());
                let result = (|| -> Result<String> {
                    let (features, num_frames) = compute_features(&chunk, REQUIRED_SAMPLE_RATE)?;
                    let mut core_guard = core
                        .lock()
                        .map_err(|e| InferError::Runtime(format!("Core lock poisoned: {}", e)))?;
                    core_guard.decode_chunk(&features, num_frames)
                })();

                log_info!("decoded text: {:?}", result);
                match result {
                    Ok(text) => {
                        decoded_text.push_str(&text);
                        let transcription = Transcription::Partial {
                            text: decoded_text.clone(),
                            confidence: 1.0,
                        };
                        if text_tx.send(Ok(transcription)).is_err() {
                            break; // receiver dropped
                        }
                    }
                    Err(e) => {
                        if text_tx.send(Err(e)).is_err() {
                            break;
                        }
                    }
                }
            }

            // All audio consumed — emit final transcription
            if !decoded_text.is_empty() {
                let _ = text_tx.send(Ok(Transcription::Final {
                    text: decoded_text,
                    confidence: 1.0,
                }));
            }
            // text_tx drops here, causing recv() to return None
        });
    }
}
