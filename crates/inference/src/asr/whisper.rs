use crate::asr::transcription::Transcription;
use crate::asr::{
    audio::pcm_to_mel, config::Config, model::Whisper as WhisperModel, token_decoder::TokenDecoder,
};
use crate::error::{InferError, Result};
use audio::{AudioData, AudioSample};
use base::Language;
use candle_core::{Device, Tensor as CandleTensor};
use candle_nn::VarBuilder;
use futures_core::Stream;
use futures_sink::Sink;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokenizers::Tokenizer;

const DEFAULT_WINDOW_SAMPLES: usize = 48000; // 3 seconds at 16kHz
const REQUIRED_SAMPLE_RATE: usize = 16000;

/// Whisper-based speech recognition system.
///
/// Implements `Sink<AudioSample>` to accept audio input and
/// `Stream<Item = Result<Transcription>>` to produce transcription results.
///
/// Audio is buffered internally. When enough samples accumulate (configurable
/// via `with_window_samples`), a transcription is triggered and the result
/// is yielded from the stream. Closing the sink flushes any remaining audio.
pub struct Whisper {
    model: Arc<WhisperModel>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
    device: Device,
    buffer: Vec<i16>,
    window_samples: usize,
    closed: bool,
    inflight: Option<Pin<Box<dyn Future<Output = Result<Transcription>> + Send>>>,
    stream_waker: Option<std::task::Waker>,
}

impl Whisper {
    /// Create a new Whisper recognizer from model, tokenizer, and config files.
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        model_path: P1,
        tokenizer_path: P2,
        config_path: P3,
        device: Device,
    ) -> Result<Self> {
        let config_json = std::fs::read_to_string(config_path)
            .map_err(|e| InferError::Io(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_json)
            .map_err(|e| InferError::Runtime(format!("Failed to parse config: {}", e)))?;

        Self::new_with_config(model_path, tokenizer_path, config, device)
    }

    /// Create a new Whisper recognizer with an explicit config.
    pub fn new_with_config<P1: AsRef<Path>, P2: AsRef<Path>>(
        model_path: P1,
        tokenizer_path: P2,
        config: Config,
        device: Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.as_ref()],
                candle_core::DType::F32,
                &device,
            )
            .map_err(|e| InferError::Runtime(format!("Failed to load model: {}", e)))?
        };

        let model = WhisperModel::load(vb, config.clone())
            .map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to build Whisper model. Expected OpenAI Whisper format from openai/whisper-tiny.en. Error: {}",
                    e
                ))
            })?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferError::Runtime(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
            device,
            buffer: Vec::new(),
            window_samples: DEFAULT_WINDOW_SAMPLES,
            closed: false,
            inflight: None,
            stream_waker: None,
        })
    }

    /// Set the number of samples to accumulate before triggering transcription.
    ///
    /// Default is 48000 (3 seconds at 16kHz).
    pub fn with_window_samples(mut self, samples: usize) -> Self {
        self.window_samples = samples;
        self
    }

    /// Spawn transcription of the given PCM samples as an inflight future.
    fn start_transcription(&mut self, samples: Vec<i16>) {
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let config = self.config.clone();
        let device = self.device.clone();

        self.inflight = Some(Box::pin(async move {
            let text = tokio::task::spawn_blocking(move || -> Result<String> {
                let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

                let mel_data = pcm_to_mel(&config, &f32_samples);
                let num_frames = mel_data.len() / config.num_mel_bins;
                let mel = CandleTensor::from_vec(
                    mel_data,
                    vec![1, config.num_mel_bins, num_frames],
                    &device,
                )
                .map_err(|e| {
                    InferError::TensorError(format!("Failed to create mel tensor: {}", e))
                })?;

                let mut decoder =
                    TokenDecoder::new((*model).clone(), (*tokenizer).clone(), &device, &config)?;

                decoder.run(&mel)
            })
            .await
            .map_err(|e| InferError::Runtime(format!("Task join error: {}", e)))??;

            Ok(Transcription::Final {
                text,
                language: Language::EnglishUs,
                confidence: 1.0,
            })
        }));
    }
}

impl Sink<AudioSample> for Whisper {
    type Error = InferError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: AudioSample) -> Result<()> {
        let this = self.get_mut();

        if item.sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(InferError::Runtime(format!(
                "Whisper requires {} Hz audio, got {} Hz. Please resample before sending.",
                REQUIRED_SAMPLE_RATE, item.sample_rate
            )));
        }

        let AudioData::Pcm(tensor) = item.data;
        this.buffer.extend(&tensor.data);

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

impl Stream for Whisper {
    type Item = Result<Transcription>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Poll inflight transcription
        if let Some(fut) = this.inflight.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    this.inflight = None;
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Start new transcription if enough data accumulated
        if this.buffer.len() >= this.window_samples {
            let samples: Vec<i16> = this.buffer.drain(..this.window_samples).collect();
            this.start_transcription(samples);
            // Poll the newly created inflight
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

        // If closed and buffer has remaining audio, flush it
        if this.closed && !this.buffer.is_empty() {
            let samples = std::mem::take(&mut this.buffer);
            this.start_transcription(samples);
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

        // Stream complete when closed and buffer is empty
        if this.closed {
            return Poll::Ready(None);
        }

        // Not enough data yet
        this.stream_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
