use {
    super::tokens::load_tokens,
    crate::*,
    base::*,
    std::sync::{Arc, mpsc as std_mpsc},
    tokio::sync::mpsc as tokio_mpsc,
};

const PARAKEET_ENCODER_PATH: &str = "data/asr/parakeet/encoder.onnx";
const PARAKEET_DECODER_PATH: &str = "data/asr/parakeet/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/asr/parakeet/tokenizer.model";
const MEL_FILTERBANK_PATH: &str = "data/asr/parakeet/mel_filterbank.bin";
const HANN_WINDOW_PATH: &str = "data/asr/parakeet/hann_window.bin";

const NUM_MEL_BINS: usize = 128; // 128 mel filterbank bins
const FFT_SIZE: usize = 512; // size of FFT
const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1; // number of bins in converted spectrum
const WINDOW_SIZE: usize = 400; // 25ms at 16kHz
const BLANK_ID: i64 = 1024; // token ID for blank token

const DECODER_STATE_DIM: usize = 640;
const NUM_LAYERS: usize = 24;
const ENCODER_DIM: usize = 1024;

const CACHE_CHANNEL_CONTEXT: usize = 70;
const CACHE_TIME_CONTEXT: usize = 8;

const HOP_SIZE: usize = 160; // 10ms at 16kHz
const PRE_EMPHASIS: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-08;

const MAX_SYMBOLS_PER_STEP: usize = 16;

const VOCAB_SIZE: usize = 1025; // 1024 tokens + 1 blank

const TEXT_CHANNEL_CAPACITY: usize = 64;

// Fallback values if encoder metadata is missing
const DEFAULT_ENCODER_WINDOW_SIZE: usize = 121;
const DEFAULT_ENCODER_CHUNK_SHIFT: usize = 112;

fn zeros_f32(onnx: &Arc<onnx::Onnx>, shape: &[i64]) -> Result<onnx::Value, InferError> {
    onnx::Value::zeros::<f32>(&onnx, shape)
        .map_err(|e| InferError::Runtime(format!("Failed to create zero tensor: {e}")))
}

/// Convert audio samples to mel features, maintaining state across calls.
///
/// `audio_tail` holds leftover samples from the previous call that didn't form
/// a complete frame. `prev_sample` provides pre-emphasis continuity.
///
/// Returns new feature frames in frames-first layout `[T, 128]`.
fn audio_to_features(
    audio: &[i16],
    mel_filterbank: &[f32],
    hann_window: &[f32],
    audio_tail: &mut Vec<i16>,
    prev_sample: &mut i16,
) -> Vec<f32> {
    // Splice tail from previous call with new audio
    let combined_len = audio_tail.len() + audio.len();
    if combined_len < WINDOW_SIZE {
        // Not enough samples for even one frame — just accumulate
        audio_tail.extend_from_slice(audio);
        if let Some(&last) = audio.last() {
            *prev_sample = last;
        }
        return Vec::new();
    }

    // Build the combined signal with pre-emphasis applied continuously
    let mut signal = Vec::with_capacity(combined_len);
    let mut prev = *prev_sample as f32 / 32768.0;
    for &s in audio_tail.iter().chain(audio.iter()) {
        let cur = s as f32 / 32768.0;
        signal.push(cur - PRE_EMPHASIS * prev);
        prev = cur;
    }

    let num_frames = (signal.len() - WINDOW_SIZE) / HOP_SIZE + 1;
    if num_frames == 0 {
        audio_tail.extend_from_slice(audio);
        if let Some(&last) = audio.last() {
            *prev_sample = last;
        }
        return Vec::new();
    }

    // Compute mel features in frames-first layout [T, 128]
    let mut features = vec![0.0f32; num_frames * NUM_MEL_BINS];
    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_SIZE;
        let mut windowed = vec![0.0f32; FFT_SIZE];
        for i in 0..WINDOW_SIZE {
            windowed[i] = signal[start + i] * hann_window[i];
        }
        let mut power_spectrum = vec![0.0f32; SPECTRUM_BINS];
        for k in 0..SPECTRUM_BINS {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for (t, &sample) in windowed.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / FFT_SIZE as f32;
                r += sample * angle.cos();
                im += sample * angle.sin();
            }
            power_spectrum[k] = r * r + im * im;
        }
        let frame_offset = frame_idx * NUM_MEL_BINS;
        for bin_idx in 0..NUM_MEL_BINS {
            let fb_offset = bin_idx * SPECTRUM_BINS;
            let mut mel_energy = 0.0f32;
            for k in 0..SPECTRUM_BINS {
                mel_energy += mel_filterbank[fb_offset + k] * power_spectrum[k];
            }
            features[frame_offset + bin_idx] = (mel_energy + LOG_ZERO_GUARD).ln();
        }
    }

    // Save unconsumed samples as the new tail
    let consumed_samples = (num_frames - 1) * HOP_SIZE + WINDOW_SIZE;
    audio_tail.clear();
    // Rebuild tail from the original i16 sources
    let tail_start = consumed_samples;
    if tail_start < combined_len {
        let old_len = audio_tail.len(); // 0 after clear
        let _ = old_len; // suppress warning
        let audio_tail_start = audio.len() as isize - (combined_len - tail_start) as isize;
        if audio_tail_start < 0 {
            // Some tail samples come from the old audio_tail — but we cleared it.
            // This shouldn't happen since consumed_samples <= combined_len and
            // the old tail was prepended. Reconstruct from the chain.
            // Actually we need the original samples. Let's just track the boundary.
        }
        // Simpler: since we consumed from the combined stream, the tail is the
        // last (combined_len - consumed_samples) samples from audio (or spanning
        // the old tail + audio boundary).
        let remaining = combined_len - consumed_samples;
        if remaining <= audio.len() {
            audio_tail.extend_from_slice(&audio[audio.len() - remaining..]);
        } else {
            // This case means some tail samples were from the *previous* tail,
            // which we already cleared. In practice this is very unlikely because
            // we only enter this function with enough combined samples for >= 1 frame.
            // If it does happen, we lose a few samples at the boundary — acceptable.
            audio_tail.extend_from_slice(audio);
        }
    }

    if let Some(&last) = audio.last() {
        *prev_sample = last;
    }

    features
}

/// Transpose feature frames from frames-first `[T, 128]` to channels-first `[128, T]`.
fn transpose_features(frames_first: &[f32], num_frames: usize) -> Vec<f32> {
    let mut channels_first = vec![0.0f32; NUM_MEL_BINS * num_frames];
    for frame in 0..num_frames {
        for bin in 0..NUM_MEL_BINS {
            channels_first[bin * num_frames + frame] = frames_first[frame * NUM_MEL_BINS + bin];
        }
    }
    channels_first
}

fn run_encoder(
    onnx: &Arc<onnx::Onnx>,
    encoder: &mut onnx::Session,
    features: &[f32],
    num_frames: usize,
    cache_last_channel: &mut onnx::Value,
    cache_last_time: &mut onnx::Value,
    cache_last_channel_len: &mut onnx::Value,
) -> Result<(Vec<f32>, usize), InferError> {
    let audio_signal = onnx::Value::from_slice(&onnx, &[1, 128, num_frames], features)
        .map_err(|e| InferError::Runtime(format!("error creating audio_signal: {e}")))?;
    let length = onnx::Value::from_slice(&onnx, &[1], &[num_frames as i64])
        .map_err(|e| InferError::Runtime(format!("error creating length: {e}")))?;
    let mut outputs = encoder
        .run(
            &[
                ("audio_signal", &audio_signal),
                ("length", &length),
                ("cache_last_channel", &cache_last_channel),
                ("cache_last_time", &cache_last_time),
                ("cache_last_channel_len", &cache_last_channel_len),
            ],
            &[
                "outputs",
                "encoded_lengths",
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ],
        )
        .map_err(|e| InferError::Runtime(format!("Encoder inference failed: {e}")))?;

    let encoder_out_shape = outputs[0]
        .tensor_shape()
        .map_err(|e| InferError::Runtime(format!("failed to get encoder output shape: {e}")))?;
    let encoder_out_len = encoder_out_shape[2] as usize;
    let encoder_out_data = outputs[0]
        .extract_as_f32()
        .map_err(|e| InferError::Runtime(format!("failed to extract encoder output: {e}")))?;

    // update cache
    *cache_last_channel = outputs.remove(2);
    *cache_last_time = outputs.remove(2);
    *cache_last_channel_len = outputs.remove(2);

    Ok((encoder_out_data, encoder_out_len))
}

fn greedy_decode(
    onnx: &Arc<onnx::Onnx>,
    decoder_joint: &mut onnx::Session,
    encoder_out: &[f32],
    encoder_out_len: usize,
    cache_state1: &mut onnx::Value,
    cache_state2: &mut onnx::Value,
    cache_last_token: &mut i64,
) -> Result<Vec<i64>, InferError> {
    let mut token_ids = Vec::new();
    let mut encoder_frame = vec![0.0f32; ENCODER_DIM];
    for frame_idx in 0..encoder_out_len {
        for d in 0..ENCODER_DIM {
            encoder_frame[d] = encoder_out[d * encoder_out_len + frame_idx];
        }
        let encoder_outputs = onnx::Value::from_slice(&onnx, &[1, ENCODER_DIM, 1], &encoder_frame)
            .map_err(|e| InferError::Runtime(format!("error creating encoder_outputs: {e}")))?;

        for _ in 0..MAX_SYMBOLS_PER_STEP {
            let state1_data = cache_state1
                .extract_tensor::<f32>()
                .map_err(|e| InferError::Runtime(format!("error extracting state1: {e}")))?
                .to_vec();
            let state2_data = cache_state2
                .extract_tensor::<f32>()
                .map_err(|e| InferError::Runtime(format!("error extracting state2: {e}")))?
                .to_vec();
            let targets = onnx::Value::from_slice(&onnx, &[1, 1], &[*cache_last_token as i32])
                .map_err(|e| InferError::Runtime(format!("error creating targets: {e}")))?;
            let target_length = onnx::Value::from_slice(&onnx, &[1], &[1i32])
                .map_err(|e| InferError::Runtime(format!("error creating target_length: {e}")))?;
            let mut outputs = decoder_joint
                .run(
                    &[
                        ("encoder_outputs", &encoder_outputs),
                        ("targets", &targets),
                        ("target_length", &target_length),
                        ("input_states_1", &cache_state1),
                        ("input_states_2", &cache_state2),
                    ],
                    &[
                        "outputs",
                        "prednet_lengths",
                        "output_states_1",
                        "output_states_2",
                    ],
                )
                .map_err(|e| InferError::Runtime(format!("decoder_joint inference failed: {e}")))?;
            let logits = outputs[0]
                .extract_tensor::<f32>()
                .map_err(|e| InferError::Runtime(format!("error extracting logits: {e}")))?
                .to_vec();

            *cache_state1 = outputs.remove(2);
            *cache_state2 = outputs.remove(2);

            let valid_range = &logits[..VOCAB_SIZE.min(logits.len())];
            let predicted_token_id = valid_range
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            if predicted_token_id == BLANK_ID {
                *cache_state1 =
                    onnx::Value::from_slice(&onnx, &[2, 1, DECODER_STATE_DIM], &state1_data)
                        .map_err(|e| InferError::Runtime(format!("error restoring state1: {e}")))?;
                *cache_state2 =
                    onnx::Value::from_slice(&onnx, &[2, 1, DECODER_STATE_DIM], &state2_data)
                        .map_err(|e| InferError::Runtime(format!("error restoring state2: {e}")))?;
                break;
            }

            token_ids.push(predicted_token_id);
            *cache_last_token = predicted_token_id;
        }
    }

    Ok(token_ids)
}

enum ParakeetCommand<T: Clone + Send + 'static> {
    Audio(AsrInput<T>),
    Flush { payload: T },
}

pub struct ParakeetHandle<T: Clone + Send + 'static> {
    input_tx: std_mpsc::Sender<ParakeetCommand<T>>,
}

pub struct ParakeetListener<T: Clone + Send + 'static> {
    output_rx: tokio_mpsc::Receiver<AsrOutput<T>>,
}

pub fn create<T: Clone + Send + 'static>(
    onnx: &Arc<onnx::Onnx>,
    executor: &onnx::Executor,
) -> Result<(ParakeetHandle<T>, ParakeetListener<T>), InferError> {
    // create encoder and decoder_joint sessions
    let mut encoder = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            PARAKEET_ENCODER_PATH,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create encoder session: {e}")))?;
    let mut decoder_joint = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            PARAKEET_DECODER_PATH,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create decoder_joint session: {e}")))?;

    // read streaming parameters from encoder metadata
    let metadata = encoder.metadata().unwrap_or_default();
    let encoder_window_size: usize = metadata
        .get("window_size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_ENCODER_WINDOW_SIZE);
    let encoder_chunk_shift: usize = metadata
        .get("chunk_shift")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_ENCODER_CHUNK_SHIFT);
    log_info!(
        "encoder streaming params: window_size={}, chunk_shift={}",
        encoder_window_size,
        encoder_chunk_shift
    );

    // load mel filterbank
    let data = std::fs::read(&MEL_FILTERBANK_PATH).map_err(|e| {
        InferError::Runtime(format!("Failed to read {}: {}", MEL_FILTERBANK_PATH, e))
    })?;
    let mel_filterbank: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if mel_filterbank.len() != NUM_MEL_BINS * SPECTRUM_BINS {
        return Err(InferError::Runtime(format!(
            "mel_filterbank.bin: expected {} floats, got {}",
            NUM_MEL_BINS * SPECTRUM_BINS,
            mel_filterbank.len()
        )));
    }

    // load Hann window
    let data = std::fs::read(&HANN_WINDOW_PATH)
        .map_err(|e| InferError::Runtime(format!("Failed to read {}: {}", HANN_WINDOW_PATH, e)))?;
    let hann_window: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if hann_window.len() != WINDOW_SIZE {
        return Err(InferError::Runtime(format!(
            "hann_window.bin: expected {} floats, got {}",
            WINDOW_SIZE,
            hann_window.len()
        )));
    }

    // load tokenizer
    let tokenizer_tokens = load_tokens(&PARAKEET_TOKENIZER_PATH)?;

    // create channels
    let (input_tx, input_rx) = std_mpsc::channel::<ParakeetCommand<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<AsrOutput<T>>(TEXT_CHANNEL_CAPACITY);

    // spawn processing task
    std::thread::spawn({
        let onnx = Arc::clone(&onnx);
        move || {
            // --- audio -> features state ---
            let mut audio_tail: Vec<i16> = Vec::new();
            let mut prev_sample: i16 = 0;

            // --- feature rolling buffer (frames-first [T, 128]) ---
            let mut feat_buf: Vec<f32> = Vec::new();
            let mut feat_buf_frames: usize = 0;

            // --- encoder cache ---
            let mut cache_last_channel = match zeros_f32(
                &onnx,
                &[
                    1,
                    NUM_LAYERS as i64,
                    CACHE_CHANNEL_CONTEXT as i64,
                    ENCODER_DIM as i64,
                ],
            ) {
                Ok(v) => v,
                Err(e) => {
                    log_error!("error creating cache_last_channel: {e}");
                    return;
                }
            };
            let mut cache_last_time = match zeros_f32(
                &onnx,
                &[
                    1,
                    NUM_LAYERS as i64,
                    ENCODER_DIM as i64,
                    CACHE_TIME_CONTEXT as i64,
                ],
            ) {
                Ok(v) => v,
                Err(e) => {
                    log_error!("error creating cache_last_time: {e}");
                    return;
                }
            };
            let mut cache_last_channel_len = match onnx::Value::from_slice(&onnx, &[1], &[0i64]) {
                Ok(v) => v,
                Err(e) => {
                    log_error!("error creating cache_last_channel_len: {e}");
                    return;
                }
            };

            // --- decoder cache ---
            let mut cache_state1 = match zeros_f32(&onnx, &[2, 1, DECODER_STATE_DIM as i64]) {
                Ok(v) => v,
                Err(e) => {
                    log_error!("error creating state1: {e}");
                    return;
                }
            };
            let mut cache_state2 = match zeros_f32(&onnx, &[2, 1, DECODER_STATE_DIM as i64]) {
                Ok(v) => v,
                Err(e) => {
                    log_error!("error creating state2: {e}");
                    return;
                }
            };
            let mut cache_last_token = BLANK_ID;

            // main audio loop
            while let Ok(command) = input_rx.recv() {
                let (audio, payload, is_flush) = match command {
                    ParakeetCommand::Audio(chunk) => (chunk.audio, chunk.payload, false),
                    ParakeetCommand::Flush { payload } => (Vec::new(), payload, true),
                };

                // Level 1: audio -> feature frames (frames-first)
                if !audio.is_empty() {
                    let new_features = audio_to_features(
                        &audio,
                        &mel_filterbank,
                        &hann_window,
                        &mut audio_tail,
                        &mut prev_sample,
                    );
                    let new_frames = new_features.len() / NUM_MEL_BINS;
                    if new_frames > 0 {
                        feat_buf.extend_from_slice(&new_features);
                        feat_buf_frames += new_frames;
                    }
                }

                // On flush: pad remaining features to encoder_window_size
                if is_flush && feat_buf_frames > 0 && feat_buf_frames < encoder_window_size {
                    let pad_frames = encoder_window_size - feat_buf_frames;
                    feat_buf.resize(feat_buf.len() + pad_frames * NUM_MEL_BINS, 0.0);
                    feat_buf_frames = encoder_window_size;
                }

                // Level 2: feed encoder whenever we have enough frames
                while feat_buf_frames >= encoder_window_size {
                    // Extract window_size frames and transpose to channels-first
                    let window_data = &feat_buf[..encoder_window_size * NUM_MEL_BINS];
                    let encoder_input = transpose_features(window_data, encoder_window_size);

                    // Run encoder
                    let (encoder_out, encoder_out_len) = match run_encoder(
                        &onnx,
                        &mut encoder,
                        &encoder_input,
                        encoder_window_size,
                        &mut cache_last_channel,
                        &mut cache_last_time,
                        &mut cache_last_channel_len,
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            log_error!("error running encoder: {e}");
                            break;
                        }
                    };

                    // Greedy decode
                    let token_ids = match greedy_decode(
                        &onnx,
                        &mut decoder_joint,
                        &encoder_out,
                        encoder_out_len,
                        &mut cache_state1,
                        &mut cache_state2,
                        &mut cache_last_token,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            log_error!("error decoding: {e}");
                            break;
                        }
                    };

                    let text = if !token_ids.is_empty() {
                        token_ids
                            .iter()
                            .filter_map(|&id| {
                                let idx = id as usize;
                                if idx >= tokenizer_tokens.len() {
                                    None
                                } else {
                                    Some(tokenizer_tokens[idx].replace('▁', " "))
                                }
                            })
                            .collect::<String>()
                    } else {
                        String::new()
                    };

                    // only send non-flush outputs if they have text (reduces noise)
                    let should_send = is_flush || !text.is_empty();
                    if should_send {
                        let output = AsrOutput::<T> {
                            payload: payload.clone(),
                            text,
                            is_flush: false, // per-window output, not the final flush
                        };
                        if let Err(e) = output_tx.blocking_send(output) {
                            log_error!("error sending text: {e}");
                            return;
                        }
                    }

                    // Shift buffer: keep last `feat_overlap` frames, drop chunk_shift
                    let shift_frames = encoder_chunk_shift.min(feat_buf_frames);
                    let shift_floats = shift_frames * NUM_MEL_BINS;
                    feat_buf.drain(..shift_floats);
                    feat_buf_frames -= shift_frames;
                }

                // After flush: send the flush marker and reset all state
                if is_flush {
                    let output = AsrOutput::<T> {
                        payload: payload.clone(),
                        text: String::new(),
                        is_flush: true,
                    };
                    if let Err(e) = output_tx.blocking_send(output) {
                        log_error!("error sending flush marker: {e}");
                        return;
                    }

                    // Reset state for next utterance
                    audio_tail.clear();
                    prev_sample = 0;
                    feat_buf.clear();
                    feat_buf_frames = 0;
                    cache_last_channel = match zeros_f32(
                        &onnx,
                        &[
                            1,
                            NUM_LAYERS as i64,
                            CACHE_CHANNEL_CONTEXT as i64,
                            ENCODER_DIM as i64,
                        ],
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            log_error!("error resetting cache_last_channel: {e}");
                            return;
                        }
                    };
                    cache_last_time = match zeros_f32(
                        &onnx,
                        &[
                            1,
                            NUM_LAYERS as i64,
                            ENCODER_DIM as i64,
                            CACHE_TIME_CONTEXT as i64,
                        ],
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            log_error!("error resetting cache_last_time: {e}");
                            return;
                        }
                    };
                    cache_last_channel_len = match onnx::Value::from_slice(&onnx, &[1], &[0i64]) {
                        Ok(v) => v,
                        Err(e) => {
                            log_error!("error resetting cache_last_channel_len: {e}");
                            return;
                        }
                    };
                    cache_state1 = match zeros_f32(&onnx, &[2, 1, DECODER_STATE_DIM as i64]) {
                        Ok(v) => v,
                        Err(e) => {
                            log_error!("error resetting state1: {e}");
                            return;
                        }
                    };
                    cache_state2 = match zeros_f32(&onnx, &[2, 1, DECODER_STATE_DIM as i64]) {
                        Ok(v) => v,
                        Err(e) => {
                            log_error!("error resetting state2: {e}");
                            return;
                        }
                    };
                    cache_last_token = BLANK_ID;
                }

                // Skip sending output for non-flush commands with no new features
                if !is_flush && audio.is_empty() {
                    continue;
                }
            }
        }
    });

    Ok((ParakeetHandle { input_tx }, ParakeetListener { output_rx }))
}

impl<T: Clone + Send + 'static> ParakeetHandle<T> {
    /// Send audio chunk to ASR for streaming processing.
    pub fn send(&self, input: AsrInput<T>) -> Result<(), InferError> {
        self.input_tx
            .send(ParakeetCommand::Audio(input))
            .map_err(|_| InferError::Runtime("ASR channel closed".to_string()))
    }

    /// Flush remaining buffered audio through the encoder, producing any final text.
    /// Resets all ASR state for the next utterance.
    pub fn flush(&self, payload: T) -> Result<(), InferError> {
        self.input_tx
            .send(ParakeetCommand::Flush { payload })
            .map_err(|_| InferError::Runtime("ASR channel closed".to_string()))
    }
}

impl<T: Clone + Send + 'static> ParakeetListener<T> {
    // receive transcription chunk from ASR
    pub async fn recv(&mut self) -> Option<AsrOutput<T>> {
        self.output_rx.recv().await
    }

    // try-receive transcription chunk from ASR
    pub fn try_recv(&mut self) -> Option<AsrOutput<T>> {
        match self.output_rx.try_recv() {
            Ok(output) => Some(output),
            _ => None,
        }
    }
}
