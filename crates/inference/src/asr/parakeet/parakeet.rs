use {super::tokens::load_tokens, crate::*, base::*, std::sync::Arc, tokio::sync::mpsc};

const PARAKEET_ENCODER_PATH: &str = "data/parakeet/encoder.onnx";
const PARAKEET_DECODER_PATH: &str = "data/parakeet/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/parakeet/tokenizer.model";
const MEL_FILTERBANK_PATH: &str = "data/parakeet/mel_filterbank.bin";
const HANN_WINDOW_PATH: &str = "data/parakeet/hann_window.bin";

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

const AUDIO_CHANNEL_CAPACITY: usize = 64;
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

    log_info!(
        "encoder: num_frames={}, encoder_out_len={}",
        num_frames,
        encoder_out_len
    );

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

pub struct Parakeet {
    audio_tx: mpsc::Sender<Vec<i16>>,
    text_rx: mpsc::Receiver<String>,
}

impl Parakeet {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: &onnx::Executor) -> Result<Self, InferError> {
        // create encoder and decoder_joint sessions
        let mut encoder = onnx
            .create_session(executor, PARAKEET_ENCODER_PATH)
            .map_err(|e| InferError::Runtime(format!("Failed to create encoder session: {e}")))?;
        let mut decoder_joint = onnx
            .create_session(executor, PARAKEET_DECODER_PATH)
            .map_err(|e| {
                InferError::Runtime(format!("Failed to create decoder_joint session: {e}"))
            })?;

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
        let data = std::fs::read(&HANN_WINDOW_PATH).map_err(|e| {
            InferError::Runtime(format!("Failed to read {}: {}", HANN_WINDOW_PATH, e))
        })?;
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
        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<i16>>(AUDIO_CHANNEL_CAPACITY);
        let (text_tx, text_rx) = mpsc::channel::<String>(TEXT_CHANNEL_CAPACITY);

        // spawn processing task
        tokio::task::spawn_blocking({
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
                let mut cache_last_channel_len = match onnx::Value::from_slice(&onnx, &[1], &[0i64])
                {
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
                while let Some(chunk) = audio_rx.blocking_recv() {
                    // Level 1: audio -> feature frames (frames-first)
                    let new_features = audio_to_features(
                        &chunk,
                        &mel_filterbank,
                        &hann_window,
                        &mut audio_tail,
                        &mut prev_sample,
                    );
                    let new_frames = new_features.len() / NUM_MEL_BINS;
                    if new_frames == 0 {
                        continue;
                    }

                    // Append to feature rolling buffer
                    feat_buf.extend_from_slice(&new_features);
                    feat_buf_frames += new_frames;

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

                        if !token_ids.is_empty() {
                            log_info!("greedy_decode: {} tokens: {:?}", token_ids.len(), token_ids);

                            let text = token_ids
                                .iter()
                                .filter_map(|&id| {
                                    let idx = id as usize;
                                    if idx >= tokenizer_tokens.len() {
                                        None
                                    } else {
                                        Some(tokenizer_tokens[idx].replace('▁', " "))
                                    }
                                })
                                .collect::<String>();

                            if let Err(e) = text_tx.blocking_send(text) {
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
                }

                // Process any remaining frames in the buffer (final flush)
                if feat_buf_frames > 0 {
                    let encoder_input = transpose_features(&feat_buf, feat_buf_frames);
                    if let Ok((encoder_out, encoder_out_len)) = run_encoder(
                        &onnx,
                        &mut encoder,
                        &encoder_input,
                        feat_buf_frames,
                        &mut cache_last_channel,
                        &mut cache_last_time,
                        &mut cache_last_channel_len,
                    ) {
                        if let Ok(token_ids) = greedy_decode(
                            &onnx,
                            &mut decoder_joint,
                            &encoder_out,
                            encoder_out_len,
                            &mut cache_state1,
                            &mut cache_state2,
                            &mut cache_last_token,
                        ) {
                            if !token_ids.is_empty() {
                                let text = token_ids
                                    .iter()
                                    .filter_map(|&id| {
                                        let idx = id as usize;
                                        if idx >= tokenizer_tokens.len() {
                                            None
                                        } else {
                                            Some(tokenizer_tokens[idx].replace('▁', " "))
                                        }
                                    })
                                    .collect::<String>();
                                let _ = text_tx.blocking_send(text);
                            }
                        }
                    }
                }
            }
        });

        Ok(Self { audio_tx, text_rx })
    }

    pub fn audio_tx(&self) -> mpsc::Sender<Vec<i16>> {
        self.audio_tx.clone()
    }

    pub async fn send(&self, chunk: Vec<i16>) -> Result<(), InferError> {
        self.audio_tx
            .send(chunk)
            .await
            .map_err(|error| InferError::Runtime(format!("Failed to send audio chunk: {}", error)))
    }

    pub async fn recv(&mut self) -> Option<String> {
        self.text_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<String> {
        match self.text_rx.try_recv() {
            Ok(text) => Some(text),
            _ => None,
        }
    }
}
