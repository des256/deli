use {
    audio::*,
    base::*,
    inference::*,
    std::{
        collections::VecDeque,
        io::Write,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        time::{Duration, Instant},
    },
};

const VOICE_PATH: &str = "data/tts/pocket/voices/stephen.bin";
const VAD_FRAME_SIZE: usize = 512;
const ASR_SAMPLE_RATE: usize = 16000;
const TTS_SAMPLE_RATE: usize = 24000;

// VAD tuning
const VAD_THRESHOLD: f32 = 0.5;
const VAD_COOLDOWN_FRAMES: u32 = 15; // ~480ms of silence before utterance ends

// Smaller chunks = lower latency for VAD detection.
// 512 * 4 = 2048 samples = 128ms at 16kHz, giving 4 clean VAD frames per chunk.
const AUDIOIN_CHUNK_SIZE: usize = VAD_FRAME_SIZE * 4;

// Number of recent audio chunks to keep as pre-roll.
// When VAD detects speech start, these are flushed to ASR first to capture the onset
// that VAD needed to make its decision. 3 chunks = ~384ms at 128ms/chunk.
const VAD_PREROLL_CHUNKS: usize = 3;

/// Payload that carries the speech-end timestamp through the pipeline.
/// The `id` field holds different IDs at each stage (utterance_id, sentence_id).
#[derive(Clone, Debug)]
struct ChatPayload {
    id: u64,
    speech_end_ms: u64,
    text: String,
}

/// VAD state machine for speech segmentation.
enum VadPhase {
    /// No speech detected. Waiting for speech to start.
    Idle,
    /// Speech is active. Audio is being accumulated for ASR.
    Speaking,
    /// Speech may have ended. Counting down silence frames before confirming.
    Cooldown(u32),
}

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    // initialize inference
    let inference = Inference::new().map_err(|e| InferError::Runtime(e.to_string()))?;

    // initialize AudioIn (smaller chunks for lower VAD latency)
    log_info!("Opening audio input...");
    let mut audioin_listener = create_audioin(Some(AudioInConfig {
        device_name: Some(
            "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.mono-fallback".to_string(),
        ),
        sample_rate: ASR_SAMPLE_RATE,
        chunk_size: AUDIOIN_CHUNK_SIZE,
        boost: 4,
    }))
    .await;

    // shared epoch for pipeline cancellation
    let epoch = Epoch::new();

    // initialize AudioOut
    log_info!("Opening audio output...");
    let (audioout_handle, mut audioout_listener) = audio::create_audioout::<TtsPayload<ChatPayload>>(
        Some(AudioOutConfig {
            device_name: Some(
                "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo".to_string(),
            ),
            sample_rate: TTS_SAMPLE_RATE,
        }),
        epoch.clone(),
    );
    let audioout_handle = Arc::new(audioout_handle);

    println!(">> {}", inference.mem_info());

    // load VAD
    log_info!("Loading VAD...");
    let mut vad = inference.use_silero(&onnx::Executor::Cpu, ASR_SAMPLE_RATE)?;
    println!(">> {}", inference.mem_info());

    // load ASR
    log_info!("Loading ASR...");
    let (asr_handle, mut asr_listener) = inference.use_parakeet::<u64>(&onnx::Executor::Cuda(0))?;
    let asr_handle = Arc::new(asr_handle);
    println!(">> {}", inference.mem_info());

    // load LLM
    log_info!("Loading LLM...");
    let (llm_handle, mut llm_listener) =
        inference.use_phi3::<ChatPayload>(&onnx::Executor::Cuda(0), epoch.clone())?;
    //inference.use_gemma3::<ChatPayload>(&onnx::Executor::Cuda(0), epoch.clone())?;
    //inference.use_llama3_3b::<ChatPayload>(&onnx::Executor::Cuda(0), epoch.clone())?;
    //inference.use_llama3_8b::<ChatPayload>(&onnx::Executor::Cuda(0), epoch.clone())?;
    let llm_handle = Arc::new(llm_handle);
    println!(">> {}", inference.mem_info());

    // load TTS
    log_info!("Loading TTS...");
    let (tts_handle, mut tts_listener) =
        inference.use_pocket::<ChatPayload>(&onnx::Executor::Cpu, &VOICE_PATH, epoch.clone())?;
    let tts_handle = Arc::new(tts_handle);
    println!(">> {}", inference.mem_info());

    // global clock
    let global_start = Instant::now();

    // shared timestamp: when VAD last signaled speech end (ms from global_start, 0 = unset)
    let speech_end_ms = Arc::new(AtomicU64::new(0));

    // chat history
    let history = Arc::new(History::new());

    // task: AudioIn + VAD pump
    // VAD gates audio to ASR: only speech segments (plus pre-roll) are sent.
    // VAD state machine controls epoch advancement (on speech start) and ASR flush (on speech end).
    log_info!("Spawning AudioIn+VAD pump...");
    tokio::spawn({
        let asr_handle = Arc::clone(&asr_handle);
        let epoch = epoch.clone();
        let speech_end_ms = Arc::clone(&speech_end_ms);
        async move {
            let mut source_audio_id = 0u64;
            let mut phase = VadPhase::Idle;
            let mut pre_roll: VecDeque<Vec<i16>> = VecDeque::with_capacity(VAD_PREROLL_CHUNKS + 1);

            while let Some(audio) = audioin_listener.recv().await {
                let now = global_start.elapsed();

                // Process VAD first to determine whether this chunk should go to ASR
                let mut speech_started = false;
                let mut speech_ended = false;

                for chunk in audio.chunks_exact(VAD_FRAME_SIZE) {
                    let prob = vad.process(chunk).unwrap_or(0.0);

                    match &mut phase {
                        VadPhase::Idle => {
                            if prob > VAD_THRESHOLD {
                                // Speech started — cancel any playing response, start new utterance
                                let new_epoch = epoch.advance();
                                println!(
                                    "[{:010}ms] VAD SA{:04}: speech start (prob={:.2}%, epoch={})",
                                    now.as_millis(),
                                    source_audio_id,
                                    prob * 100.0,
                                    new_epoch,
                                );
                                phase = VadPhase::Speaking;
                                speech_started = true;
                            }
                        }
                        VadPhase::Speaking => {
                            if prob < VAD_THRESHOLD {
                                // Silence detected — start cooldown
                                phase = VadPhase::Cooldown(VAD_COOLDOWN_FRAMES);
                            }
                        }
                        VadPhase::Cooldown(remaining) => {
                            if prob > VAD_THRESHOLD {
                                // Speech resumed — back to speaking (same utterance)
                                phase = VadPhase::Speaking;
                            } else {
                                *remaining -= 1;
                                if *remaining == 0 {
                                    // Cooldown expired — utterance complete, flush ASR
                                    println!(
                                        "[{:010}ms] VAD SA{:04}: speech end, flushing ASR",
                                        now.as_millis(),
                                        source_audio_id,
                                    );
                                    speech_end_ms.store(now.as_millis() as u64, Ordering::Release);
                                    speech_ended = true;
                                    phase = VadPhase::Idle;
                                }
                            }
                        }
                    }
                }

                // Send pre-roll buffer to ASR on speech onset (captures the beginning
                // of the utterance that VAD needed to make its detection decision)
                if speech_started {
                    for pre_chunk in pre_roll.drain(..) {
                        if let Err(error) = asr_handle.send(AsrInput {
                            payload: source_audio_id,
                            audio: pre_chunk,
                        }) {
                            log_error!("ASR pre-roll send failed: {}", error);
                        }
                    }
                }

                // Send audio to ASR only during speech (Speaking, Cooldown, or the
                // chunk where speech just ended — it still contains trailing speech)
                let in_speech =
                    matches!(phase, VadPhase::Speaking | VadPhase::Cooldown(_)) || speech_ended;

                if in_speech || speech_started {
                    if let Err(error) = asr_handle.send(AsrInput {
                        payload: source_audio_id,
                        audio,
                    }) {
                        log_error!("ASR send failed: {}", error);
                    }
                } else {
                    // Idle — buffer for pre-roll
                    if pre_roll.len() >= VAD_PREROLL_CHUNKS {
                        pre_roll.pop_front();
                    }
                    pre_roll.push_back(audio);
                }

                // Flush ASR after sending the final chunk (so it processes everything first)
                if speech_ended {
                    if let Err(error) = asr_handle.flush(source_audio_id) {
                        log_error!("ASR flush failed: {}", error);
                    }
                }

                source_audio_id += 1;
            }
        }
    });

    // task: TTS -> AudioOut pump
    // Forwards TTS audio directly to AudioOut. Epoch checking drops stale audio.
    // No gate needed — during speech, LLM/TTS are idle (no output to buffer).
    // When user interrupts mid-response, epoch advance cancels playback.
    log_info!("Spawning TTS->AudioOut pump...");
    tokio::spawn({
        let audioout_handle = Arc::clone(&audioout_handle);
        let epoch = epoch.clone();
        async move {
            while let Some(stamped) = tts_listener.recv().await {
                // drop stale chunks
                if !epoch.is_current(stamped.epoch) {
                    continue;
                }

                let _now = global_start.elapsed();
                let output = stamped.inner;
                let chunk = AudioOutChunk {
                    payload: output.payload,
                    data: output.data,
                };
                //println!(
                //    "[{:010}ms] TTS LL{:04}({:04}): -> AudioOut",
                //    now.as_millis(),
                //    chunk.payload.payload,
                //    chunk.payload.id,
                //);
                if let Err(error) = audioout_handle.send(chunk) {
                    log_error!("AudioOut send failed: {}", error);
                }
            }
        }
    });

    // task: AudioOut status pump
    log_info!("Spawning AudioOut pump...");
    tokio::spawn({
        let global_start = global_start;
        let history = Arc::clone(&history);
        async move {
            let mut current_index = 0usize;
            let mut current_sentence_id = u64::MAX;
            let mut sentence_text = String::new();
            let mut samples_per_char = 0.0f64;
            let mut thinking_reported_for_ms = 0u64;
            while let Some(status) = audioout_listener.recv().await {
                let now = global_start.elapsed();
                match status {
                    AudioOutStatus::Started(payload) => {
                        if current_sentence_id != payload.payload.id {
                            current_sentence_id = payload.payload.id;
                            current_index = 0;
                            sentence_text = payload.payload.text.clone();
                        }
                        // Report thinking time on the first audio chunk after each speech end
                        let end_ms = payload.payload.speech_end_ms;
                        if end_ms > 0 && end_ms != thinking_reported_for_ms {
                            let now_ms = now.as_millis() as u64;
                            println!(">> thinking time: {}ms", now_ms.saturating_sub(end_ms));
                            thinking_reported_for_ms = end_ms;
                        }
                    }
                    AudioOutStatus::Finished { payload, index } => {
                        if current_sentence_id == payload.payload.id {
                            current_index += index;
                            if payload.last {
                                history
                                    .add_entry(Entry {
                                        timestamp: now.as_millis() as u64,
                                        speaker: Speaker::Model,
                                        sentence: sentence_text.clone(),
                                    })
                                    .await;
                                if sentence_text.len() > 10 {
                                    samples_per_char =
                                        current_index as f64 / sentence_text.len() as f64;
                                }
                            }
                        }
                    }
                    AudioOutStatus::Canceled { payload, index } => {
                        if current_sentence_id == payload.payload.id {
                            current_index += index;
                            let truncated: String = if samples_per_char > 0.0 {
                                let chars_played =
                                    (current_index as f64 / samples_per_char).round() as usize;
                                sentence_text.chars().take(chars_played).collect()
                            } else {
                                sentence_text.clone()
                            };
                            history
                                .add_entry(Entry {
                                    timestamp: now.as_millis() as u64,
                                    speaker: Speaker::Model,
                                    sentence: format!("{}...", truncated),
                                })
                                .await;
                        }
                    }
                }
            }
        }
    });

    // task: ASR pump
    // Accumulates ASR text during an utterance. On flush (speech end), sends the
    // complete utterance to LLM as a single prompt. Uses epoch to detect new
    // utterances and discard stale text.
    log_info!("Spawning ASR pump...");
    tokio::spawn({
        let llm_handle = Arc::clone(&llm_handle);
        let epoch = epoch.clone();
        let speech_end_ms = Arc::clone(&speech_end_ms);
        let history = Arc::clone(&history);
        async move {
            let mut accumulator = String::new();
            let mut current_epoch = epoch.current();
            let mut utterance_id = 0u64;

            while let Some(chunk) = asr_listener.recv().await {
                let now = global_start.elapsed();

                // Detect new utterance (epoch advanced by VAD pump)
                let now_epoch = epoch.current();
                if now_epoch != current_epoch {
                    accumulator.clear();
                    current_epoch = now_epoch;
                    utterance_id += 1;
                }

                // Log ASR output
                if !chunk.text.is_empty() {
                    println!(
                        "[{:010}ms] ASR SA{:04}: \"{}\"{}",
                        now.as_millis(),
                        chunk.payload,
                        chunk.text,
                        if chunk.is_flush { " (flush)" } else { "" },
                    );
                }

                // Accumulate text
                if chunk.text.chars().any(|c| c.is_alphanumeric()) {
                    accumulator.push_str(&chunk.text);
                }

                // On flush: send accumulated text to LLM
                if chunk.is_flush {
                    let text = accumulator.trim().to_string();
                    accumulator.clear();

                    if text.is_empty() {
                        println!(
                            "[{:010}ms] ASR flush: empty utterance, skipping LLM",
                            now.as_millis(),
                        );
                        continue;
                    }

                    println!(
                        "[{:010}ms] ASR flush UT{:04}: \"{}\" -> LLM",
                        now.as_millis(),
                        utterance_id,
                        text,
                    );

                    history
                        .add_entry(Entry {
                            timestamp: now.as_millis() as u64,
                            speaker: Speaker::User,
                            sentence: text.clone(),
                        })
                        .await;

                    let prompt = llm_handle.format_prompt(&history, 5).await;
                    println!(">> prompt: {}", prompt);
                    if let Err(error) = llm_handle.send(LlmInput {
                        payload: ChatPayload {
                            id: utterance_id,
                            speech_end_ms: speech_end_ms.load(Ordering::Acquire),
                            text: String::new(),
                        },
                        prompt,
                    }) {
                        log_error!("LLM send failed: {}", error);
                    }
                }
            }
        }
    });

    // task: LLM pump
    // Splits LLM token stream into sentences and sends each to TTS immediately.
    log_info!("Spawning LLM pump...");
    tokio::spawn({
        let tts_handle = Arc::clone(&tts_handle);
        let epoch = epoch.clone();
        async move {
            let mut sentence = String::new();
            let mut current_epoch = 0u64;
            let mut llm_id = 0u64;
            while let Some(stamped) = llm_listener.recv().await {
                let now = global_start.elapsed();

                // epoch changed — clear stale sentence buffer
                if stamped.epoch != current_epoch {
                    sentence.clear();
                    current_epoch = stamped.epoch;
                }

                // drop stale tokens
                if !epoch.is_current(stamped.epoch) {
                    continue;
                }

                match stamped.inner {
                    LlmOutput::Token { payload, token } => {
                        sentence.push_str(&token);
                        if ends_with_sentence_boundary(&sentence) {
                            let trimmed = sentence.trim().to_string();
                            if !trimmed.is_empty() {
                                println!(
                                    "[{:010}ms] LLM UT{:04}: \"{}\" -> TTS LL{:04}",
                                    now.as_millis(),
                                    payload.id,
                                    trimmed,
                                    llm_id,
                                );
                                let padded = format!("   {}     ", trimmed);
                                if let Err(error) = tts_handle.send(TtsInput {
                                    payload: ChatPayload {
                                        id: llm_id,
                                        speech_end_ms: payload.speech_end_ms,
                                        text: trimmed,
                                    },
                                    text: padded,
                                }) {
                                    log_error!("TTS send failed: {}", error);
                                }
                                llm_id += 1;
                            }
                            sentence.clear();
                        }
                    }
                    LlmOutput::Eos { payload } => {
                        let trimmed = sentence.trim().to_string();
                        if !trimmed.is_empty() {
                            println!(
                                "[{:010}ms] LLM UT{:04}: \"{}\" -> TTS LL{:04} (EOS)",
                                now.as_millis(),
                                payload.id,
                                trimmed,
                                llm_id,
                            );
                            if let Err(error) = tts_handle.send(TtsInput {
                                payload: ChatPayload {
                                    id: llm_id,
                                    speech_end_ms: payload.speech_end_ms,
                                    text: trimmed.clone(),
                                },
                                text: trimmed,
                            }) {
                                log_error!("TTS send failed: {}", error);
                            }
                            llm_id += 1;
                        } else {
                            println!("[{:010}ms] LLM: EOS", now.as_millis());
                        }
                        sentence.clear();
                    }
                }
            }
        }
    });

    log_info!("Chat ready. Ctrl+D to exit.");
    loop {
        print!("> ");
        std::io::stdout()
            .flush()
            .map_err(|e| InferError::Runtime(e.to_string()))?;
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| InferError::Runtime(e.to_string()))?;
        if input.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}

/// Check if the buffer ends at a sentence boundary.
/// Matches common sentence-ending punctuation: . ! ? and also : ;
fn ends_with_sentence_boundary(text: &str) -> bool {
    let trimmed = text.trim_end();
    matches!(
        trimmed.as_bytes().last(),
        Some(b'.' | b'!' | b'?' | b':' | b';')
    )
}
