use {
    audio::*,
    base::*,
    com::*,
    image::*,
    inference::*,
    std::{
        sync::Arc,
        time::{Duration, Instant},
    },
    testy::*,
    video::*,
};

const DEFAULT_ADDR: &str = "0.0.0.0:5090";
const ASR_SAMPLE_RATE: usize = 16000;
const VAD_THRESHOLD: f32 = 0.5;
const VAD_COOLDOWN_FRAMES: u32 = 15;
const VAD_FRAME_SIZE: usize = 512;
const AUDIOIN_CHUNK_SIZE: usize = VAD_FRAME_SIZE * 4;
const TTS_SAMPLE_RATE: usize = 24000;
const VOICE_PATH: &str = "data/tts/pocket/voices/hannah.bin";

enum VadPhase {
    Idle,
    Speaking,
    Cooldown(u32),
}

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();
    let global_start = Instant::now();

    log_info!("initializing inference");
    let inference = Inference::new().map_err(|e| InferError::Runtime(e.to_string()))?;

    let epoch = Epoch::new();

    log_info!("opening audio input");
    let mut audioin_listener = create_audioin(Some(AudioInConfig {
        sample_rate: ASR_SAMPLE_RATE,
        chunk_size: AUDIOIN_CHUNK_SIZE,
        boost: 4,
        ..Default::default()
    }))
    .await;

    log_info!("opening audio output");
    let (audioout_handle, mut audioout_listener) = audio::create_audioout::<TtsPayload<u64>>(
        Some(AudioOutConfig {
            sample_rate: TTS_SAMPLE_RATE,
            ..Default::default()
        }),
        epoch.clone(),
    );
    let audioout_handle = Arc::new(audioout_handle);

    log_info!("opening video input");
    let mut videoin = realsense::create(Some(realsense::RealsenseConfig {
        color: Some(Vec2::new(640, 480)),
        frame_rate: Some(30.0),
        ..Default::default()
    }))
    .map_err(|e| InferError::Runtime(e.to_string()))?;

    log_info!("opening websocket server at {}", DEFAULT_ADDR);
    let server: Arc<WsServer<ToMonitor>> = Arc::new(
        WsServer::bind(DEFAULT_ADDR)
            .await
            .map_err(|e| InferError::Runtime(e.to_string()))?,
    );

    log_info!("loading VAD");
    let mut vad = inference.use_silero(&onnx::Executor::Cpu, ASR_SAMPLE_RATE)?;

    log_info!("loading ASR");
    let (asr_handle, mut asr_listener) = inference.use_parakeet::<u64>(&onnx::Executor::Cuda(0))?;
    let asr_handle = Arc::new(asr_handle);

    log_info!("loading LLM");
    let (llm_handle, mut llm_listener) =
        inference.use_gemma3::<u64>(&onnx::Executor::Cuda(0), epoch.clone())?;
    let llm_handle = Arc::new(llm_handle);

    log_info!("loading TTS");
    let (tts_handle, mut tts_listener) =
        inference.use_pocket::<u64>(&onnx::Executor::Cpu, &VOICE_PATH, epoch.clone())?;
    let tts_handle = Arc::new(tts_handle);

    log_info!("spawning AudioIn pump");
    tokio::spawn({
        let asr_handle = Arc::clone(&asr_handle);
        let epoch = epoch.clone();
        async move {
            let mut source_audio_id = 0u64;
            let mut phase = VadPhase::Idle;
            while let Some(audio) = audioin_listener.recv().await {
                let now = global_start.elapsed();
                if let Err(error) = asr_handle.send(AsrInput {
                    payload: source_audio_id,
                    audio: audio.clone(),
                }) {
                    log_error!("ASR send failed: {}", error);
                }
                for chunk in audio.chunks_exact(VAD_FRAME_SIZE) {
                    let probability = vad.process(chunk).unwrap_or(0.0);
                    match &mut phase {
                        VadPhase::Idle => {
                            if probability > VAD_THRESHOLD {
                                epoch.advance();
                                println!(
                                    "[{:010}ms] VAD SA{:04}: speech start, advance epoch",
                                    now.as_millis(),
                                    source_audio_id
                                );
                                phase = VadPhase::Speaking;
                            }
                        }
                        VadPhase::Speaking => {
                            if probability < VAD_THRESHOLD {
                                phase = VadPhase::Cooldown(VAD_COOLDOWN_FRAMES);
                            }
                        }
                        VadPhase::Cooldown(remaining) => {
                            if probability > VAD_THRESHOLD {
                                phase = VadPhase::Speaking;
                            } else {
                                *remaining -= 1;
                                if *remaining == 0 {
                                    println!(
                                        "[{:010}ms] VAD SA{:04}: speech end, flush ASR",
                                        now.as_millis(),
                                        source_audio_id
                                    );
                                    if let Err(error) = asr_handle.flush(source_audio_id) {
                                        log_error!("ASR flush failed: {}", error);
                                    }
                                    phase = VadPhase::Idle;
                                }
                            }
                        }
                    }
                }
                source_audio_id += 1;
            }
        }
    });

    log_info!("spawning ASR pump");
    tokio::spawn({
        let llm_handle = Arc::clone(&llm_handle);
        let epoch = epoch.clone();
        async move {
            let mut accumulator = String::new();
            let mut current_epoch = epoch.current();
            let mut utterance_id = 0u64;
            while let Some(utterance) = asr_listener.recv().await {
                let now = global_start.elapsed();
                let now_epoch = epoch.current();
                if now_epoch != current_epoch {
                    accumulator.clear();
                    current_epoch = now_epoch;
                    utterance_id += 1;
                }
                if !utterance.text.is_empty() {
                    println!(
                        "[{:010}ms] ASR SA{:04}: \"{}\"{}",
                        now.as_millis(),
                        utterance.payload,
                        utterance.text,
                        if utterance.is_flush { " (flush)" } else { "" },
                    );
                }
                if utterance.text.chars().any(|c| c.is_alphanumeric()) {
                    accumulator.push_str(&utterance.text);
                }
                if utterance.is_flush {
                    let text = accumulator.trim().to_string();
                    accumulator.clear();
                    if text.is_empty() {
                        println!(
                            "[{:010}ms] ASR flush: empty utterance, skip LLM",
                            now.as_millis()
                        );
                        continue;
                    }
                    println!(
                        "[{:010}ms] ASR flush UT{:04}: prompt LLM \"{}\"",
                        now.as_millis(),
                        utterance_id,
                        text
                    );
                    let prompt = format!(
                        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                        text
                    );
                    if let Err(error) = llm_handle.send(LlmInput {
                        payload: utterance_id,
                        prompt,
                    }) {
                        log_error!("LLM send failed: {}", error);
                    }
                }
            }
        }
    });

    log_info!("spawning LLM pump");
    tokio::spawn({
        let tts_handle = Arc::clone(&tts_handle);
        let epoch = epoch.clone();
        async move {
            let mut sentence = String::new();
            let mut current_epoch = epoch.current();
            let mut response_id = 0u64;
            while let Some(token) = llm_listener.recv().await {
                let now = global_start.elapsed();
                if token.epoch != current_epoch {
                    sentence.clear();
                    current_epoch = token.epoch;
                }
                if !epoch.is_current(token.epoch) {
                    continue;
                }
                match token.inner {
                    LlmOutput::Token { payload, token } => {
                        sentence.push_str(&token);
                        if ends_with_sentence_boundary(&sentence) {
                            let trimmed = sentence.trim().to_string();
                            if !trimmed.is_empty() {
                                println!(
                                    "[{:010}ms] LLM UT{:04}: send to TTS: \"{}\" (RS{:04})",
                                    now.as_millis(),
                                    payload,
                                    trimmed,
                                    response_id
                                );
                                if let Err(error) = tts_handle.send(TtsInput {
                                    payload: response_id,
                                    text: trimmed,
                                }) {
                                    log_error!("TTS send failed: {}", error);
                                }
                                response_id += 1;
                            }
                            sentence.clear();
                        }
                    }
                    LlmOutput::Eos { payload } => {
                        let trimmed = sentence.trim().to_string();
                        if !trimmed.is_empty() {
                            println!(
                                "[{:010}ms] LLM UT{:04}: send to TTS: \"{}\" (RS{:04})",
                                now.as_millis(),
                                payload,
                                trimmed,
                                response_id
                            );
                            if let Err(error) = tts_handle.send(TtsInput {
                                payload: response_id,
                                text: trimmed,
                            }) {
                                log_error!("TTS send failed: {}", error);
                            }
                            response_id += 1;
                        } else {
                            println!("[{:010}ms] LLM UT{:04}: EOS", now.as_millis(), payload);
                        }
                    }
                }
            }
        }
    });

    log_info!("spawning TTS pump");
    tokio::spawn({
        let audioout_handle = Arc::clone(&audioout_handle);
        let epoch = epoch.clone();
        async move {
            while let Some(audio) = tts_listener.recv().await {
                if !epoch.is_current(audio.epoch) {
                    continue;
                }
                let now = global_start.elapsed();
                let output = audio.inner;
                let chunk = AudioOutChunk {
                    payload: output.payload,
                    data: output.data,
                };
                println!(
                    "[{:010}ms] TTS: sound {:04}:{:04}",
                    now.as_millis(),
                    chunk.payload.payload,
                    chunk.payload.id
                );
                if let Err(error) = audioout_handle.send(chunk) {
                    log_error!("AudioOut send failed: {}", error);
                }
            }
        }
    });

    log_info!("spawning AudioOut status pump");
    tokio::spawn({
        async move {
            let mut current_index = 0usize;
            let mut current_sentence_id = 0u64;
            while let Some(status) = audioout_listener.recv().await {
                let now = global_start.elapsed();
                match status {
                    AudioOutStatus::Started(payload) => {
                        if current_sentence_id != payload.payload {
                            current_sentence_id = payload.payload;
                            current_index = 0;
                        }
                    }
                    AudioOutStatus::Finished { payload, index } => {
                        if current_sentence_id == payload.payload {
                            current_index += index;
                        }
                        println!(
                            "[{:010}ms] AudioOut: finished {:04}:{:04}",
                            now.as_millis(),
                            current_sentence_id,
                            payload.id
                        );
                    }
                    AudioOutStatus::Canceled { payload, index } => {
                        if current_sentence_id == payload.payload {
                            current_index += index;
                            println!(
                                "[{:010}ms] AudioOut: canceled {:04}:{:04} at {} ({:.2}s)",
                                now.as_millis(),
                                current_sentence_id,
                                payload.id,
                                current_index,
                                current_index as f32 / TTS_SAMPLE_RATE as f32,
                            );
                        }
                    }
                }
            }
        }
    });

    /*
    log_info!("spawning video pump");
    tokio::spawn({
        let server = Arc::clone(&server);
        async move {
            // TODO: detect poses from the video stream
            // TODO: magically map poses to humans without any problems
            // TODO: for now just pass the color frame to the websocket clients
            loop {
                let frame = match videoin.recv().await {
                    Some(frame) => frame,
                    None => {
                        log_error!("video capture failed");
                        continue;
                    }
                };
                let jpeg =
                    color_to_jpeg(frame.color.size, &frame.color.data, frame.color.format, 80);
                if let Err(error) = server.send(&ToMonitor::Jpeg(jpeg)).await {
                    log_error!("websocket send failed: {}", error);
                    continue;
                }
            }
        }
    });
    */

    println!("running. Ctrl-C to exit.");
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    Ok(())
}

fn ends_with_sentence_boundary(text: &str) -> bool {
    let trimmed = text.trim_end();
    matches!(
        trimmed.as_bytes().last(),
        Some(b'.' | b'!' | b'?' | b':' | b';')
    )
}

fn color_to_jpeg(size: Vec2<usize>, data: &[u8], format: PixelFormat, quality: u8) -> Vec<u8> {
    match format {
        PixelFormat::Jpeg => data.to_vec(),
        PixelFormat::Yuyv => yuyv_to_jpeg(size, data, quality),
        PixelFormat::Srggb10p => srggb10p_to_jpeg(size, data, quality),
        PixelFormat::Yu12 => yu12_to_jpeg(size, data, quality),
        PixelFormat::Rgb8 => rgb_to_jpeg(size, data, quality),
        PixelFormat::Argb8 => argb_to_jpeg(size, data, quality),
    }
}
