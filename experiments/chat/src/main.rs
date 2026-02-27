use {
    audio::*,
    base::*,
    inference::*,
    std::{
        io::Write,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
        time::Duration,
    },
    tokio::sync::RwLock,
};

const VOICE_PATH: &str = "data/pocket/voices/desmond.bin";
const ASR_SAMPLE_RATE: usize = 16000;
const TTS_SAMPLE_RATE: usize = 24000;

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    // initialize inference
    let inference = Inference::new().map_err(|e| InferError::Runtime(e.to_string()))?;

    // initialize AudioIn
    log_info!("Opening audio input...");
    let mut audioin_listener = create_audioin(Some(AudioInConfig {
        sample_rate: ASR_SAMPLE_RATE,
        chunk_size: 8000,
        boost: 4,
        ..Default::default()
    }))
    .await;

    // shared epoch for pipeline cancellation
    let epoch = Epoch::new();

    // initialize AudioOut
    log_info!("Opening audio output...");
    let (audioout_handle, mut audioout_listener) = audio::create_audioout::<TtsPayload<u64>>(
        Some(AudioOutConfig {
            sample_rate: TTS_SAMPLE_RATE,
            ..Default::default()
        }),
        epoch.clone(),
    );
    let audioout_handle = Arc::new(audioout_handle);

    // load ASR
    log_info!("Loading ASR...");
    let (asr_handle, mut asr_listener) = inference.use_parakeet::<()>(&onnx::Executor::Cuda(0))?;
    let asr_handle = Arc::new(asr_handle);

    // load LLM
    log_info!("Loading LLM...");
    let (llm_handle, mut llm_listener) =
        inference.use_phi3(&onnx::Executor::Cuda(0), epoch.clone())?;
    let llm_handle = Arc::new(llm_handle);

    // load TTS
    log_info!("Loading TTS...");
    let (tts_handle, mut tts_listener) =
        inference.use_pocket::<u64>(&onnx::Executor::Cpu, &VOICE_PATH, epoch.clone())?;
    let tts_handle = Arc::new(tts_handle);

    // state
    let input_accumulator: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));
    let gate = Arc::new(AtomicBool::new(false));
    let output_sentences: Arc<RwLock<Vec<String>>> = Arc::new(RwLock::new(Vec::new()));
    let current_id = Arc::new(AtomicU64::new(1));

    // task: AudioIn pump
    log_info!("Spawning AudioIn->ASR pump...");
    tokio::spawn({
        let asr_handle = Arc::clone(&asr_handle);
        async move {
            while let Some(audio) = audioin_listener.recv().await {
                if let Err(error) = asr_handle.send(AsrInput { payload: (), audio }) {
                    log_error!("ASR send failed: {}", error);
                }
            }
        }
    });

    // task: TTS pump
    log_info!("Spawning TTS->AudioOut pump...");
    tokio::spawn({
        let audioout_handle = Arc::clone(&audioout_handle);
        async move {
            while let Some(stamped) = tts_listener.recv().await {
                let output = stamped.inner;
                if let Err(error) = audioout_handle.send(AudioOutChunk {
                    payload: output.payload,
                    data: output.data,
                }) {
                    log_error!("AudioOut send failed: {}", error);
                }
            }
        }
    });

    // task: AudioOut pump
    log_info!("Spawning AudioOut pump...");
    tokio::spawn({
        let input_accumulator = Arc::clone(&input_accumulator);
        async move {
            let mut current_index = 0usize;
            let mut current_sentence_id = 0u64;
            while let Some(status) = audioout_listener.recv().await {
                match status {
                    AudioOutStatus::Started(payload) => {
                        if current_sentence_id != payload.payload {
                            // new sentence started
                            current_sentence_id = payload.payload;
                            current_index = 0;

                            // clear input accumulator (maybe not for all sentences, but only the first one in a set)
                            input_accumulator.write().await.clear();
                        }
                    }
                    AudioOutStatus::Finished { payload, index } => {
                        //println!("AudioOut: finished {} ({} samples)", id, index);
                        if current_sentence_id == payload.payload {
                            // a chunk of our current sentence finished
                            current_index += index;
                        }
                    }
                    AudioOutStatus::Canceled { payload, index } => {
                        // a chunk of our current sentence was canceled
                        if current_sentence_id == payload.payload {
                            current_index += index;
                            println!(
                                "AudioOut: canceled ({}:{}) at {}: {:.2}%",
                                current_sentence_id,
                                payload.id,
                                current_index,
                                current_index as f64 / TTS_SAMPLE_RATE as f64 * 100.0
                            );
                        }
                    }
                }
            }
        }
    });

    // task: ASR pump
    log_info!("Spawning ASR pump...");
    tokio::spawn({
        let llm_handle = Arc::clone(&llm_handle);
        let input_accumulator = Arc::clone(&input_accumulator);
        let gate = Arc::clone(&gate);
        let output_sentences = Arc::clone(&output_sentences);
        let current_id = Arc::clone(&current_id);
        let tts_handle_asr = Arc::clone(&tts_handle);
        let epoch = epoch.clone();
        async move {
            while let Some(chunk) = asr_listener.recv().await {
                // ignore ASR output that contains no alphanumeric characters
                let has_text = chunk.text.chars().any(|c| c.is_alphanumeric());
                println!("=====>> ASR: \"{}\" ({}) <<=====", chunk.text, if has_text { "text" } else { "empty" });

                if !has_text {
                    println!("ASR:    empty, opening gate, flushing output sentences to TTS:");

                    // open gate
                    gate.store(true, Ordering::Relaxed);

                    // flush output sentences to TTS
                    {
                        let mut write = output_sentences.write().await;
                        for sentence in write.drain(..) {
                            let id = current_id.load(Ordering::Relaxed);
                            println!("ASR:        {}: \"{}\"", id, sentence);
                            if let Err(error) = tts_handle_asr.send(TtsInput {
                                payload: id,
                                text: sentence,
                            }) {
                                log_error!("TTS send failed: {}", error);
                            }
                            current_id.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                } else {
                    // add to input accumulator
                    {
                        let mut write = input_accumulator.write().await;
                        write.push_str(&chunk.text);
                        println!("ASR:    \"{}\"", write);
                    }

                    // close gate, advance epoch to cancel entire pipeline
                    println!("ASR:    closing gate, advancing epoch");
                    gate.store(false, Ordering::Relaxed);
                    epoch.advance();

                    // clear output sentences
                    {
                        let mut write = output_sentences.write().await;
                        write.clear();
                    }

                    // restart LLM with new prompt
                    println!("ASR:    restarting LLM with new prompt");
                    let prompt = format!(
                        "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n",
                        input_accumulator.read().await
                    );
                    if let Err(error) = llm_handle.send(&prompt) {
                        log_error!("LLM send failed: {}", error);
                    }
                }
            }
        }
    });

    // task: LLM pump
    log_info!("Spawning LLM pump...");
    tokio::spawn({
        let tts_handle = Arc::clone(&tts_handle);
        let gate = Arc::clone(&gate);
        let output_sentences = Arc::clone(&output_sentences);
        let current_id = Arc::clone(&current_id);
        let epoch = epoch.clone();
        async move {
            let mut sentence = String::new();
            let mut current_epoch = 0u64;
            while let Some(stamped) = llm_listener.recv().await {
                // epoch changed — clear stale sentence buffer
                if stamped.epoch != current_epoch {
                    sentence.clear();
                    output_sentences.write().await.clear();
                    current_epoch = stamped.epoch;
                }

                // drop stale tokens
                if !epoch.is_current(stamped.epoch) {
                    continue;
                }

                match stamped.inner {
                    LlmOutput::Token(token) => {
                        sentence.push_str(&token);
                        if ends_with_sentence_boundary(&sentence) {
                            println!("LLM: \"{}\"", sentence.trim());
                            // push to output sentences
                            {
                                let mut write = output_sentences.write().await;
                                write.push(sentence.trim().to_string());

                                // if gate open, flush output sentences to TTS
                                if gate.load(Ordering::Relaxed) {
                                    println!(
                                        "LLM:    gate open, flushing output sentences to TTS:"
                                    );
                                    for sentence in write.drain(..) {
                                        let id = current_id.load(Ordering::Relaxed);
                                        println!("LLM:        {}: \"{}\"", id, sentence);
                                        if let Err(error) = tts_handle.send(TtsInput {
                                            payload: id,
                                            text: sentence,
                                        }) {
                                            log_error!("TTS send failed: {}", error);
                                        }
                                        current_id.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                            sentence.clear();
                        }
                    }
                    LlmOutput::Eos => {
                        println!("LLM: EOS");

                        // push whatever is left to output sentences
                        {
                            let trimmed = sentence.trim().to_string();
                            if !trimmed.is_empty() {
                                let mut write = output_sentences.write().await;
                                write.push(trimmed);

                                // if gate open, flush output sentences to TTS
                                if gate.load(Ordering::Relaxed) {
                                    println!("LLM: gate open: flushing output sentences to TTS:");
                                    for sentence in write.drain(..) {
                                        let id = current_id.load(Ordering::Relaxed);
                                        println!("    {}: \"{}\"", id, sentence);
                                        if let Err(error) = tts_handle.send(TtsInput {
                                            payload: id,
                                            text: sentence,
                                        }) {
                                            log_error!("TTS send failed: {}", error);
                                        }
                                        current_id.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                        sentence.clear();
                    }
                }
            }
        }
    });

    log_info!("Chat loop: (empty line to exit)");
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

    /*
    // chat loop
    loop {
        // print prompt
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|e| InferError::Runtime(e.to_string()))?;

        // read user input
        let mut user_input = String::new();
        io::stdin()
            .read_line(&mut user_input)
            .map_err(|e| InferError::Runtime(e.to_string()))?;
        let user_input = user_input.trim();

        // exit on empty input
        if user_input.is_empty() {
            break;
        }

        // truncate history if needed (before building prompt to limit context size)
        if history.len() >= MAX_HISTORY_TURNS {
            log_info!("truncating history");
            history.drain(0..(history.len() - MAX_HISTORY_TURNS + 1));
        }

        // build prompt
        let mut prompt = String::new();
        prompt.push_str("<|system|>\nYou are in a really bad mood.<|end|>\n");
        for (source, message) in &history {
            prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", source, message));
        }
        history.push(("user".to_string(), user_input.to_string()));
        prompt.push_str(&format!("<|user|>\n{}<|end|>\n", user_input));
        prompt.push_str("<|assistant|>\n");

        // generate LLM response
        if let Err(e) = llm.send(&prompt).await {
            log_error!("LLM forward failed: {}", e);
            continue;
        }

        // receive LLM response
        let mut response = String::new();
        let mut sentence_buf = String::new();
        loop {
            match llm.recv().await {
                Some(LlmToken::Text(token)) => {
                    print!("{}", token);
                    io::stdout().flush().ok();
                    response.push_str(&token);
                    sentence_buf.push_str(&token);

                    // check if the buffer ends with sentence-ending punctuation
                    if ends_with_sentence_boundary(&sentence_buf) {
                        let sentence = sentence_buf.trim().to_string();
                        sentence_buf.clear();
                        if !sentence.is_empty() {
                            if let Err(e) = tts_input_tx.send(TtsInput { text: sentence }).await {
                                log_error!("TTS send failed: {}", e);
                            }
                        }
                    }
                }
                Some(LlmToken::Eos) => {
                    break;
                }
                None => {
                    log_error!("LLM stream ended");
                    break;
                }
            }
        }
        println!();

        if response.is_empty() {
            continue;
        }

        // flush any remaining text
        let remaining = sentence_buf.trim().to_string();
        if !remaining.is_empty() {
            if let Err(e) = tts_input_tx.send(TtsInput { text: remaining }).await {
                log_error!("TTS send failed: {}", e);
            }
        }

        // add to history
        history.push(("assistant".to_string(), response.clone()));
    }
    */

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
