use {
    audio::*,
    base::*,
    inference::*,
    std::{
        io::Write,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    },
    tokio::sync::RwLock,
};

const VOICE_PATH: &str = "data/pocket/voices/hannah.bin";
const ASR_SAMPLE_RATE: usize = 16000;
const TTS_SAMPLE_RATE: usize = 24000;

#[tokio::main]
async fn main() -> Result<(), InferError> {
    base::init_stdout_logger();

    // initialize inference
    let inference = Inference::new().map_err(|e| InferError::Runtime(e.to_string()))?;

    // initialize AudioIn
    log_info!("Opening audio input...");
    let mut audioin = AudioInListener::open(Some(AudioInConfig {
        sample_rate: ASR_SAMPLE_RATE,
        ..Default::default()
    }))
    .await;

    // initialize AudioOut
    log_info!("Opening audio output...");
    let audioout = Arc::new(
        AudioOutHandle::open(Some(AudioOutConfig {
            sample_rate: TTS_SAMPLE_RATE,
            ..Default::default()
        }))
        .await,
    );

    // load ASR
    log_info!("Loading ASR...");
    let (asr_handle, mut asr_listener) = inference.use_parakeet(&onnx::Executor::Cuda(0))?;
    let asr_handle = Arc::new(asr_handle);

    // load LLM
    log_info!("Loading LLM...");
    let (llm_handle, mut llm_listener) = inference.use_phi3(&onnx::Executor::Cuda(0))?;
    let llm_handle = Arc::new(llm_handle);

    // load TTS
    log_info!("Loading TTS...");
    let (tts_handle, mut tts_listener) = inference.use_pocket(&onnx::Executor::Cpu, &VOICE_PATH)?;
    let tts_handle = Arc::new(tts_handle);

    // this is a simple pump that blocks on incoming audio samples and sends them to the ASR
    log_info!("Spawning AudioIn->ASR pump...");
    tokio::spawn({
        let asr_handle = Arc::clone(&asr_handle);
        async move {
            while let Some(audio) = audioin.recv().await {
                if let Err(error) = asr_handle.send(AsrInput { audio }) {
                    log_error!("ASR send failed: {}", error);
                }
            }
        }
    });

    // this is a simple pump that blocks on incoming TTS audio samples and sends them to AudioOut
    log_info!("Spawning TTS->AudioOut pump...");
    tokio::spawn({
        async move {
            while let Some(chunk) = tts_listener.recv().await {
                if let Err(error) = audioout.send(chunk.audio) {
                    log_error!("AudioOut send failed: {}", error);
                }
            }
        }
    });

    // machine state
    let current_sentence: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));
    let first_round: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));
    let asr_empty: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));

    // this pump blocks on incoming ASR text fragments, collects them and restarts LLM+TTS pumps with the accumulated sentence
    log_info!("Spawning ASR pump...");
    tokio::spawn({
        let current_sentence = Arc::clone(&current_sentence);
        let first_round = Arc::clone(&first_round);
        let asr_empty = Arc::clone(&asr_empty);
        let llm_handle = Arc::clone(&llm_handle);
        let tts_handle = Arc::clone(&tts_handle);
        async move {
            while let Some(chunk) = asr_listener.recv().await {
                println!("from ASR: {}", chunk.text);

                // keep track of silence from ASR
                asr_empty.store(chunk.text.is_empty(), Ordering::Relaxed);

                // if nothing was heard, continue
                if chunk.text.is_empty() {
                    continue;
                }

                // add chunk to current sentence
                {
                    let mut write = current_sentence.write().await;
                    write.push_str(&chunk.text);
                }

                println!(
                    "-> current input sentence: {}",
                    current_sentence.read().await
                );

                // cancel LLM and TTS
                println!("-> canceling LLM and TTS");
                llm_handle.cancel();
                tts_handle.cancel();

                // reset first round flag (so next full sentence from LLM clears current_sentence)
                println!("-> resetting first round flag");
                first_round.store(true, Ordering::Relaxed);

                // restart LLM with new sentence (formatted as Phi3 chat prompt)
                let prompt = format!(
                    "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n",
                    current_sentence.read().await
                );
                println!("-> sending to LLM");
                if let Err(error) = llm_handle.send(&prompt) {
                    log_error!("LLM send failed: {}", error);
                }
            }
        }
    });

    // this pump blocks on incoming LLM text tokens, collects them and sends them as finished sentences to the TTS
    log_info!("Spawning LLM pump...");
    tokio::spawn({
        let current_sentence = Arc::clone(&current_sentence);
        let first_round = Arc::clone(&first_round);
        let asr_empty = Arc::clone(&asr_empty);
        let tts_handle = Arc::clone(&tts_handle);
        async move {
            let mut sentence_buffer = String::new();
            let mut sentence_holding = Vec::<String>::new();
            loop {
                match llm_listener.recv().await {
                    Some(LlmToken::Text(token)) => {
                        // accumulate tokens into buffer
                        sentence_buffer.push_str(&token);
                        if ends_with_sentence_boundary(&sentence_buffer) {
                            println!("from LLM: to holding: {}", sentence_buffer);

                            // place sentence in holding
                            sentence_holding.push(sentence_buffer.clone());
                            sentence_buffer.clear();
                        }

                        // if there is a sentence in holding and nothing was heard from ASR, send sentence to TTS and clear current_sentence
                        if !sentence_holding.is_empty() && asr_empty.load(Ordering::Relaxed) {
                            println!("from LLM holding: {:?}", sentence_holding);
                            if first_round.swap(false, Ordering::Relaxed) {
                                println!("-> clearing current input sentence");
                                current_sentence.write().await.clear();
                            }
                            for sentence in &sentence_holding {
                                println!("-> sending to TTS: {}", sentence);
                                if let Err(error) = tts_handle.send(TtsInput {
                                    text: sentence.clone(),
                                }) {
                                    log_error!("TTS send failed: {}", error);
                                }
                            }
                            sentence_holding.clear();
                        }
                    }
                    Some(LlmToken::Eos) => {
                        // Flush any remaining text in the sentence buffer
                        let sentence = sentence_buffer.trim().to_string();
                        sentence_buffer.clear();
                        if !sentence.is_empty() {
                            println!("from LLM (EOS flush): {}", sentence);
                            if first_round.swap(false, Ordering::Relaxed) {
                                println!("-> clearing current sentence");
                                current_sentence.write().await.clear();
                            }
                            println!("-> sending to TTS");
                            if let Err(error) = tts_handle.send(TtsInput { text: sentence }) {
                                log_error!("TTS send failed: {}", error);
                            }
                        }
                        // Continue waiting for next generation round
                    }
                    None => {
                        log_error!("LLM stream ended");
                        break;
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
