use {
    audio::{AudioData, AudioOut, AudioOutConfig},
    base::*,
    futures_util::{SinkExt, StreamExt},
    inference::Inference,
    std::{
        io::{self, Write},
        time::Duration,
    },
};

const POCKET_VOICE_PATH: &str = "data/pocket/voices/hannah.bin";
const SAMPLE_RATE: usize = 24000;
const MAX_TOKENS: usize = 2048;
const MAX_HISTORY_TURNS: usize = 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Initialize inference
    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(feature = "cuda")]
    log_info!("CUDA inference initialized");
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;
    #[cfg(not(feature = "cuda"))]
    log_info!("CPU inference initialized");

    // Load LLM
    log_info!("Loading LLM...");
    //let mut llm = inference.use_gemma3()?.with_max_tokens(MAX_TOKENS);
    //let chat_format = ChatFormat::Gemma;
    //let mut llm = inference.use_llama32()?.with_max_tokens(MAX_TOKENS);
    //let chat_format = ChatFormat::Llama3;
    let mut llm = inference.use_phi3()?.with_max_tokens(MAX_TOKENS);
    let chat_format = ChatFormat::Phi3;
    //let mut llm = inference.use_smollm3()?.with_max_tokens(MAX_TOKENS);
    //let chat_format = ChatFormat::ChatML;
    log_info!("LLM loaded");

    // Load Pocket TTS
    log_info!("Loading TTS...");
    let mut tts = inference.use_pocket_tts(&POCKET_VOICE_PATH)?;
    log_info!("Pocket TTS loaded");

    // Initialize AudioOut
    log_info!("Opening audio output...");
    let mut audioout = AudioOut::open(None).await;
    audioout
        .select(AudioOutConfig {
            sample_rate: SAMPLE_RATE,
            ..Default::default()
        })
        .await;
    log_info!("Audio output ready");

    // Conversation history: (user_message, assistant_response)
    let mut history: Vec<(String, String)> = Vec::new();

    println!("\nChat initialized. Type your message and press Enter. Empty line to exit.\n");

    // Chat loop
    loop {
        // Print prompt
        print!("> ");
        io::stdout().flush()?;

        // Read user input
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        // Exit on empty input
        if user_input.is_empty() {
            log_info!("Empty input received, exiting");
            break;
        }

        // Truncate history if needed (before building prompt to limit context size)
        if history.len() >= MAX_HISTORY_TURNS {
            log_info!(
                "Truncating conversation history (keeping last {} turns)",
                MAX_HISTORY_TURNS
            );
            history.drain(0..(history.len() - MAX_HISTORY_TURNS + 1));
        }

        // Build prompt with chat template
        let prompt = build_prompt(chat_format, &history, user_input);

        // Generate LLM response (streaming) and pipeline sentences to TTS
        log_info!("Generating response...");
        if let Err(e) = llm.forward(&prompt) {
            log_error!("LLM forward failed: {}", e);
            eprintln!("Error starting generation: {}", e);
            continue;
        }

        // Stream tokens from LLM, split into sentences, and feed each to TTS
        // as soon as it's complete. TTS processes sentences in parallel with
        // ongoing LLM generation.
        let mut response = String::new();
        let mut sentence_buf = String::new();
        let mut tts_error = false;
        let mut total_samples = 0usize;
        let mut sentences_sent = 0usize;
        let mut sentences_done = 0usize;

        loop {
            match llm.recv().await {
                Some(Ok(token)) => {
                    print!("{}", token);
                    io::stdout().flush().ok();
                    response.push_str(&token);
                    sentence_buf.push_str(&token);

                    // Check if the buffer ends with sentence-ending punctuation
                    if ends_with_sentence_boundary(&sentence_buf) {
                        let sentence = sentence_buf.trim().to_string();
                        sentence_buf.clear();
                        if !sentence.is_empty() {
                            eprint!("\n[sent.]\n");
                            if let Err(e) = tts.send(sentence).await {
                                log_error!("TTS send failed: {}", e);
                                tts_error = true;
                            } else {
                                sentences_sent += 1;
                            }
                        }
                    }

                    // Drain any ready TTS audio without blocking
                    sentences_done +=
                        drain_ready_audio(&mut tts, &mut audioout, &mut total_samples).await;
                }
                Some(Err(e)) => {
                    log_error!("LLM generation error: {}", e);
                    eprintln!("\nError during generation: {}", e);
                    break;
                }
                None => break,
            }
        }
        println!(); // Newline after streamed output

        if response.is_empty() {
            continue;
        }

        // Flush any remaining text that didn't end with punctuation
        let remaining = sentence_buf.trim().to_string();
        if !remaining.is_empty() && !tts_error {
            eprint!("\nsent.");
            if let Err(e) = tts.send(remaining).await {
                log_error!("TTS send failed: {}", e);
                tts_error = true;
            } else {
                sentences_sent += 1;
            }
        }

        // Drain all remaining TTS audio until every sentence has been synthesized.
        // Each sentence produces an end-of-utterance marker (empty PCM) when done.
        while sentences_done < sentences_sent && !tts_error {
            match tts.next().await {
                Some(Ok(sample)) => {
                    let n = match &sample.data {
                        AudioData::Pcm(tensor) => tensor.data.len(),
                    };
                    if n == 0 {
                        sentences_done += 1;
                        continue;
                    }
                    total_samples += n;
                    audioout.play(sample).await;
                }
                Some(Err(e)) => {
                    log_error!("TTS synthesis failed: {}", e);
                    eprintln!("Error synthesizing speech: {}", e);
                    tts_error = true;
                }
                None => break,
            }
        }

        // Add to history
        history.push((user_input.to_string(), response.clone()));

        log_info!("Streamed {} total audio samples", total_samples);

        // Brief wait for remaining audio to drain from the ring buffer.
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    log_info!("Chat session ended");
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

/// Drain any TTS audio chunks that are immediately ready, without blocking.
/// Returns the number of end-of-utterance markers received (completed sentences).
async fn drain_ready_audio(
    tts: &mut inference::tts::pocket::PocketTts,
    audioout: &mut AudioOut,
    total_samples: &mut usize,
) -> usize {
    use futures_util::FutureExt;
    let mut completed = 0;
    loop {
        // Poll tts.next() without waiting — return immediately if nothing ready
        let maybe = tts.next().now_or_never();
        match maybe {
            Some(Some(Ok(sample))) => {
                let n = match &sample.data {
                    AudioData::Pcm(tensor) => tensor.data.len(),
                };
                if n == 0 {
                    completed += 1;
                    continue;
                }
                *total_samples += n;
                audioout.play(sample).await;
            }
            Some(Some(Err(e))) => {
                base::log_error!("TTS drain error: {}", e);
                break;
            }
            _ => break, // Nothing ready or stream ended
        }
    }
    completed
}

#[derive(Clone, Copy)]
enum ChatFormat {
    /// ChatML: <|im_start|>role\n...<|im_end|> (SmolLM3)
    ChatML,
    /// Llama 3: <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
    Llama3,
    /// Gemma: <start_of_turn>role\n...<end_of_turn>
    Gemma,
    /// Phi3: <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n
    Phi3,
}

fn build_prompt(fmt: ChatFormat, history: &[(String, String)], user_input: &str) -> String {
    match fmt {
        ChatFormat::ChatML => build_chatml_prompt(history, user_input),
        ChatFormat::Llama3 => build_llama3_prompt(history, user_input),
        ChatFormat::Gemma => build_gemma_prompt(history, user_input),
        ChatFormat::Phi3 => build_phi3_prompt(history, user_input),
    }
}

/// ChatML format (SmolLM3, Phi3)
fn build_chatml_prompt(history: &[(String, String)], user_input: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_msg));
        prompt.push_str(&format!(
            "<|im_start|>assistant\n{}<|im_end|>\n",
            assistant_msg
        ));
    }
    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_input));
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Llama 3 format
fn build_llama3_prompt(history: &[(String, String)], user_input: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|begin_of_text|>");
    prompt.push_str(
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>",
    );
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            user_msg
        ));
        prompt.push_str(&format!(
            "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
            assistant_msg
        ));
    }
    prompt.push_str(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
        user_input
    ));
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Phi3 format: <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n
fn build_phi3_prompt(history: &[(String, String)], user_input: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|system|>\nYou are a helpful assistant.<|end|>\n");
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!("<|user|>\n{}<|end|>\n", user_msg));
        prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", assistant_msg));
    }
    prompt.push_str(&format!("<|user|>\n{}<|end|>\n", user_input));
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Gemma format (system message prepended to first user turn)
fn build_gemma_prompt(history: &[(String, String)], user_input: &str) -> String {
    let system_msg = "You are a helpful assistant.";
    let mut prompt = String::new();
    prompt.push_str("<bos>");

    if history.is_empty() {
        prompt.push_str(&format!(
            "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
            system_msg, user_input
        ));
    } else {
        let (first_user, first_assistant) = &history[0];
        prompt.push_str(&format!(
            "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n{}<end_of_turn>\n",
            system_msg, first_user, first_assistant
        ));
        for (user_msg, assistant_msg) in &history[1..] {
            prompt.push_str(&format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n{}<end_of_turn>\n",
                user_msg, assistant_msg
            ));
        }
        prompt.push_str(&format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            user_input
        ));
    }

    prompt
}
