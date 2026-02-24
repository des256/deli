use {
    audio::{AudioData, AudioOut, AudioOutConfig},
    base::*,
    futures_util::{SinkExt, StreamExt},
    inference::Inference,
    std::{
        io::{self, Write},
        path::PathBuf,
        time::Duration,
    },
};

const SMOLLM3_MODEL_PATH: &str = "data/smollm3/model_int8.onnx";
const SMOLLM3_TOKENIZER_PATH: &str = "data/smollm3/tokenizer.json";
const POCKET_TEXT_CONDITIONER: &str = "data/pocket/text_conditioner.onnx";
const POCKET_FLOW_MAIN: &str = "data/pocket/flow_lm_main_int8.onnx";
const POCKET_FLOW_STEP: &str = "data/pocket/flow_lm_flow_int8.onnx";
const POCKET_MIMI_DECODER: &str = "data/pocket/mimi_decoder_int8.onnx";
const POCKET_TOKENIZER: &str = "data/pocket/tokenizer.json";
const POCKET_VOICE: &str = "data/pocket/voices/stephen.bin";
const SAMPLE_RATE: usize = 24000;
const MAX_TOKENS: usize = 2048;
const MAX_HISTORY_TURNS: usize = 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Validate model files exist
    log_info!("Validating model files...");
    let smollm3_model = PathBuf::from(SMOLLM3_MODEL_PATH);
    let smollm3_tokenizer = PathBuf::from(SMOLLM3_TOKENIZER_PATH);
    let pocket_paths = [
        POCKET_TEXT_CONDITIONER,
        POCKET_FLOW_MAIN,
        POCKET_FLOW_STEP,
        POCKET_MIMI_DECODER,
        POCKET_TOKENIZER,
        POCKET_VOICE,
    ];

    if !smollm3_model.exists() || !smollm3_tokenizer.exists() {
        eprintln!("SmolLM3 model files missing. Expected:");
        eprintln!("  - {}", SMOLLM3_MODEL_PATH);
        eprintln!("  - {}", SMOLLM3_TOKENIZER_PATH);
        std::process::exit(1);
    }

    for path in &pocket_paths {
        if !PathBuf::from(path).exists() {
            eprintln!("Pocket TTS model file missing: {}", path);
            std::process::exit(1);
        }
    }

    log_info!("Model files validated");

    // Initialize inference
    let cpu_inference = Inference::cpu()?;
    log_info!("CPU inference initialized");

    // also use CUDA if available
    #[cfg(feature = "cuda")]
    let cuda_inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let cuda_inference = Inference::cpu()?;
    log_info!("CUDA inference initialized (if available)");

    // Load LLM
    log_info!("Loading SmolLM3...");
    let mut llm = cuda_inference
        .use_smollm3(&smollm3_model, &smollm3_tokenizer)?
        .with_max_tokens(MAX_TOKENS);
    log_info!("SmolLM3 loaded");

    // Load Pocket TTS
    log_info!("Loading Pocket TTS...");
    let mut tts = cpu_inference.use_pocket_tts(
        POCKET_TEXT_CONDITIONER,
        POCKET_FLOW_MAIN,
        POCKET_FLOW_STEP,
        POCKET_MIMI_DECODER,
        POCKET_TOKENIZER,
        POCKET_VOICE,
    )?;
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
        let prompt = build_prompt(&history, user_input);

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
                            log_info!("Sending sentence to TTS: {:?}", sentence);
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
            log_info!("Sending remaining text to TTS: {:?}", remaining);
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

/// Build ChatML prompt with system message, history, and current user input
fn build_prompt(history: &[(String, String)], user_input: &str) -> String {
    let mut prompt = String::new();

    // System message
    prompt.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");

    // Add conversation history
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_msg));
        prompt.push_str(&format!(
            "<|im_start|>assistant\n{}<|im_end|>\n",
            assistant_msg
        ));
    }

    // Add current user message and prompt for assistant
    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_input));
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}
