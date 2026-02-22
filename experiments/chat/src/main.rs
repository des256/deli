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

const QWEN3_MODEL_PATH: &str = "data/qwen3/qwen3-4b-q4_k_m.gguf";
const QWEN3_TOKENIZER_PATH: &str = "data/qwen3/tokenizer.json";
const KOKORO_MODEL_PATH: &str = "data/kokoro/kokoro-v1.0.onnx";
const KOKORO_VOICE_PATH: &str = "data/kokoro/af_nicole.npy";
const KOKORO_ESPEAK_DATA_PATH: &str = "/usr/lib/x86_64-linux-gnu/espeak-ng-data";
const SAMPLE_RATE: usize = 24000;
const SAMPLE_LEN: usize = 2048;
const MAX_HISTORY_TURNS: usize = 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    // Validate model files exist
    log_info!("Validating model files...");
    let qwen3_model = PathBuf::from(QWEN3_MODEL_PATH);
    let qwen3_tokenizer = PathBuf::from(QWEN3_TOKENIZER_PATH);
    let kokoro_model = PathBuf::from(KOKORO_MODEL_PATH);
    let kokoro_voice = PathBuf::from(KOKORO_VOICE_PATH);

    if !qwen3_model.exists() || !qwen3_tokenizer.exists() {
        eprintln!("Qwen3 model files missing. Expected:");
        eprintln!("  - {}", QWEN3_MODEL_PATH);
        eprintln!("  - {}", QWEN3_TOKENIZER_PATH);
        std::process::exit(1);
    }

    if !kokoro_model.exists() || !kokoro_voice.exists() {
        eprintln!("Kokoro model files missing. Expected:");
        eprintln!("  - {}", KOKORO_MODEL_PATH);
        eprintln!("  - {}", KOKORO_VOICE_PATH);
        std::process::exit(1);
    }

    log_info!("Model files validated");

    // Initialize CUDA inference
    log_info!("Initializing CUDA inference...");
    let inference = Inference::cuda(0)?;
    log_info!("CUDA inference initialized");

    // Load Qwen3 LLM
    log_info!("Loading Qwen3 LLM...");
    let llm = inference.use_qwen3(&qwen3_model, &qwen3_tokenizer)?;
    log_info!("Qwen3 model loaded");

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
            log_info!("Truncating conversation history (keeping last {} turns)", MAX_HISTORY_TURNS);
            history.drain(0..(history.len() - MAX_HISTORY_TURNS + 1));
        }

        // Build prompt with chat template
        let prompt = build_prompt(&history, user_input);

        // Generate LLM response
        log_info!("Generating response...");
        let response = match llm.forward(&prompt, SAMPLE_LEN).await {
            Ok(text) => text,
            Err(e) => {
                log_error!("LLM inference failed: {}", e);
                eprintln!("Error generating response: {}", e);
                continue;
            }
        };

        // Print response
        println!("\n{}\n", response);

        // Add to history
        history.push((user_input.to_string(), response.clone()));

        // Synthesize and play speech
        log_info!("Loading Kokoro TTS...");
        let mut kokoro = match inference.use_kokoro(&kokoro_model, &kokoro_voice, Some(KOKORO_ESPEAK_DATA_PATH)) {
            Ok(k) => k,
            Err(e) => {
                log_error!("TTS initialization failed: {}", e);
                eprintln!("Error initializing TTS: {}", e);
                continue;
            }
        };
        log_info!("Synthesizing speech...");

        if let Err(e) = kokoro.send(response).await {
            log_error!("TTS send failed: {}", e);
            eprintln!("Error sending to TTS: {}", e);
            continue;
        }
        if let Err(e) = kokoro.close().await {
            log_error!("TTS close failed: {}", e);
            eprintln!("Error closing TTS: {}", e);
            continue;
        }

        let sample = match kokoro.next().await {
            Some(Ok(sample)) => sample,
            Some(Err(e)) => {
                log_error!("TTS synthesis failed: {}", e);
                eprintln!("Error synthesizing speech: {}", e);
                continue;
            }
            None => {
                log_error!("TTS stream ended without audio");
                eprintln!("Error: no audio generated");
                continue;
            }
        };

        // Calculate playback duration before moving sample
        let num_samples = match &sample.data {
            AudioData::Pcm(tensor) => tensor.data.len(),
        };
        log_info!("Generated {} audio samples", num_samples);

        // Play audio
        audioout.play(sample).await;

        // Wait for playback to complete
        let duration_secs = num_samples as f64 / SAMPLE_RATE as f64;
        let duration_ms = (duration_secs * 1000.0) as u64 + 500;
        tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    }

    log_info!("Chat session ended");
    Ok(())
}

/// Build Qwen3 ChatML prompt with system message, history, and current user input
fn build_prompt(history: &[(String, String)], user_input: &str) -> String {
    let mut prompt = String::new();

    // System message
    prompt.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");

    // Add conversation history
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_msg));
        prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", assistant_msg));
    }

    // Add current user message and prompt for assistant
    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user_input));
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}
