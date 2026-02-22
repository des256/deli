# Chat Experiment Implementation Plan

Created: 2026-02-22
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No

> **Status Lifecycle:** PENDING → COMPLETE → VERIFIED
> **Iterations:** Tracks implement→verify cycles (incremented by verify phase)
>
> - PENDING: Initial state, awaiting implementation
> - COMPLETE: All tasks implemented
> - VERIFIED: All checks passed
>
> **Approval Gate:** Implementation CANNOT proceed until `Approved: Yes`
> **Worktree:** Set at plan creation (from dispatcher). `Yes` uses git worktree isolation; `No` works directly on current branch (default)

## Summary

**Goal:** Build a new experiment binary called `chat` that initializes the Qwen3 LLM and Kokoro TTS on CUDA, presents an interactive stdin-based chat loop, sends user input to the LLM with Qwen3 chat template formatting, displays the LLM's text response, and speaks it through TTS via PulseAudio. Uses async (tokio) throughout.

**Architecture:** A single `main.rs` binary that initializes `Inference::cuda(0)`, loads Qwen3 and Kokoro, then enters an async loop reading lines from stdin. Each user message is formatted using Qwen3's `<|im_start|>/<|im_end|>` chat template, sent to `Qwen3::forward()`, the response is printed to stdout, then sent through Kokoro TTS → AudioOut for playback. Conversation history is maintained in-memory for multi-turn context.

**Tech Stack:** Rust, tokio (async runtime), inference crate (Qwen3 + Kokoro), audio crate (AudioOut/PulseAudio), base crate (logging)

## Scope

### In Scope

- New experiment directory `experiments/chat/` with `Cargo.toml` and `src/main.rs`
- Initialize Inference with CUDA (hardcoded, no feature flags)
- Load Qwen3 LLM (4B Q4 model) and Kokoro TTS
- Interactive stdin chat loop with `> ` prompt
- Qwen3 chat template formatting (`<|im_start|>system/user/assistant<|im_end|>`)
- Conversation history tracking for multi-turn chat
- Display LLM response text to stdout
- Speak LLM response through TTS → AudioOut
- Graceful exit on empty input or Ctrl+C

### Out of Scope

- Streaming token-by-token output (Qwen3::forward returns complete text)
- Voice input (ASR/microphone)
- Web UI or WebSocket interface
- Model selection via CLI arguments
- Conversation persistence/save/load

## Prerequisites

- CUDA-capable GPU with drivers installed
- Model files present:
  - `data/qwen3/qwen3-4b-q4_k_m.gguf`
  - `data/qwen3/tokenizer.json`
  - `data/kokoro/kokoro-v1.0.onnx`
  - `data/kokoro/af_nicole.npy` (voice file)
- PulseAudio running (for audio output)
- espeak-ng data at `/usr/lib/x86_64-linux-gnu/espeak-ng-data`

## Context for Implementer

- **Patterns to follow:** Follow the experiment structure from `experiments/tts-play/` — same Cargo.toml layout with `[[bin]]`, same tokio main, same `base::init_stdout_logger()` init. For CUDA without feature flags, follow `experiments/wav-asr/Cargo.toml:11` which uses `inference = { path = "../../crates/inference", features = ["cuda"] }`.
- **Conventions:** Use `base::log_info!()` / `log_error!()` macros for logging. Use `PathBuf::from()` for model paths. Validate model files exist before loading.
- **Key files:**
  - `crates/inference/src/inference.rs` — `Inference::cuda(0)`, `use_qwen3()`, `use_kokoro()` factory methods
  - `crates/inference/src/llm/qwen3.rs` — `Qwen3::forward(prompt, sample_len)` async method, returns full response string
  - `crates/inference/src/tts/kokoro.rs` — `Kokoro` implements `Sink<String>` + `Stream<Item=Result<AudioSample>>`
  - `crates/audio/src/audioout.rs` — `AudioOut::open()`, `.select()`, `.play()` for PulseAudio output
  - `experiments/tts-play/src/main.rs` — Reference for TTS + AudioOut playback pattern
  - `products/testy/robot/src/bin/testy.rs` — Reference for loading all three models together
- **Gotchas:**
  - `Qwen3::forward()` returns the complete generated text, not streaming tokens. The entire response must complete before TTS can start.
  - Kokoro TTS uses a `Sink`/`Stream` pattern: send text via `SinkExt::send()`, close with `SinkExt::close()`, then read audio via `StreamExt::next()`.
  - AudioOut doesn't flush on drop — need to wait for playback duration after `.play()`.
  - Qwen3 uses `<|im_start|>role\n content<|im_end|>` chat template format. The prompt must include the opening `<|im_start|>assistant\n` to trigger response generation.
  - Using the 4B model (`data/qwen3/qwen3-4b-q4_k_m.gguf`), same as the `testy` product.
- **Domain context:** This is a simple interactive chatbot experiment. The user types text, the LLM responds, and the response is spoken aloud. Multi-turn conversation is maintained by accumulating the chat history in the prompt.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create experiment scaffold with Cargo.toml and basic main
- [x] Task 2: Implement chat loop with LLM and TTS integration

**Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create experiment scaffold with Cargo.toml and basic main

**Objective:** Create the `experiments/chat/` directory with a properly configured `Cargo.toml` and a minimal `main.rs` that initializes logging, CUDA inference, and validates model files exist.

**Dependencies:** None

**Files:**

- Create: `experiments/chat/Cargo.toml`
- Create: `experiments/chat/src/main.rs`

**Key Decisions / Notes:**

- Use `inference = { path = "../../crates/inference", features = ["cuda"] }` to hardcode CUDA — no feature flags, matching the `wav-asr` pattern at `experiments/wav-asr/Cargo.toml:11`
- Binary name: `chat` (matching `[[bin]] name = "chat"`)
- Include `tokio` with `rt`, `rt-multi-thread`, `macros` features
- Include `futures-util` with `sink` feature for Kokoro's Sink/Stream interface
- Include `audio` and `base` crate dependencies
- The scaffold should compile and run (printing "model files validated" or erroring if files are missing)

**Definition of Done:**

- [ ] `experiments/chat/Cargo.toml` exists with correct dependencies and `features = ["cuda"]` on inference
- [ ] `experiments/chat/src/main.rs` compiles with `cargo build -p chat`
- [ ] Running the binary initializes CUDA, validates model file paths, and exits cleanly

**Verify:**

- `cargo build -p chat` — compiles without errors
- `cargo run -p chat` — initializes CUDA inference, logs model loading, exits cleanly

### Task 2: Implement chat loop with LLM and TTS integration

**Objective:** Implement the full interactive chat loop: read user input from stdin, format it with Qwen3 chat template, send to LLM, print response, synthesize and play speech via TTS + AudioOut.

**Dependencies:** Task 1

**Files:**

- Modify: `experiments/chat/src/main.rs`

**Key Decisions / Notes:**

- **Chat template format** (Qwen3 ChatML):
  ```
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  {user_message}<|im_end|>
  <|im_start|>assistant
  ```
  The model generates until `<|im_end|>` or `<|endoftext|>`.
- **Conversation history:** Maintain a `Vec<(String, String)>` of (user, assistant) message pairs. Rebuild the full prompt each turn by concatenating system message + all history + current user message + assistant opening.
- **Token limit:** Use `sample_len = 2048` for generation (reasonable for chat responses).
- **TTS flow** (follow `tts-play` pattern at `experiments/tts-play/src/main.rs:37-67`):
  1. Create a fresh `Kokoro` instance per response (or reuse and send text)
  2. Send response text via `kokoro.send(text).await`
  3. Close with `kokoro.close().await`
  4. Read audio with `kokoro.next().await`
  5. Play via `audioout.play(sample).await`
  6. Wait for playback duration
- **AudioOut config:** `sample_rate: 24000` (Kokoro's output rate), matching `tts-play` pattern
- **Stdin reading:** Use `tokio::io::BufReader` + `AsyncBufReadExt::read_line` for async stdin, or use `std::io::stdin().read_line()` in sync since stdin is blocking anyway. Use sync stdin for simplicity — tokio's async stdin is just a wrapper around blocking reads.
- **Exit condition:** Empty line (user just presses Enter) or EOF (Ctrl+D)
- Print `> ` prompt before each input line using `print!()` + `flush()`

**Definition of Done:**

- [ ] Chat loop reads user input from stdin with `> ` prompt
- [ ] User input is formatted with Qwen3 `<|im_start|>/<|im_end|>` chat template
- [ ] Conversation history is maintained across turns
- [ ] LLM response is printed to stdout
- [ ] LLM response is spoken through TTS → AudioOut
- [ ] Program exits cleanly on empty input or EOF
- [ ] Full chat interaction works end-to-end (type message → see response → hear response)

**Verify:**

- `cargo build -p chat` — compiles without errors
- `cargo run -p chat` — starts, shows prompt, accepts input, generates LLM response, plays audio

## Testing Strategy

- **Unit tests:** Not applicable — this is a thin integration binary with no isolated business logic. The underlying `Qwen3`, `Kokoro`, and `AudioOut` components have their own tests.
- **Integration tests:** Not applicable — requires CUDA GPU, model files, and PulseAudio.
- **Manual verification:** Run `cargo run -p chat`, type a message, verify text response appears and audio plays.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Prompt grows too large for model context window | Med | Med | Limit conversation history to last 10 turns; log warning when truncating |
| TTS playback blocks next input | Low | Low | Wait for playback to complete before showing next prompt (acceptable for experiment) |
| Model loading takes long time | Low | Low | Log progress messages during each model load step |

## Goal Verification

### Truths (what must be TRUE for the goal to be achieved)

- User can type text and receive a text response from the LLM
- LLM responses are spoken aloud through speakers via TTS
- Conversation maintains context across multiple turns
- CUDA is used for inference without any feature flag configuration by the user

### Artifacts (what must EXIST to support those truths)

- `experiments/chat/Cargo.toml` — workspace member with hardcoded CUDA feature on inference dep
- `experiments/chat/src/main.rs` — complete chat binary with LLM + TTS integration

### Key Links (critical connections that must be WIRED)

- User stdin input → Qwen3 chat template formatting → `Qwen3::forward()` → stdout display
- LLM response text → `Kokoro` TTS Sink → `AudioOut::play()` → PulseAudio speaker output
- Conversation history accumulates and is included in each new prompt

## Open Questions

- None — the design is straightforward given the existing crate APIs.
