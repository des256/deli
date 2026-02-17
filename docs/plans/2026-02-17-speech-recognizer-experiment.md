# Speech Recognizer Experiment Implementation Plan

Created: 2026-02-17
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

**Goal:** Create a `speech` experiment that captures live audio from `AudioIn`, pipes it to `SpeechRecognizer` (Whisper ASR via CUDA), and prints recognized text to stdout.

**Architecture:** A single-binary experiment at `experiments/speech/` that initializes `Inference::cuda(0)`, loads the Whisper model via `use_speech_recognizer()`, creates `AudioIn` at 16kHz (Whisper's required sample rate) with 100ms chunks, and runs a loop: accumulate audio chunks until a configurable window is reached (e.g., 3 seconds), transcribe the window, print the result, then start a new window. The program runs until Ctrl+C.

**Tech Stack:** `deli-infer` (Inference, SpeechRecognizer), `deli-audio` (AudioIn), `deli-base` (Tensor, logging), `tokio`

## Scope

### In Scope

- `experiments/speech/` crate with Cargo.toml and single binary
- Initialize `Inference::cuda(0)` and load Whisper model
- Create `AudioIn` at 16kHz with 1600 chunk_frames (100ms)
- Accumulate audio chunks into a buffer, transcribe every 3 seconds
- Print transcribed text to stdout
- Graceful Ctrl+C shutdown via `tokio::signal`

### Out of Scope

- Device selection (uses default device via `None`)
- Streaming/incremental decoding (Whisper is a batch model — it transcribes fixed-length segments)
- VAD (voice activity detection) — transcribe every window regardless of speech presence
- Tests for the experiment binary (experiments are manual-run tools — matching existing experiments which have zero tests)
- Changes to `deli-infer`, `deli-audio`, `deli-base`, or any other crate

## Prerequisites

- `deli-infer` crate with `SpeechRecognizer` (already implemented)
- `deli-audio` crate with `AudioIn` (already implemented)
- Whisper model files at `models/whisper-tiny.en/` (`model.safetensors`, `tokenizer.json`, `config.json`)
- NVIDIA GPU with CUDA drivers installed
- PulseAudio server running for audio capture

## Context for Implementer

- **Patterns to follow:** `experiments/audio/src/record.rs` for `AudioIn` usage with 16kHz/1600 chunks. `experiments/camera-pose/src/main.rs:24-51` for `#[tokio::main]` entry point with model initialization and inference loop.
- **Conventions:** Edition 2024, `deli_base::log` for logging, `Result<(), Box<dyn std::error::Error>>` return type for main, path dependencies to `../../crates/`.
- **Key files:**
  - `crates/deli-infer/src/inference.rs` — `Inference::cuda(0)`, `use_speech_recognizer(model_path, tokenizer_path, config_path)`
  - `crates/deli-infer/src/asr/recognizer.rs` — `SpeechRecognizer::transcribe(&self, audio: &BaseTensor<i16>, sample_rate: u32) -> Result<String>`
  - `crates/deli-audio/src/audio_in.rs` — `AudioIn::new(None, sample_rate, chunk_frames)`, `recv().await -> Result<Vec<i16>, AudioError>`
  - `crates/deli-base/src/tensor.rs` — `Tensor::new(shape: Vec<usize>, data: Vec<T>) -> Result<Tensor<T>, TensorError>`
- **Gotchas:**
  - `Inference::cuda(0)` requires the `cuda` feature enabled on `deli-infer`. It returns `Result<Self, InferError>`.
  - `SpeechRecognizer::transcribe()` requires exactly 16000 Hz sample rate. It will return an error for any other rate.
  - `transcribe()` takes `&BaseTensor<i16>` (a `deli_base::Tensor<i16>`). Construct it with `Tensor::new(vec![samples.len()], samples)` for a 1D tensor.
  - `transcribe()` is async — it internally uses `spawn_blocking` for the compute-heavy inference.
  - The model path defaults should point to `models/whisper-tiny.en/` relative to the project root. Use `env::var("DELI_WHISPER_MODEL")` with fallback to a reasonable default.
  - `AudioIn::new()` panics outside a tokio runtime — must use `#[tokio::main]`.
  - `AudioIn::recv()` returns `chunk_frames` samples per call. At 16kHz with 1600 chunk_frames, that's 100ms per chunk. For 3-second windows, that's 48000 samples = 30 recv() calls.

## Runtime Environment

- **Run command:** `cargo run -p speech --features cuda -- [model-dir]`
- **Manual verification:** Expected startup logs (in order): (1) "Initializing CUDA...", (2) "CUDA initialized", (3) "Model directory: models/whisper-tiny.en", (4) "Whisper model loaded", (5) "Audio capture started". Expected runtime behavior: every ~3 seconds, if speech is detected, transcription text is printed via `println!` (no prefix). On Ctrl+C, logs "Shutting down..." and exits cleanly.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create speech experiment crate and binary

**Total Tasks:** 1 | **Completed:** 1 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create speech experiment crate and binary

**Objective:** Create the `experiments/speech/` crate with Cargo.toml and the main binary that captures audio, transcribes it with Whisper, and prints results.

**Dependencies:** None

**Files:**

- Create: `experiments/speech/Cargo.toml`
- Create: `experiments/speech/src/main.rs`

**Key Decisions / Notes:**

- **Cargo.toml:** Single `[[bin]]` section — `speech` at `src/main.rs`. Dependencies: `deli-infer` (path, with `cuda` feature), `deli-audio` (path), `deli-base` (path), `tokio` (rt, rt-multi-thread, macros, signal).
- **CLI:** Optional argument for model directory path. Default to `models/whisper-tiny.en` (relative to CWD). Print usage information at startup.
- **Initialization sequence:** (1) `init_stdout_logger()`, (2) parse CLI args for model dir, (3) validate model directory: check that `model.safetensors`, `tokenizer.json`, and `config.json` exist in the directory — if any missing, print clear error listing expected files and exit with code 1, (4) log startup info including model directory path, (5) `Inference::cuda(0)?` — log "Initializing CUDA..." before and "CUDA initialized" after, (6) `inference.use_speech_recognizer(model_path, tokenizer_path, config_path)?` — log "Whisper model loaded", (7) `AudioIn::new(None, 16000, 1600)`, (8) verify audio capture by calling `audio_in.recv()` with a 5-second timeout (`tokio::time::timeout`) — if timeout, print error "Audio capture failed. Check that PulseAudio is running and a microphone is connected." and exit with code 1, (9) log "Audio capture started" (the first chunk is discarded — it just verified connectivity).
- **Main loop (interleaved recv/transcribe):** Wrap recognizer in `Arc<SpeechRecognizer>` so it can be shared with spawned tasks. Use `tokio::select!` to continuously call `audio_in.recv()` and check for Ctrl+C. Accumulate chunks into `Vec<i16>`. When accumulated samples >= 48000 (3 seconds at 16kHz): (1) if a previous transcription task is running, await it and print its result (only if non-empty after trimming, otherwise `log::debug!("No speech detected in window")`), (2) take the buffer via `std::mem::take(&mut buffer)`, (3) construct `Tensor::new(vec![samples.len()], samples)?`, (4) spawn a new `tokio::spawn` task with `Arc::clone(&recognizer)` that calls `recognizer.transcribe(&tensor, 16000).await`, (5) store the `JoinHandle` for the next iteration. This ensures `recv()` is called continuously — audio capture never blocks on transcription. The `tokio::spawn` pattern works because `SpeechRecognizer` fields (`Arc<Whisper>`, `Arc<Tokenizer>`, `Config`, `Device`) are all `Send + Sync`.
- **Transcription timing:** Log transcription duration at debug level inside the spawned task: `log::debug!("Transcription took {:?}", elapsed)`. This provides visibility into inference performance without cluttering normal output.
- **Shutdown:** Ctrl+C breaks the `tokio::select!` loop. Before exiting: if there's a pending transcription task, await it and print the result. If the buffer has >= 8000 samples (0.5 seconds), transcribe and print it before exit. Log "Shutting down..." before exit.
- **Error handling:** Use `?` operator throughout. If `Inference::cuda(0)` fails (no GPU), print a clear error. If model files are missing, the error from `use_speech_recognizer` will propagate.

**Definition of Done:**

- [ ] No diagnostics errors (linting, type checking)
- [ ] `cargo build -p speech --features cuda` compiles without errors
- [ ] Running `speech` with no arguments uses default model path and starts listening (logs "Audio capture started")
- [ ] Running `speech` with a model directory argument uses that path
- [ ] Running `speech` with an invalid model directory prints error listing expected files and exits with code 1
- [ ] Transcription runs in a `tokio::spawn` task while the main loop continues calling `recv()` — no audio chunks are dropped during inference
- [ ] Non-empty transcription results are printed to stdout with `println!`
- [ ] Empty transcriptions are logged at debug level with `log::debug!` but not printed to stdout
- [ ] Transcription duration is logged at debug level
- [ ] Ctrl+C exits cleanly with "Shutting down..." log message, transcribing any pending/partial buffer before exit

**Verify:**

- `cargo check -p speech --features cuda` — no compiler errors or warnings
- `cargo run -p speech --features cuda 2>&1 | head -10` — shows initialization logs
- `cargo run -p speech --features cuda -- models/whisper-tiny.en` — transcribes speech (manual)

## Testing Strategy

- **Unit tests:** None — experiment binary is a manual-run tool, matching the existing experiment pattern (camera-view, camera-pose, camera-viewer, audio all have zero tests).
- **Build verification:** `cargo check -p speech --features cuda` and `cargo build -p speech --features cuda` must succeed with zero errors and zero warnings.
- **Manual verification:** Run the binary, speak into microphone, verify text output matches speech.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| No CUDA GPU available | Low | High | `Inference::cuda(0)` returns `Result` — the error message from candle will indicate CUDA is not available. The binary logs "Initializing CUDA..." before the call so the user knows where it failed. |
| Model files not found at default path | Medium | Medium | Log the resolved model directory path at startup: `log::info!("Model directory: {}", model_dir)`. If `use_speech_recognizer()` fails, the error message includes the file path. Accept model dir as CLI argument for custom locations. |
| Whisper returns empty string for silence | Medium | Low | When `transcribe()` returns an empty string after trimming: (1) `log::debug!("No speech detected in window")`, (2) skip `println!`, (3) continue to next accumulation window. This keeps stdout clean while allowing debug-level verification that the loop is running. |
| AudioIn chunk dropping under load | Low | Low | The main loop uses `tokio::select!` to call `recv()` continuously — transcription runs in a separate `tokio::spawn` task, so `recv()` is never blocked by inference. The only processing per chunk is `Vec::extend` (microseconds). AudioIn's 4-chunk buffer (400ms) won't overflow since chunks are consumed every 100ms with no blocking. |

## Open Questions

- None — requirements are clear.

### Deferred Ideas

- Voice Activity Detection (VAD) to only transcribe when speech is detected
- Sliding window with overlap for better continuity between segments
- CPU fallback when CUDA is not available
- Configurable window duration via CLI flag
- Real-time streaming ASR with partial results
