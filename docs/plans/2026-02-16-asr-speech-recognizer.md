# ASR SpeechRecognizer Implementation Plan

Created: 2026-02-16
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

**Goal:** Add an `asr/` submodule to `deli-infer` containing a `SpeechRecognizer` struct that wraps a ported Candle Whisper model (tiny only) for speech-to-text transcription. Input is `Tensor<i16>` (16kHz PCM), output is `String`.

**Architecture:** Port the Whisper model code from `candle-transformers/src/models/whisper/` (Apache-2.0/MIT) and decoding logic from `candle-examples/examples/whisper/main.rs`. The model code (encoder, decoder, attention) fits in a single `model.rs` (~290 lines without tracing). Audio preprocessing (FFT, mel spectrogram) is ported from candle-transformers' `audio.rs`. A `TokenDecoder` handles the autoregressive greedy decoding loop. The public `SpeechRecognizer` ties everything together with an async `transcribe()` method via `spawn_blocking`. Mel filter data (80-bin, ~64KB) is embedded via `include_bytes!`.

**Tech Stack:** candle-core 0.9, candle-nn 0.9, tokenizers (HuggingFace), serde + serde_json (config parsing), num-traits (Float trait for audio), deli-base

## Scope

### In Scope

- Port Whisper model (AudioEncoder, TextDecoder, MultiHeadAttention, ResidualAttentionBlock) from candle-transformers
- Port audio preprocessing (FFT, mel spectrogram) from candle-transformers
- Port greedy token decoding loop from candle-examples
- `SpeechRecognizer` public API with async `transcribe()`
- Tiny model size only (hardcoded config variant available)
- English transcription (no language detection for tiny.en)
- `Inference::use_speech_recognizer()` factory method
- Embedded 80-bin mel filter data

### Out of Scope

- Model sizes beyond tiny (base, small, medium, large)
- Language detection / multilingual support
- Quantized model support (GGUF)
- Temperature-based sampling fallback (greedy only)
- Timestamps / segment-level output
- Model downloading from HuggingFace Hub
- Streaming / real-time transcription
- Translation mode

## Prerequisites

- **Already available** at `models/whisper-tiny.en/` (gitignored):
  - `model.safetensors` — model weights (~150MB)
  - `tokenizer.json` — BPE tokenizer (~2.4MB)
  - `config.json` — model config (~2KB)
  - `melfilters.bytes` — 80-bin mel filter data (~64KB, Apache-2.0/MIT from candle repo)
- The `melfilters.bytes` must be copied to `crates/deli-infer/src/asr/melfilters.bytes` for embedding via `include_bytes!` at compile time:
  ```bash
  cp models/whisper-tiny.en/melfilters.bytes crates/deli-infer/src/asr/melfilters.bytes
  ```

## Context for Implementer

- **Patterns to follow:** The `pose_detector/` module structure in `crates/deli-infer/src/pose_detector/` — separate files for model, preprocessing, public API. Error handling via `InferError` enum in `crates/deli-infer/src/error.rs:1-35`. Factory method pattern in `crates/deli-infer/src/inference.rs:27-29`.
- **Conventions:** Edition 2024. All internal model structs are `pub(crate)`. Public API only exposes `SpeechRecognizer`. The `transcribe()` method is async via `tokio::task::spawn_blocking`. Model (`Arc<Whisper>`) and tokenizer (`Arc<Tokenizer>`) are shared immutably; a fresh `TokenDecoder` with clean KV-cache is created per transcription call inside `spawn_blocking`.
- **Key files the implementer must read:**
  - `crates/deli-infer/src/pose_detector/detector.rs` — Pattern for public API struct, spawn_blocking usage
  - `crates/deli-infer/src/inference.rs` — Where to add `use_speech_recognizer()` factory
  - `crates/deli-infer/src/error.rs` — Shared error type
  - `crates/deli-base/src/tensor.rs` — `Tensor<T>` struct (shape + data)
- **Gotchas:**
  - The candle-transformers model uses `crate::models::with_tracing::{linear, linear_no_bias}` — replace with `candle_nn::linear()` and `candle_nn::linear_no_bias()` directly. Remove all tracing spans.
  - `TextDecoder` has KV-cache that must be reset between transcriptions — solved by creating a fresh `TokenDecoder` per `transcribe()` call rather than sharing mutable state.
  - Whisper expects PCM at exactly 16kHz — `transcribe()` takes `sample_rate: u32` parameter and validates it equals 16000, returning error otherwise.
  - The audio FFT uses `num_traits::Float` trait — need `num-traits` dependency.
  - Mel filters are raw little-endian f32 bytes — read with `f32::from_le_bytes()` in groups of 4.
  - The `tokenizers` crate is heavy (~50 deps) but unavoidable for Whisper's BPE tokenizer format.
- **Domain context:**
  - Whisper processes audio in 30-second chunks. Input longer than 30s is processed in segments.
  - The mel spectrogram converts time-domain PCM to frequency-domain features (80 mel bins for tiny).
  - Decoding is autoregressive: the decoder generates one token at a time, feeding previous tokens back as input.
  - Special tokens: `<|startoftranscript|>`, `<|transcribe|>`, `<|notimestamps|>`, `<|endoftext|>`.
  - The encoder processes audio features once; the decoder attends to them via cross-attention at each step.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Config, constants, and dependencies
- [x] Task 2: Audio preprocessing (PCM to mel spectrogram)
- [x] Task 3: Whisper model (encoder + decoder)
- [x] Task 4: Token decoder (greedy decoding loop)
- [x] Task 5: SpeechRecognizer public API

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Config, constants, and dependencies

**Objective:** Define the Whisper `Config` struct, audio/tokenizer constants, and add all new crate dependencies.

**Dependencies:** None

**Files:**

- Create: `crates/deli-infer/src/asr/mod.rs`
- Create: `crates/deli-infer/src/asr/config.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add `mod asr;`)
- Modify: `crates/deli-infer/Cargo.toml` (add tokenizers, serde, serde_json, num-traits)
- Test: `crates/deli-infer/tests/asr_config_test.rs`

**Key Decisions / Notes:**

- Port `Config` struct from `candle-transformers/src/models/whisper/mod.rs`. Derive `Debug, Clone, Deserialize`. Fields: `num_mel_bins`, `max_source_positions`, `d_model`, `encoder_attention_heads`, `encoder_layers`, `vocab_size`, `max_target_positions`, `decoder_attention_heads`, `decoder_layers`, `suppress_tokens` (Vec<u32>).
- Add `Config::tiny_en()` constructor that returns hardcoded config for tiny.en variant (num_mel_bins=80, d_model=384, encoder_layers=4, decoder_layers=4, encoder_attention_heads=6, decoder_attention_heads=6, max_source_positions=1500, max_target_positions=448, vocab_size=51865, suppress_tokens from Whisper spec).
- Port constants: `SAMPLE_RATE=16000`, `N_FFT=400`, `HOP_LENGTH=160`, `CHUNK_LENGTH=30`, `N_SAMPLES=480000`, `N_FRAMES=3000`, `NO_SPEECH_THRESHOLD=0.6`, `LOGPROB_THRESHOLD=-1.0`, `COMPRESSION_RATIO_THRESHOLD=2.4`.
- Port special token constants: `SOT_TOKEN`, `TRANSCRIBE_TOKEN`, `NO_TIMESTAMPS_TOKEN`, `EOT_TOKEN`, `NO_SPEECH_TOKENS`.
- `asr/mod.rs` declares submodules and re-exports `SpeechRecognizer` (placeholder for now).
- Cargo.toml additions: `tokenizers = "0.21"`, `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`, `num-traits = "0.2"`.

**Definition of Done:**

- [ ] All tests pass (Config deserializes from JSON string, `tiny_en()` returns valid config)
- [ ] No diagnostics errors
- [ ] Constants match Whisper spec values (SAMPLE_RATE=16000, N_FFT=400, etc.)
- [ ] `cargo build -p deli-infer` succeeds with new dependencies

**Verify:**

- `cargo test -p deli-infer --test asr_config_test -q` — config tests pass
- `cargo build -p deli-infer -q` — compiles with new deps

---

### Task 2: Audio preprocessing (PCM to mel spectrogram)

**Objective:** Port the FFT/mel spectrogram computation from candle-transformers and embed the 80-bin mel filter data.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-infer/src/asr/audio.rs`
- Create: `crates/deli-infer/src/asr/melfilters.bytes` (binary, copy from candle repo)
- Modify: `crates/deli-infer/src/asr/mod.rs` (add `mod audio;`)
- Test: `crates/deli-infer/tests/asr_audio_test.rs`

**Key Decisions / Notes:**

- Port `fft()`, `dft()`, `log_mel_spectrogram_w()`, `log_mel_spectrogram_()`, `pcm_to_mel()` from `candle-transformers/src/models/whisper/audio.rs`. The code uses `num_traits::Float` for generic f32/f64 support.
- Use single-threaded mel spectrogram computation (remove `std::thread::scope` parallelism from candle source). FFT on 30s audio is ~10ms on modern CPU — parallelism adds complexity without meaningful speedup, and nested thread spawning inside `spawn_blocking` risks thread pool exhaustion under concurrent load.
- Embed mel filters via `include_bytes!("melfilters.bytes")` — load as `Vec<f32>` using `f32::from_le_bytes`.
- Provide `pub(crate) fn load_mel_filters() -> Vec<f32>` for loading embedded filters.
- Provide `pub(crate) fn pcm_to_mel(config: &Config, samples: &[f32]) -> Vec<f32>` that wraps the internal function with embedded filters.
- The `melfilters.bytes` file must be copied from `candle-examples/examples/whisper/melfilters.bytes` in the HuggingFace candle repo. Download URL: `https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/melfilters.bytes`.
- Attribution comment at top of audio.rs referencing Apache-2.0/MIT license.

**Definition of Done:**

- [ ] All tests pass (FFT of known input matches expected, mel spectrogram output has correct length)
- [ ] No diagnostics errors
- [ ] `pcm_to_mel` with 480000 samples (30s at 16kHz) produces correct shape (80 × 3000 = 240000 values)
- [ ] Mel filters load correctly (non-zero values, expected count for 80 bins)
- [ ] FFT correctness verified with known audio: generate 1-second 440Hz sine wave (16000 samples), compute mel spectrogram, verify energy peak is in the correct frequency bin range for 440Hz

**Verify:**

- `cargo test -p deli-infer --test asr_audio_test -q` — audio tests pass

---

### Task 3: Whisper model (encoder + decoder)

**Objective:** Port the Whisper neural network architecture (AudioEncoder, TextDecoder, MultiHeadAttention, ResidualAttentionBlock) from candle-transformers.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-infer/src/asr/model.rs`
- Modify: `crates/deli-infer/src/asr/mod.rs` (add `mod model;`)
- Test: `crates/deli-infer/tests/asr_model_test.rs`

**Key Decisions / Notes:**

- Port from `candle-transformers/src/models/whisper/model.rs`. Remove all tracing spans. Replace `crate::models::with_tracing::{linear, linear_no_bias, Linear}` with `candle_nn::{linear, linear_no_bias}` and `candle_nn::Linear`.
- Structs to port (all `pub(crate)`): `MultiHeadAttention`, `ResidualAttentionBlock`, `AudioEncoder`, `TextDecoder`, `Whisper`.
- Helper functions to port: `conv1d()`, `layer_norm()`, `sinusoids()`.
- `Whisper` struct is the top-level model: has `encoder: AudioEncoder`, `decoder: TextDecoder`, `config: Config`. Loaded via `Whisper::load(vb, config)`.
- VarBuilder paths match HuggingFace model: `model.encoder.conv1`, `model.encoder.layers.N`, `model.decoder.embed_tokens`, `model.decoder.layers.N`, etc.
- `TextDecoder::final_linear` reuses the embedding weights (weight tying) — no separate linear layer.
- `MultiHeadAttention` has KV-cache for efficient autoregressive decoding. `reset_kv_cache()` must be called between different audio inputs.
- Estimated ~290 lines without tracing. If it exceeds 300, split into `model.rs` (Whisper + helpers) and `attention.rs` (MultiHeadAttention + ResidualAttentionBlock).
- Attribution comment at top referencing Apache-2.0/MIT license.

**Definition of Done:**

- [ ] All tests pass (encoder forward, decoder forward, full Whisper forward with VarMap)
- [ ] No diagnostics errors
- [ ] Encoder: input `[1, 80, 3000]` → output `[1, 1500, 384]` for tiny config
- [ ] Decoder: token input `[1, 4]` + encoder output → logits `[1, 4, 51865]` for tiny config
- [ ] File stays under 300 lines (or split into 2 files)
- [ ] If `models/whisper-tiny.en/model.safetensors` exists, verify model loads from real weights without errors (skip test if file absent)

**Verify:**

- `cargo test -p deli-infer --test asr_model_test -q` — model tests pass

---

### Task 4: Token decoder (greedy decoding loop)

**Objective:** Port the autoregressive decoding loop from candle-examples that generates text tokens from mel spectrogram features.

**Dependencies:** Task 1, Task 3

**Files:**

- Create: `crates/deli-infer/src/asr/token_decoder.rs`
- Modify: `crates/deli-infer/src/asr/mod.rs` (add `mod token_decoder;`)
- Test: `crates/deli-infer/tests/asr_token_decoder_test.rs`

**Key Decisions / Notes:**

- Port `Decoder` struct and `decode()` method from `candle-examples/examples/whisper/main.rs`. Rename to `TokenDecoder` to avoid confusion with the neural network TextDecoder.
- Greedy decoding only (temperature=0): argmax over logits at each step. No temperature sampling, no `rand` dependency.
- `TokenDecoder` fields: `model: Whisper` (mut — KV-cache), `tokenizer: Tokenizer`, `suppress_tokens: Tensor`, `sot_token: u32`, `transcribe_token: u32`, `eot_token: u32`, `no_speech_token: u32`, `no_timestamps_token: u32`.
- `TokenDecoder::new(model, tokenizer, device, config)` sets up token IDs and suppress_tokens tensor.
- `TokenDecoder::decode(&mut self, mel: &Tensor) -> Result<DecodingResult>` runs encoder forward, then autoregressive decoder loop until EOT or max_target_positions/2. `DecodingResult` includes `text: String` and `truncated: bool` (true if max token limit reached without EOT).
- `TokenDecoder::run(&mut self, mel: &Tensor) -> Result<String>` processes mel in 30-second chunks (N_FRAMES=3000 frames per chunk), concatenating decoded text. Skips segments with no_speech_prob > threshold. For audio not evenly divisible by 30s, the final segment is zero-padded to N_SAMPLES. Logs a warning if any segment was truncated (no EOT produced).
- `pub(crate) fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32>` helper for looking up special tokens.
- The `tokenizers` crate's `Tokenizer::from_file()` loads `tokenizer.json`.

**Definition of Done:**

- [ ] All tests pass (token_id lookup, decoder produces tokens ending in EOT with VarMap model)
- [ ] No diagnostics errors
- [ ] Greedy decode loop terminates on EOT token or max_target_positions/2 limit
- [ ] `DecodingResult.truncated` is true when max limit reached without EOT
- [ ] `run()` handles multi-chunk audio (>30s) by processing N_FRAMES segments
- [ ] Final segment of non-30s-aligned audio is zero-padded correctly
- [ ] No-speech segments are skipped when no_speech_prob > threshold

**Verify:**

- `cargo test -p deli-infer --test asr_token_decoder_test -q` — token decoder tests pass

---

### Task 5: SpeechRecognizer public API

**Objective:** Create the `SpeechRecognizer` struct that loads a Whisper model and provides async transcription. Wire up `Inference::use_speech_recognizer()`.

**Dependencies:** Task 2, Task 4

**Files:**

- Create: `crates/deli-infer/src/asr/recognizer.rs`
- Modify: `crates/deli-infer/src/asr/mod.rs` (add `mod recognizer;`, re-export `SpeechRecognizer`)
- Modify: `crates/deli-infer/src/inference.rs` (add `use_speech_recognizer()`)
- Modify: `crates/deli-infer/src/lib.rs` (add `SpeechRecognizer` to public re-exports)
- Test: `crates/deli-infer/tests/asr_recognizer_test.rs`

**Key Decisions / Notes:**

- `SpeechRecognizer` fields: `model: Arc<Whisper>`, `tokenizer: Arc<Tokenizer>`, `config: Config`, `device: Device`. No Mutex — a fresh `TokenDecoder` with clean KV-cache is constructed inside each `spawn_blocking` call. This enables true parallel transcriptions and avoids async-Mutex deadlock risk.
- Constructor: `SpeechRecognizer::new(model_path, tokenizer_path, config_path, device)` loads safetensors, tokenizer, config. Validates model format by checking that the safetensors file contains expected keys (e.g., `model.encoder.conv1.weight`, `model.decoder.embed_tokens.weight`) — returns `InferError` with helpful message if keys are missing (wrong model type/format).
- Alternative constructor: `SpeechRecognizer::new_with_config(model_path, tokenizer_path, config, device)` for using `Config::tiny_en()` directly.
- `pub async fn transcribe(&self, audio: &Tensor<i16>, sample_rate: u32) -> Result<String, InferError>`:
  1. Validate sample_rate == 16000, return `InferError` if not (resampling is caller's responsibility)
  2. Validate input: 1D tensor, non-empty
  3. Convert i16 to f32 (`sample as f32 / 32768.0`)
  4. Inside `spawn_blocking`: construct fresh `TokenDecoder` from cloned `Arc<Whisper>` + `Arc<Tokenizer>`, compute mel spectrogram, run TokenDecoder::run, return text
- `Inference::use_speech_recognizer(model_path, tokenizer_path, config_path)` factory creates SpeechRecognizer on the inference device.
- Public re-exports from `lib.rs`: add `SpeechRecognizer`.

**Definition of Done:**

- [ ] All tests pass (construction with VarMap, transcribe returns String with VarMap model)
- [ ] No diagnostics errors
- [ ] `Inference::cpu().use_speech_recognizer(model, tokenizer, config)` constructs successfully
- [ ] `transcribe()` accepts `Tensor<i16>` + `sample_rate: u32` and returns `String`
- [ ] `transcribe()` rejects sample_rate != 16000 with clear error
- [ ] `transcribe()` is async and runs inference off the async runtime
- [ ] Input validation rejects non-1D tensors and empty audio
- [ ] Constructor validates safetensors keys match Whisper format (returns error on wrong model type)
- [ ] Public API surface: `SpeechRecognizer` added to `lib.rs` re-exports
- [ ] `cargo build -p deli-infer -q` succeeds with clean public API
- [ ] If `models/whisper-tiny.en/` files exist, integration test loads real model and produces non-empty transcription (skip if files absent)

**Verify:**

- `cargo test -p deli-infer --test asr_recognizer_test -q` — recognizer tests pass
- `cargo test -p deli-infer -q` — all crate tests pass
- `cargo build -p deli-infer -q` — final build clean

## Testing Strategy

- **Unit tests:** Each task has a dedicated test file. Model tested with VarMap random weights (verify shapes). Audio preprocessing tested with known FFT inputs and mel spectrogram dimensions. Token decoder tested with VarMap model (verify termination and token generation). Config tested with JSON deserialization.
- **Integration tests:** Task 5 test exercises the full pipeline: Inference → SpeechRecognizer (VarMap) → transcribe synthetic Tensor<i16> → verify String output. Tests with real Whisper model files verify end-to-end accuracy — these automatically skip (not fail) if `models/whisper-tiny.en/` files are absent, and run when present.
- **Manual verification:** Download whisper-tiny.en model files, load a test WAV, run transcribe, verify recognizable English text output.

## Runtime Environment

- **Build command:** `cargo build -p deli-infer`
- **Unit tests:** `cargo test -p deli-infer` (uses VarMap synthetic models, no external files needed)
- **Integration test (requires model files):** `cargo test -p deli-infer --test asr_recognizer_test -- --ignored --nocapture`
- **Model file setup for integration tests:**
  - Download from HuggingFace: `openai/whisper-tiny.en`
  - Files needed: `model.safetensors`, `tokenizer.json`, `config.json`
  - Place in: `models/whisper-tiny.en/` (gitignored)
- **Mel filters data:** Copy `melfilters.bytes` to `crates/deli-infer/src/asr/melfilters.bytes`
- **Diagnostics:** `cargo clippy -p deli-infer`

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model weight key names don't match port | Med | High | Use same `vb.pp()` prefix strings as candle-transformers verbatim; constructor validates key presence (e.g., `model.encoder.conv1.weight`) before building model; test with real model if available (skip-if-absent) |
| Wrong model type loaded (e.g., distil-whisper, non-pose variant) | Med | High | Constructor probes safetensors for expected keys, returns `InferError` with message: "Expected OpenAI Whisper format from openai/whisper-tiny.en" on mismatch |
| `tokenizers` crate version incompatibility with HF tokenizer.json format | Low | Med | Pin `tokenizers = "0.21"` which supports current Whisper tokenizer format |
| Audio FFT produces incorrect mel spectrogram | Low | High | Port FFT tests from candle-transformers audio.rs; test pcm_to_mel output dimensions match expected 80 × 3000 |
| Greedy decoding loops infinitely (no EOT token generated) | Low | High | Hard limit at `max_target_positions / 2` tokens per segment; test with VarMap to verify loop termination |
| `tokenizers` adds ~50 transitive dependencies | High | Low | Unavoidable for Whisper BPE tokenizer format; no lighter alternative exists |
| File exceeds 300-line limit (model.rs) | Med | Low | Pre-estimated at ~290 lines; if over, split MultiHeadAttention + ResidualAttentionBlock into `attention.rs` |

## Open Questions

- None — all design decisions resolved.

### Deferred Ideas

- Multi-language support with language detection
- Temperature-based sampling with fallback for robustness
- Timestamp/segment-level transcription output
- Streaming transcription (process audio as it arrives)
- Support for larger model sizes (base, small, medium, large)
- Quantized model support (GGUF format)
- Batch transcription API for processing multiple audio inputs efficiently
- Cancellation token support for aborting in-progress transcriptions
