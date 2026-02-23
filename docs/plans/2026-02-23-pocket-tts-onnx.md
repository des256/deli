# Pocket TTS ONNX Inference Implementation Plan

Created: 2026-02-23
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
> **Worktree:** No — working directly on current branch

## Summary

**Goal:** Implement Pocket TTS (Kyutai, 100M params, MIT license) text-to-speech inference via ONNX Runtime in the `inference` crate, following the same Sink/Stream pattern as Kokoro TTS. Support voice cloning via audio prompt and streaming audio output.

**Architecture:** Five ONNX models orchestrated by a `PocketCore` struct: text_conditioner (tokenize→embed), flow_lm_main (stateful transformer backbone with KV-cache), flow_lm_flow (stateless flow matching ODE solver), mimi_encoder (audio→latents for voice cloning), and mimi_decoder (stateful neural codec producing audio). The autoregressive generation loop and LSD (Lagrangian Self Distillation) decode inner loop run in Rust. `PocketTts` wraps the core with async Sink/Stream APIs identical to Kokoro.

**Tech Stack:** Custom `onnx` crate (ONNX Runtime FFI), `tokenizers` crate (SentencePiece tokenizer loading), `audio` crate for AudioSample/AudioData types.

## Scope

### In Scope

- Add `bool` TensorElement support to `onnx` crate (needed for mimi_decoder state tensors)
- Prepare tokenizer data file (convert SentencePiece .model → tokenizer.json)
- `PocketCore` struct: 5 ONNX sessions, state tensor management, AR loop, LSD decode
- `PocketTts` struct: `Sink<String>` + `Stream<Item = Result<AudioSample>>`, text preparation
- Voice cloning via audio file (mimi_encoder path)
- `Inference::use_pocket_tts()` factory method
- Integration tests in `inference/src/tts/pocket/tests/`

### Out of Scope

- Pre-computed voice state loading (PyTorch safetensors format, incompatible with ONNX state layout)
- GPU-accelerated inference (CPU-first, same as upstream Pocket TTS design)
- Sentence splitting for very long texts (initial impl handles single chunks)
- INT8 vs FP32 model selection (we use the INT8 models the user provided)

## Prerequisites

- ONNX model files at `data/pocket/`: `text_conditioner.onnx`, `flow_lm_main_int8.onnx`, `flow_lm_flow_int8.onnx`, `mimi_encoder.onnx`, `mimi_decoder_int8.onnx` (confirmed present)
- Tokenizer: SentencePiece model from `hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model` must be converted to `data/pocket/tokenizer.json` (Task 2 handles this)
- A voice audio file at `data/pocket/voice.wav` for testing (WAV format, mono, any speaker, ~5-10s of clear speech). Can be any recording — LibriSpeech sample, personal recording, or download from a TTS demo site.

## Context for Implementer

- **Patterns to follow:** Kokoro TTS (`crates/inference/src/tts/kokoro/kokoro.rs`) for Sink/Stream pattern, text preparation, async spawn_blocking. Sherpa ASR (`crates/inference/src/asr/sherpa/asrcore.rs`) for stateful multi-session ONNX with state carry between runs.
- **Conventions:** Module structure `tts/pocket/mod.rs` re-exports public type. Internal files are `pub(crate)`. Error handling uses `crate::error::{InferError, Result}`. Factory methods on `Inference` follow `use_<name>()` naming.
- **Key files:**
  - `crates/inference/src/inference.rs` — `Inference` struct, `onnx_session()`, `use_*()` methods
  - `crates/inference/src/tts/kokoro/kokoro.rs` — Reference Sink/Stream TTS pattern
  - `crates/inference/src/asr/sherpa/asrcore.rs` — Reference stateful ONNX state management
  - `crates/onnx/src/value.rs` — `Value::from_slice`, `Value::zeros`, `TensorElement` trait
  - `crates/onnx/src/session.rs` — `Session::run()`, `input_name()`, `input_shape()`, `input_element_type()`
- **Gotchas:**
  - **Mimi decoder has BOOL state tensors** — the `onnx` crate's `Value` currently only supports f32/f64/i64/i32. Must add bool before mimi_decoder can work.
  - **NaN for BOS token** — flow_lm_main expects `NaN` values in the sequence input to signal beginning-of-sequence. `Value::from_slice::<f32>` handles NaN natively.
  - **State tensors with shape [0]** — some flow_lm states (`current_end`) have shape `[0]` (empty). `Value::from_slice` with empty data must work correctly.
  - **Voice conditioning is mandatory** — Pocket TTS always requires voice conditioning. Audio → mimi_encoder → latents → flow_lm_main conditioning step → KV-cache state. There's no "voiceless" mode.
  - **emb_std/emb_mean are baked into mimi_decoder ONNX** — The mimi_decoder wrapper applies `latent * emb_std + emb_mean` internally, so we pass raw latents from flow_lm to mimi_decoder.
- **Domain context:** The inference pipeline is:
  1. Tokenize text → `token_ids: [1, seq] i64`
  2. Text conditioner: `token_ids` → `embeddings: [1, seq, 1024] f32`
  3. Voice conditioning (mimi_encoder): `audio: [1, 1, samples] f32` → `latents: [1, T, 1024] f32`
  4. Concatenate: `combined = cat([text_embeddings, voice_latents], dim=1)`
  5. FlowLM main (conditioning step): `sequence=[1,0,32]`, `text_embeddings=combined`, initial states → updated states (output ignored)
  6. AR loop (per generation step):
     a. FlowLM main: `sequence=[1,1,32]` (NaN for first, prev latent for rest), `text_embeddings=[1,0,1024]` (empty), states → `conditioning [1,1024]`, `eos_logit [1,1]`, updated states
     b. LSD decode (N steps): for i in 0..N: flow_lm_flow(`c=conditioning, s=i/N, t=(i+1)/N, x`) → update x. Initial x = noise from N(0,temp) clamped.
     c. Mimi decoder: `latent=[1,1,32]`, states → `audio_frame [1,1,samples]`, updated states → yield audio
     d. Check EOS from step (a), count frames_after_eos

### ONNX Model I/O Signatures

**text_conditioner.onnx:**
- Input: `token_ids: [1, seq_len] INT64`
- Output: `embeddings: [1, seq_len, 1024] FLOAT`

**flow_lm_main_int8.onnx:**
- Inputs: `sequence: [1, seq_len, 32] FLOAT`, `text_embeddings: [1, text_len, 1024] FLOAT`, 18 state tensors (6 layers × {cache: [2,1,1000,16,64] FLOAT, current_end: [0] FLOAT, step: [1] INT64})
- Outputs: `conditioning: [1, 1024] FLOAT`, `eos_logit: [1, 1] FLOAT`, 18 out_state tensors

**flow_lm_flow_int8.onnx:**
- Inputs: `c: [batch, 1024] FLOAT`, `s: [batch, 1] FLOAT`, `t: [batch, 1] FLOAT`, `x: [batch, 32] FLOAT`
- Output: `flow_dir: [batch, 32] FLOAT`

**mimi_encoder.onnx:**
- Input: `audio: [1, 1, audio_len] FLOAT`
- Output: `latents: [1, frames, 1024] FLOAT`

**mimi_decoder_int8.onnx:**
- Inputs: `latent: [1, seq_len, 32] FLOAT`, 56 state tensors (mixed FLOAT/INT64/BOOL)
- Outputs: `audio_frame: [1, 1, samples] FLOAT`, 56 out_state tensors

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add bool TensorElement to onnx crate
- [x] Task 2: Prepare tokenizer and create pocket module skeleton
- [x] Task 3: Implement PocketCore inference engine
- [x] Task 4: Implement PocketTts Sink/Stream API and integration tests

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add bool TensorElement to onnx crate

**Objective:** Add `bool` support to the `Value` type so mimi_decoder state tensors (BOOL dtype) can be created and extracted.

**Dependencies:** None

**Files:**

- Modify: `crates/onnx/src/value.rs`
- Test: `crates/onnx/src/value.rs` (inline test)

**Key Decisions / Notes:**

- Add `impl sealed::Sealed for bool {}` and `impl TensorElement for bool { fn element_type() -> Bool }`.
- ONNX Runtime represents bools as single bytes (like u8). Rust `bool` is also 1 byte with values 0/1, so `from_slice`/`extract_tensor` work via pointer cast without conversion.
- Update the `initialize_states` pattern in asrcore.rs to handle Bool type. Actually, the state initialization in the pocket module will handle all three types (f32/i64/bool), so this is just adding the trait impl.
- Add test: `assert_eq!(bool::element_type(), ONNXTensorElementDataType::Bool)`.
- Also test empty tensor (shape `[0]`): `Value::from_slice::<f32>(&[0], &[])` — needed for flow_lm `current_end` states. If ONNX Runtime rejects zero-length buffers, add special-case handling (e.g., 1-byte backing buffer).

**Definition of Done:**

- [ ] `bool` implements `TensorElement` with element type `Bool`
- [ ] `Value::from_slice::<bool>(&[1], &[true])` creates a valid tensor
- [ ] `Value::zeros::<bool>(&[1])` creates a tensor containing `false`
- [ ] `Value::zeros::<bool>(&[1, 10])` round-trips: extract returns all `false`
- [ ] Empty tensor `Value::from_slice::<f32>(&[0], &[])` creates without error
- [ ] Existing tests still pass

**Verify:**

- `cargo test -p onnx -q` — all onnx crate tests pass
- `cargo check -p onnx` — no compile errors

### Task 2: Prepare tokenizer and create pocket module skeleton

**Objective:** Download the SentencePiece tokenizer model, convert to tokenizer.json for use with the `tokenizers` crate, and create the module skeleton files.

**Dependencies:** None (parallel with Task 1)

**Files:**

- Create: `data/pocket/tokenizer.json` (converted from SentencePiece .model)
- Create: `crates/inference/src/tts/pocket/mod.rs`
- Modify: `crates/inference/src/tts/mod.rs` (add `pub mod pocket;`)

**Key Decisions / Notes:**

- Download tokenizer.model from HuggingFace: `huggingface-cli download kyutai/pocket-tts-without-voice-cloning tokenizer.model --local-dir /tmp/pocket-tokenizer`
- Convert with Python: load via `tokenizers` library's `SentencePieceUnigramTokenizer` or use `from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained("kyutai/pocket-tts-without-voice-cloning"); t.backend_tokenizer.save("data/pocket/tokenizer.json")`
- The tokenizer has vocab_size=4000, uses SentencePiece Unigram model.
- `mod.rs` declares submodules and re-exports `PocketTts`.
- **Tokenizer validation:** After conversion, verify the tokenizer produces expected token IDs by tokenizing a test string (e.g., "Hello world.") and checking: (1) output contains only integer IDs in range [0, 3999], (2) vocab_size matches 4000, (3) tokenizer does not insert unexpected BOS/EOS tokens by default (the pocket-tts Python code adds tokens manually). If conversion produces wrong behavior, use the `sentencepiece` crate directly as fallback.
- **Voice audio file:** Download a reference voice sample or use any WAV recording. Place at `data/pocket/voice.wav`. Must be mono, any sample rate (resampled to 24kHz in code if needed), 5-10 seconds of clear speech.

**Definition of Done:**

- [ ] `data/pocket/tokenizer.json` exists and can be loaded by `tokenizers::Tokenizer::from_file()`
- [ ] Tokenizer vocab_size = 4000
- [ ] Tokenizer encodes "Hello world." to valid token IDs (all in range [0, 3999])
- [ ] `data/pocket/voice.wav` exists (mono WAV, 5-10s speech)
- [ ] `crates/inference/src/tts/pocket/mod.rs` exists with module declarations
- [ ] `crates/inference/src/tts/mod.rs` includes `pub mod pocket;`
- [ ] `cargo check -p inference` compiles

**Verify:**

- `cargo check -p inference` — no compile errors
- `ls data/pocket/tokenizer.json` — file exists

### Task 3: Implement PocketCore inference engine

**Objective:** Implement the core inference engine that orchestrates all 5 ONNX sessions, manages state tensors, and implements the autoregressive generation loop with LSD decode.

**Dependencies:** Task 1, Task 2

**Files:**

- Create: `crates/inference/src/tts/pocket/core.rs`
- Modify: `crates/inference/src/tts/pocket/mod.rs` (add `pub(crate) mod core;`)
- Modify: `crates/inference/Cargo.toml` (add `rand = "0.8"` and `rand_distr = "0.4"` dependencies)

**Key Decisions / Notes:**

- `PocketCore` struct holds: `text_conditioner: Session`, `flow_main: Session`, `flow_step: Session`, `mimi_encoder: Session`, `mimi_decoder: Session`, plus `flow_state_names: Vec<String>`, `flow_state_output_names: Vec<String>`, `mimi_state_names: Vec<String>`, `mimi_state_output_names: Vec<String>`.
- **State discovery:** At construction, iterate model inputs to discover state tensor names (everything except `sequence`/`text_embeddings` for flow_main, everything except `latent` for mimi_decoder). Output names use `out_` prefix. This follows the Sherpa `discover_encoder_states` pattern in `asrcore.rs:180-203`.
- **State initialization:** Create zero-filled tensors matching each state input's shape and element type. Explicitly match `Float` → `Value::zeros::<f32>`, `Int64` → `Value::zeros::<i64>`, `Bool` → `Value::zeros::<bool>` (do NOT default to f32 for unknown types — error instead). States with shape `[0]` produce empty tensors via `Value::from_slice::<f32>(&[0], &[])`.
- **`encode_voice(&mut self, audio: &[f32], sample_rate: usize) -> Result<Vec<f32>>`**: Run mimi_encoder on audio, return latent embeddings `[1, T, 1024]`.
- **`condition_voice(&mut self, voice_latents: &[f32], latent_frames: usize) -> Result<()>`**: Feed voice latents as `text_embeddings` into flow_main with empty sequence `[1,0,32]`. This updates KV-cache state. Ignore conditioning/eos outputs.
- **`condition_text(&mut self, embeddings: &[f32], seq_len: usize) -> Result<()>`**: Same pattern — feed text embeddings into flow_main conditioning step.
- **`generate_step(&mut self) -> Result<(Vec<f32>, bool)>`**: One AR step: run flow_main with sequence `[1,1,32]` (NaN or prev latent) + empty text, get conditioning + eos. Run LSD decode (10 steps default). Return (latent_32d, is_eos).
- **`decode_audio(&mut self, latent: &[f32]) -> Result<Vec<f32>>`**: Run mimi_decoder on `[1,1,32]` latent, return audio samples.
- **`run_text_conditioner(&mut self, token_ids: &[i64]) -> Result<Vec<f32>>`**: Simple session.run, return embeddings.
- **LSD decode implementation** (in a private method): Initialize noise `x ~ N(0, temp)` clamped to `[-noise_clamp, noise_clamp]`. For `i` in `0..num_steps`: compute `s = i/N`, `t = (i+1)/N`, call flow_step(c, s, t, x) → flow_dir, update `x += flow_dir / N`. Return x. Use `rand` crate for noise generation.
- **Constants:** `DEFAULT_TEMPERATURE = 0.9`, `DEFAULT_LSD_STEPS = 10`, `DEFAULT_NOISE_CLAMP = 10.0`, `DEFAULT_EOS_THRESHOLD = -4.0`, `SAMPLE_RATE = 24000`, `LATENT_DIM = 32`, `CONDITIONING_DIM = 1024`.
- **NaN handling:** First AR step uses `[f32::NAN; 32]` as input sequence. Subsequent steps use the previous latent output. NaN is standard IEEE-754 and the KevinAHM ONNX export project uses NaN explicitly. If ONNX Runtime rejects NaN for any reason, the integration test in Task 4 will catch it immediately.
- **State snapshotting for voice conditioning:** After voice+text conditioning is complete, extract all flow_main state and mimi_decoder state tensors to Rust `Vec<u8>` (raw bytes). Store these as the "voice-conditioned snapshot." For each new utterance, recreate `Value` tensors from the stored bytes via `Value::from_slice()`. This avoids needing `Clone` on `Value` (which holds raw OrtValue pointers).
- **EOS threshold (-4.0)** comes from the upstream Python implementation in `KevinAHM/pocket-tts-onnx-export`. Add debug logging for EOS logit values during integration tests to validate this threshold. Add max_tokens cap (1000 by default) to prevent infinite loops.

**Definition of Done:**

- [ ] `PocketCore::new()` loads all 5 sessions and discovers state tensor names
- [ ] State initialization explicitly matches Float/Int64/Bool types (no f32 default for unknown types)
- [ ] Empty state tensors (shape [0]) initialize without error
- [ ] `rand` and `rand_distr` dependencies added to `crates/inference/Cargo.toml`
- [ ] `run_text_conditioner()` returns embeddings with correct shape [1, seq, 1024]
- [ ] `generate_step()` executes one AR step and returns a 32-dim latent + EOS flag
- [ ] `decode_audio()` converts latent to audio samples via mimi_decoder
- [ ] LSD decode loop correctly iterates N steps calling flow_step
- [ ] State tensors carry correctly: integration test runs `generate_step()` twice, second call produces different latents than first, no crash from state shape/type mismatch
- [ ] State snapshot/restore mechanism works: extract conditioned states to bytes, recreate Values from bytes

**Verify:**

- `cargo check -p inference` — compiles clean
- `cargo test -p inference -- pocket -q` — unit tests pass

### Task 4: Implement PocketTts Sink/Stream API and integration tests

**Objective:** Create the public `PocketTts` struct with async Sink/Stream interface, text preparation logic, tokenization, and full integration tests.

**Dependencies:** Task 3

**Files:**

- Create: `crates/inference/src/tts/pocket/pocket.rs`
- Create: `crates/inference/src/tts/pocket/tests/pocket_tts_test.rs`
- Modify: `crates/inference/src/tts/pocket/mod.rs` (add pocket.rs module, test module, re-export PocketTts)
- Modify: `crates/inference/src/inference.rs` (add `use_pocket_tts()` factory method)
- Modify: `crates/inference/src/tts/mod.rs` (add `PocketTts` to re-exports)
- Modify: `crates/inference/src/lib.rs` (add `PocketTts` to public exports if needed)
**Key Decisions / Notes:**

- **`PocketTts` struct:** Holds `core: Arc<Mutex<PocketCore>>`, `tokenizer: Arc<tokenizers::Tokenizer>`, pending queue, inflight future, stream_waker, closed flag. Same pattern as Kokoro.
- **Constructor `PocketTts::new(sessions..., tokenizer_path, voice_audio_path)`**: Load tokenizer from JSON. Initialize PocketCore. Run voice conditioning (mimi_encoder → flow_main conditioning). Snapshot conditioned state (via PocketCore's state snapshot mechanism from Task 3).
- **Text preparation** (port from Python `prepare_text_prompt`):
  - Strip whitespace, replace newlines with spaces
  - Capitalize first character
  - Add period if text ends with alphanumeric
  - Pad short texts (<5 words) with 8 leading spaces
  - Determine `frames_after_eos` (3 for ≤4 words, 1 otherwise) + 2 padding
- **`start_synthesis(text: String)`**: Spawns `spawn_blocking` closure that: restores voice-conditioned state snapshot → prepares text → tokenizes → runs text conditioner → conditions text → AR loop until EOS or max_tokens → collects audio chunks → returns concatenated AudioSample.
- **Sink/Stream impl:** Identical to Kokoro pattern. `Sink<String>` queues text, `Stream<Item = Result<AudioSample>>` yields audio samples. 1:1 mapping: each text string produces one AudioSample.
- **`Inference::use_pocket_tts()`**: Takes paths to all 5 ONNX models + tokenizer.json + voice WAV file. Creates sessions via `self.onnx_session()`, constructs PocketTts.
- **State reset per utterance:** Each `start_synthesis` call restores state from the voice-conditioned snapshot (stored as raw bytes + shapes + types). This avoids needing `Clone` on `Value` and ensures each utterance starts from identical voice-conditioned KV-cache state.
- **Integration tests** (`pocket_tts_test.rs`):
  - `test_pocket_tts_is_send()` — Send trait check
  - `test_implements_sink_and_stream()` — trait check
  - `test_pocket_tts_integration()` — `#[ignore]`, loads models, generates "Hello world", verifies audio output has samples with non-zero variance, checks duration is reasonable (1-10s for "Hello world"), writes to `/tmp/pocket_tts_output.raw`
  - `test_pocket_tts_multiple_utterances()` — `#[ignore]`, generates 3 texts sequentially, verifies each produces audio

**Definition of Done:**

- [ ] `PocketTts` implements `Sink<String>` and `Stream<Item = Result<AudioSample>>`
- [ ] `PocketTts` is `Send`
- [ ] Text preparation capitalizes, adds punctuation, pads short texts
- [ ] Tokenization via `tokenizers::Tokenizer` produces correct token IDs
- [ ] `Inference::use_pocket_tts()` factory method exists and creates PocketTts
- [ ] Integration test generates audio from "Hello world" with >0 samples at 24kHz
- [ ] Integration test verifies audio samples contain non-zero variance (not silence)
- [ ] State resets between utterances (multiple generations work)
- [ ] Output file `/tmp/pocket_tts_output.raw` written with size >0 bytes for manual verification

**Verify:**

- `cargo test -p inference -- pocket -q` — unit tests pass (Send, Sink/Stream traits)
- `cargo test -p inference -- --ignored pocket_tts -q` — integration tests with real models pass
- `cargo check -p inference` — clean build
- `ffplay -f f32le -ar 24000 -ac 1 /tmp/pocket_tts_output.raw` — audible speech output

## Testing Strategy

- **Unit tests:** TensorElement bool support (onnx crate), text preparation logic, trait checks (Send, Sink, Stream)
- **Integration tests:** Full pipeline with real ONNX models (marked `#[ignore]`). Generate audio from text, verify non-empty output with non-zero variance, write raw audio for manual playback.
- **Manual verification:** Play generated audio with `ffplay -f f32le -ar 24000 -ac 1 /tmp/pocket_tts_output.raw` to verify intelligible speech.

## Runtime Environment

- **Integration test command:** `cargo test -p inference -- --ignored pocket_tts -q`
- **Prerequisites:** Voice audio file at `data/pocket/voice.wav` (any speaker, ~5-10s), all 5 ONNX models in `data/pocket/`
- **Manual verification:** `ffplay -f f32le -ar 24000 -ac 1 /tmp/pocket_tts_output.raw`
- **Expected output:** Audible speech at 24kHz sample rate, file size proportional to text length

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BOOL state tensors unsupported in onnx crate | High | High | Task 1 adds bool TensorElement support before any mimi_decoder work |
| SentencePiece tokenizer conversion fails | Low | High | Multiple conversion paths available (transformers library, manual protobuf parse). Fallback: ship .model file and add sentencepiece crate. |
| Empty tensors (shape [0]) cause ONNX Runtime errors | Med | Med | Test with `Value::from_slice::<f32>(&[0], &[])` during Task 1. If fails, use `Value::zeros` with shape [0]. |
| State tensor count/order mismatch between model versions | Low | High | Discover state names dynamically from model metadata (Sherpa pattern), never hardcode indices. |
| AR generation doesn't converge (no EOS) | Med | Med | Cap max_tokens at 1000 (configurable). Log warning if max reached without EOS. |
| Noise generation affects reproducibility | Low | Low | Use seeded RNG for tests. Production uses thread_rng for variety. |
| Voice conditioning requires audio file at construction | Med | Med | Document requirement clearly in Prerequisites. Voice file must be real speech (not silence — zero-filled latents produce garbage output). Constructor returns error if voice file missing or invalid. |

## Goal Verification

### Truths (what must be TRUE for the goal to be achieved)

- Text input produces audible speech audio at 24kHz via ONNX inference
- Voice cloning works: providing a reference audio changes the output voice
- The API follows the same Sink/Stream pattern as existing Kokoro TTS
- Multiple sequential text-to-speech generations work correctly
- The implementation uses the 5 ONNX models provided in `data/pocket/`

### Artifacts (what must EXIST to support those truths)

- `crates/onnx/src/value.rs` — bool TensorElement impl
- `crates/inference/src/tts/pocket/core.rs` — PocketCore with 5-session orchestration + AR loop
- `crates/inference/src/tts/pocket/pocket.rs` — PocketTts with Sink/Stream
- `crates/inference/src/tts/pocket/mod.rs` — module wiring
- `crates/inference/src/tts/pocket/tests/pocket_tts_test.rs` — integration tests
- `data/pocket/tokenizer.json` — converted SentencePiece tokenizer

### Key Links (critical connections that must be WIRED)

- `Inference::use_pocket_tts()` → creates sessions via `onnx_session()` → constructs PocketTts
- PocketTts `start_synthesis()` → tokenizer encode → PocketCore methods → AudioSample output
- PocketCore `generate_step()` → flow_main session.run() → LSD decode via flow_step session.run() → latent output
- PocketCore `decode_audio()` → mimi_decoder session.run() with state carry → audio frame
- PocketCore state tensors carry between successive session.run() calls (flow_main states, mimi_decoder states)

## Open Questions

- None — the ONNX export format is well-documented by the KevinAHM/pocket-tts-onnx-export project and all model I/O signatures have been inspected.

### Deferred Ideas

- Sentence splitting for very long texts (split on clause boundaries, generate chunks sequentially)
- Pre-computed voice state caching (save/load ONNX-format voice states to avoid re-encoding audio)
- Dynamic LSD step count tuning (quality vs latency tradeoff)
- Streaming output per-chunk during generation (currently collects all audio then yields)
