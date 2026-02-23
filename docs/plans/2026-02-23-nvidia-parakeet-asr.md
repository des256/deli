# NVidia Parakeet ASR Implementation Plan

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
> **Worktree:** Set at plan creation (from dispatcher). `Yes` uses git worktree isolation; `No` works directly on current branch (default)

## Summary

**Goal:** Add NVidia Parakeet (FastConformer-Transducer) as a streaming ASR engine at `crates/inference/src/asr/parakeet/`, implementing the `Sink<AudioSample> + Stream<Item = Result<Transcription>>` interface identical to the existing Sherpa ASR, using pre-downloaded ONNX models from `data/parakeet/`.

**Architecture:** Two-model RNNT pipeline — a FastConformer encoder (`encoder-model.int8.onnx`, 652MB) produces 1024-dim embeddings with 8x temporal downsampling, and a combined decoder-joint model (`decoder_joint-model.int8.onnx`, 18MB) performs greedy RNNT decoding with stateful prediction network. The streaming interface accumulates audio, processes chunks through the encoder, and performs frame-by-frame greedy decoding while carrying decoder states across chunks.

**Tech Stack:** Rust, ONNX Runtime (via the `onnx` crate), `futures_core::Stream`, `futures_sink::Sink`, `tokio::task::spawn_blocking`

## Scope

### In Scope

- Parakeet module at `crates/inference/src/asr/parakeet/` with `Sink<AudioSample>` + `Stream<Item = Result<Transcription>>`
- 128-bin log-mel feature extraction (pre-emphasis 0.97, per-feature normalization)
- Vocab loading from `data/parakeet/vocab.txt` (SentencePiece format, 8193 tokens)
- RNNT greedy decoding with decoder state carryover across chunks
- Integration into `Inference` struct (`use_parakeet_asr` method)
- Re-export from `asr/mod.rs`
- Unit tests for features, vocab, and Send trait

### Out of Scope

- Beam search decoding (greedy only, matching sherpa)
- CUDA-specific optimizations (CPU works via existing `onnx_session`)
- Streaming encoder with cache states (model is offline encoder; streaming is via chunk processing)
- Language model integration
- New experiments (user can adapt existing `wav-streaming` experiment)

## Prerequisites

- ONNX Runtime library installed (already used by sherpa)
- Model files in `data/parakeet/`: `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `vocab.txt`

## Context for Implementer

> This section is critical for cross-session continuity.

- **Patterns to follow:** Follow the exact module structure of `crates/inference/src/asr/sherpa/` — the Parakeet module mirrors it file-for-file: `mod.rs`, `features.rs`, `tokens.rs`, `asrcore.rs`, `parakeet.rs`, `tests/`
- **Conventions:** All ASR engines export a single public struct (e.g., `Sherpa`, `Whisper`) from their module. They implement `Sink<AudioSample>` + `Stream<Item = Result<Transcription>>`. Error type is `InferError` from `crate::error`.
- **Key files:**
  - `crates/inference/src/asr/sherpa/sherpa.rs:35-293` — Reference `Sink`/`Stream` implementation
  - `crates/inference/src/asr/sherpa/asrcore.rs:1-268` — Reference core decoding logic
  - `crates/inference/src/asr/sherpa/features.rs:1-203` — Reference mel feature extraction (80 bins; Parakeet needs 128)
  - `crates/inference/src/asr/sherpa/tokens.rs:1-67` — Token loading (reusable as-is for vocab.txt)
  - `crates/inference/src/asr/mod.rs` — Module registry + `Transcription` enum
  - `crates/inference/src/inference.rs:149-165` — `use_streaming_asr` method pattern
- **Gotchas:**
  - Parakeet encoder input is `[B, 128, T]` (channels-first, 128 mel bins), NOT `[B, T, 80]` like sherpa's feature layout
  - Decoder-joint model outputs 8198 logits but only 8193 are valid (indices 0-8192). Must mask indices > 8192 during argmax
  - Blank token is at index **8192** (last), not 0 (first) like in sherpa
  - Decoder-joint uses `int32` for `targets` and `target_length`, not `int64` like sherpa's decoder
  - Decoder-joint has 2 hidden state tensors (`input_states_1/2` shape `[2, B, 640]`) that must carry across calls for decoder continuity
  - Encoder has no internal state (offline FastConformer) — each chunk is encoded independently; only decoder states carry across chunks
  - Feature computation differs: 128 mel bins, pre-emphasis 0.97, per-feature normalization (subtract mean, divide by std across time axis), no Slaney normalization
- **Domain context:** RNNT (RNN-Transducer) decoding iterates over encoder output frames. For each frame, the decoder-joint is called in a loop: predict token → if not blank, emit token and update state, repeat; if blank, move to next frame. `max_symbols_per_step` limits the inner loop to prevent infinite emission.

## Model I/O Reference

### Encoder (`encoder-model.int8.onnx`)

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `audio_signal` | `[B, 128, T]` | float32 |
| Input | `length` | `[B]` | int64 |
| Output | `outputs` | `[B, 1024, T/8]` | float32 |
| Output | `encoded_lengths` | `[B]` | int64 |

### Decoder-Joint (`decoder_joint-model.int8.onnx`)

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `encoder_outputs` | `[B, 1024, 1]` | float32 |
| Input | `targets` | `[B, 1]` | int32 |
| Input | `target_length` | `[B]` | int32 |
| Input | `input_states_1` | `[2, B, 640]` | float32 |
| Input | `input_states_2` | `[2, B, 640]` | float32 |
| Output | `outputs` | `[B, 1, 1, 8198]` | float32 |
| Output | `prednet_lengths` | `[B]` | int32 |
| Output | `output_states_1` | `[2, B, 640]` | float32 |
| Output | `output_states_2` | `[2, B, 640]` | float32 |

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Vocab loading module
- [x] Task 2: 128-bin mel feature extraction
- [x] Task 3: Core RNNT decoding (encoder + decoder-joint + greedy search)
- [x] Task 4: Streaming Sink/Stream implementation
- [x] Task 5: Integration into Inference and module registry

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Vocab Loading Module

**Objective:** Create `tokens.rs` that loads `vocab.txt` into a `Vec<String>` indexed by token ID. The format is identical to sherpa's `tokens.txt` (`<token> <id>` per line), so the logic can be reused directly.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/asr/parakeet/tokens.rs`
- Test: `crates/inference/src/asr/parakeet/tests/parakeet_test.rs`

**Key Decisions / Notes:**

- Copy sherpa's `tokens.rs` logic (it's a small, self-contained function)
- Vocab has 8193 entries (indices 0-8192), blank token `<blk>` at index 8192
- Same `load_tokens<P: AsRef<Path>>` signature returning `Result<Vec<String>>`

**Definition of Done:**

- [ ] `load_tokens` successfully parses `data/parakeet/vocab.txt` and returns 8193 tokens
- [ ] Token at index 8192 is `<blk>`
- [ ] Error handling for missing/malformed files
- [ ] Unit tests for valid file, missing file, and malformed file

**Verify:**

- `cargo test -p inference parakeet -- --nocapture` — tests pass

### Task 2: 128-bin Mel Feature Extraction

**Objective:** Create `features.rs` that computes 128-dimensional log-mel filterbank features from 16kHz PCM i16 audio, matching the NeMo FastConformer preprocessing pipeline.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/asr/parakeet/features.rs`
- Modify: `crates/inference/src/asr/parakeet/tests/parakeet_test.rs`

**Key Decisions / Notes:**

- Based on sherpa's `features.rs` but with these differences:
  - 128 mel bins (not 80)
  - Per-feature normalization: for each mel bin, subtract mean and divide by std across the time axis (std clamped to minimum 1e-5)
  - Returns features in **channels-first** layout: flattened `Vec<f32>` of shape `[128, num_frames]` (transposed from the `[num_frames, 128]` computation order)
- Same window/hop as sherpa: 25ms window (400 samples), 10ms hop (160 samples), 512-point FFT
- Pre-emphasis coefficient: 0.97
- HTK mel scale, triangular filters, NO Slaney normalization
- Log guard: `ln(energy + 1e-10)`

**Definition of Done:**

- [ ] `compute_features(&[i16], usize) -> Result<(Vec<f32>, usize)>` returns `(features, num_frames)` where features is `[128 * num_frames]` in channels-first order
- [ ] Rejects non-16kHz sample rates
- [ ] Rejects audio shorter than one window (400 samples)
- [ ] Per-feature normalization applied (mean≈0, std≈1 per bin)
- [ ] Unit tests: dimensions check, invalid sample rate, short audio

**Verify:**

- `cargo test -p inference parakeet -- --nocapture` — feature tests pass

### Task 3: Core RNNT Decoding

**Objective:** Create `asrcore.rs` containing `AsrCore` struct that holds ONNX sessions and performs encoder inference + frame-by-frame greedy RNNT decoding with the decoder-joint model.

**Dependencies:** Task 1, Task 2

**Files:**

- Create: `crates/inference/src/asr/parakeet/asrcore.rs`

**Key Decisions / Notes:**

- `AsrCore` struct holds: `encoder: Session`, `decoder_joint: Session`, `tokens: Vec<String>`, `blank_id: i64`, decoder states (`state1`, `state2` as `Vec<onnx::Value>`), and `last_token: i32`
- `decode_chunk(features: &[f32], num_frames: usize) -> Result<String>`:
  1. Build encoder input: `audio_signal` as `[1, 128, num_frames]` float32, `length` as `[1]` int64
  2. Run encoder → get `outputs` `[1, 1024, T']` and `encoded_lengths`
  3. For each encoder frame `t` in `0..enc_len`:
     - Extract frame `[1, 1024, 1]`
     - Inner loop (max 10 iterations):
       - Run decoder-joint with frame, `last_token`, states
       - Get logits `[8198]`, take `argmax` over first 8193 elements only
       - If predicted == blank_id (8192): break inner loop
       - Else: append token, update `last_token` and states
  4. Convert token IDs to text via vocab lookup
- Decoder states persist across `decode_chunk` calls (critical for streaming)
- `reset()` method to zero out states for starting a new utterance
- `tokens_to_text`: join tokens, replace `▁` with space, trim leading space

**Definition of Done:**

- [ ] `AsrCore::new(encoder, decoder_joint, tokens)` initializes with zero states
- [ ] `decode_chunk` runs encoder + greedy decode and returns text
- [ ] Decoder states carry across consecutive `decode_chunk` calls
- [ ] `argmax` restricted to valid vocab range (0-8192)
- [ ] `tokens_to_text` correctly handles SentencePiece `▁` tokens

**Verify:**

- `cargo test -p inference parakeet -- --nocapture` — core tests pass
- `cargo build -p inference` — no compiler errors

### Task 4: Streaming Sink/Stream Implementation

**Objective:** Create `parakeet.rs` containing the `Parakeet` struct that implements `Sink<AudioSample>` for accepting audio and `Stream<Item = Result<Transcription>>` for producing transcriptions, following the exact same pattern as `sherpa.rs`.

**Dependencies:** Task 3

**Files:**

- Create: `crates/inference/src/asr/parakeet/parakeet.rs`
- Modify: `crates/inference/src/asr/parakeet/tests/parakeet_test.rs`

**Key Decisions / Notes:**

- Mirror `sherpa.rs` structure exactly:
  - `core: Arc<Mutex<AsrCore>>`
  - `sample_buffer: Vec<i16>`, `decoded_text: String`
  - `chunk_samples: usize` — configurable, default to ~32000 samples (2 seconds at 16kHz, producing ~200 mel frames → 25 encoder output frames per chunk — good balance of latency and accuracy)
  - `pending_chunks: VecDeque<Vec<i16>>`, `closed: bool`
  - `inflight: Option<Pin<Box<dyn Future<...>>>>`
  - `stream_waker: Option<Waker>`
- `Sink::start_send`: validate 16kHz, extract PCM, buffer, split into chunks
- `Sink::poll_close`: zero-pad remaining samples, enqueue final chunk
- `Stream::poll_next`: poll inflight decode, dequeue next chunk, start decode via `spawn_blocking`
- `start_decode`: spawns `tokio::task::spawn_blocking` that calls `AsrCore::decode_chunk` with mel features computed inside the blocking task
- `Parakeet::new(encoder: Session, decoder_joint: Session, vocab_path: P) -> Result<Self>`
- `with_chunk_samples(self, samples: usize) -> Self` for configuring chunk size

**Definition of Done:**

- [ ] `Parakeet` implements `Sink<AudioSample>` with 16kHz validation
- [ ] `Parakeet` implements `Stream<Item = Result<Transcription>>`
- [ ] Partial transcriptions emitted for each processed chunk
- [ ] Final transcription emitted when stream is closed
- [ ] `Parakeet` is `Send` (verified by compile-time test)
- [ ] Chunk-based processing with configurable chunk size

**Verify:**

- `cargo test -p inference parakeet -- --nocapture` — all tests pass including Send check
- `cargo build -p inference` — no compiler errors

### Task 5: Integration into Inference and Module Registry

**Objective:** Wire the Parakeet module into the module tree (`asr/mod.rs`) and add a `use_parakeet_asr` constructor method on `Inference`.

**Dependencies:** Task 4

**Files:**

- Create: `crates/inference/src/asr/parakeet/mod.rs`
- Modify: `crates/inference/src/asr/mod.rs` — add `pub mod parakeet; pub use parakeet::Parakeet;`
- Modify: `crates/inference/src/inference.rs` — add `use_parakeet_asr` method

**Key Decisions / Notes:**

- `parakeet/mod.rs`: expose `Parakeet` publicly, keep `asrcore`, `features`, `tokens` as `pub(crate)`
- `Inference::use_parakeet_asr(encoder_path, decoder_joint_path, vocab_path) -> Result<Parakeet>`:
  - Loads encoder and decoder-joint ONNX sessions via `self.onnx_session()`
  - Calls `Parakeet::new(encoder, decoder_joint, vocab_path)`
- Follow the pattern of `use_streaming_asr` at `inference.rs:149`

**Definition of Done:**

- [ ] `use inference::asr::Parakeet` compiles
- [ ] `Inference::use_parakeet_asr()` loads models and returns working `Parakeet`
- [ ] Full pipeline test: load models from `data/parakeet/`, feed `test.wav`, get non-empty transcription
- [ ] `cargo build -p inference` succeeds with no warnings

**Verify:**

- `cargo test -p inference parakeet -- --nocapture` — all tests pass
- `cargo build -p inference` — clean build

## Testing Strategy

- **Unit tests:** Token loading (valid/invalid/missing files), feature dimensions and normalization, Send trait assertion
- **Integration tests:** Full decode pipeline with real models from `data/parakeet/` and `test.wav` — verify non-empty, sensible transcription output
- **Manual verification:** Run existing `wav-streaming` experiment pointed at `data/parakeet` to confirm streaming behavior

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Feature extraction mismatch with NeMo | Med | High | Verified empirically with Python prototype — "preemph+no-slaney+per-feature-norm" produces readable text. Integration test with real audio confirms quality. |
| Encoder performance on large audio (652MB model) | Low | Med | Use `spawn_blocking` to avoid blocking the async runtime, same pattern as sherpa |
| Decoder state corruption at chunk boundaries | Low | Med | Decoder states are carried across chunks by persisting them in `AsrCore`. Reset on new utterance. |
| Out-of-vocab logit indices (8193-8197) selected during decode | Low | High | Argmax restricted to valid range `[0, 8193)` — verified in Python prototype |

## Goal Verification

> Derived from the plan's goal using goal-backward methodology.

### Truths (what must be TRUE for the goal to be achieved)

- A `Parakeet` struct exists that implements both `Sink<AudioSample>` and `Stream<Item = Result<Transcription>>`
- Feeding 16kHz PCM audio to the Parakeet sink and reading the stream produces English text transcription
- The Parakeet module is accessible via `inference::asr::Parakeet` and constructable via `Inference::use_parakeet_asr()`
- The implementation uses the ONNX models from `data/parakeet/` (encoder + decoder-joint + vocab)

### Artifacts (what must EXIST to support those truths)

- `crates/inference/src/asr/parakeet/parakeet.rs` — Sink/Stream struct with async chunked processing
- `crates/inference/src/asr/parakeet/asrcore.rs` — RNNT encoder + decoder-joint + greedy decode
- `crates/inference/src/asr/parakeet/features.rs` — 128-bin mel feature extraction
- `crates/inference/src/asr/parakeet/tokens.rs` — vocab loading
- `crates/inference/src/asr/parakeet/mod.rs` — module definition
- `crates/inference/src/asr/parakeet/tests/parakeet_test.rs` — unit + integration tests

### Key Links (critical connections that must be WIRED)

- `asr/mod.rs` exports `pub mod parakeet` and `pub use parakeet::Parakeet`
- `inference.rs` has `use_parakeet_asr()` that calls `self.onnx_session()` for both models then `Parakeet::new()`
- `Parakeet::start_decode()` calls `AsrCore::decode_chunk()` inside `spawn_blocking`
- `AsrCore::decode_chunk()` calls `features::compute_features()` then `run_encoder()` then `greedy_decode()`

## Open Questions

- None — architecture verified empirically with Python prototype

### Deferred Ideas

- Beam search decoding for better accuracy
- Streaming encoder with cache states (would require re-exporting the model)
- Experiment binary (`experiments/wav-streaming-parakeet`) for easy CLI testing
