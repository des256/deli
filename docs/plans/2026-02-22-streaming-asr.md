# Streaming ASR Implementation Plan

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

**Goal:** Implement streaming speech recognition using a Zipformer-Transducer ONNX model in the `inference` crate, producing text tokens frame-by-frame from audio input via the existing Sink/Stream API pattern.

**Architecture:** A new `StreamingAsr` struct that loads 3 ONNX sessions (encoder, decoder, joiner) and a tokens file. It implements `Sink<AudioSample>` + `Stream<Item = Result<Transcription>>` following the same pattern as `Whisper` and `Kokoro`. Internally it accumulates audio samples, computes log-mel features, runs the encoder chunk-by-chunk while maintaining hidden states, performs greedy transducer decoding (encoder → decoder → joiner → argmax), and emits `Transcription::Partial` as tokens arrive and `Transcription::Final` on stream close.

**Tech Stack:** Rust, the existing `onnx` crate (custom FFI wrapper around ONNX Runtime), `audio` crate for `AudioSample`/`AudioData`, 80-dimensional log-mel filterbank features computed in pure Rust.

## Scope

### In Scope

- `StreamingAsr` struct with Sink/Stream implementation
- Loading encoder, decoder, joiner ONNX sessions and tokens file
- Log-mel filterbank feature extraction (80-dim, 16kHz, 25ms frame, 10ms hop)
- Greedy transducer decode loop (max-sym-per-frame=1)
- Encoder state management across chunks
- `Inference::use_streaming_asr()` factory method
- Unit tests for feature extraction, token loading, decode logic
- Integration into `lib.rs` exports

### Out of Scope

- Beam search decoding (greedy only)
- Endpoint detection / utterance segmentation
- Voice activity detection (VAD)
- Model downloading or conversion (user provides ONNX files)
- Punctuation/capitalization (model outputs lowercase without punctuation)
- GPU acceleration specific to this model (uses existing ONNX device config)

## Prerequisites

- ONNX Runtime installed (already required for Kokoro TTS)
- Sherpa ONNX streaming zipformer model files: `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt` (user provides these)

## Context for Implementer

- **Patterns to follow:** The `Kokoro` TTS implementation at `crates/inference/src/tts/kokoro.rs` is the closest pattern — it wraps an ONNX `Session` in `Arc<Mutex<>>`, implements `Sink<Input>` + `Stream<Item = Result<Output>>`, and spawns blocking inference via `tokio::task::spawn_blocking`. Follow this pattern exactly.
- **Conventions:** Models are created via `Inference::use_*()` factory methods (see `crates/inference/src/inference.rs:63-135`). The factory calls `self.onnx_session()` to create sessions. Public types are re-exported from `lib.rs`.
- **Key files:**
  - `crates/onnx/src/session.rs` — `Session::run()` takes `&[(&str, &Value)]` inputs and `&[&str]` output names, returns `Vec<Value>`
  - `crates/onnx/src/value.rs` — `Value::from_slice::<T>(shape, data)` creates input tensors, `value.extract_tensor::<T>()` reads outputs
  - `crates/inference/src/asr/transcription.rs` — `Transcription::Partial` and `Transcription::Final` variants
  - `crates/inference/src/asr/whisper.rs` — Existing ASR Sink/Stream impl for reference
  - `crates/audio/src/lib.rs` — `AudioSample { data: AudioData::Pcm(Tensor<i16>), sample_rate: usize }`
- **Gotchas:**
  - `Session::run()` takes `&mut self` — the session must be behind `Arc<Mutex<>>` (same as Kokoro)
  - `Value` is not `Clone` — output data must be extracted within the lock scope
  - The streaming encoder has many state tensors (variable count depending on model config) — tensor names are dynamic, discovered at runtime from the ONNX model's input/output metadata. The implementation must introspect the model to find state tensor names.
  - The decoder `context_size` is typically 2 (the decoder needs the last 2 token IDs as input)
  - Token ID 0 is `<blk>` (blank), used as the transducer blank symbol
  - The `▁` character in tokens represents word boundaries (SentencePiece convention)

## ONNX Model I/O Specification

### Encoder (streaming)

**Inputs:**
- `x`: `f32 [batch=1, T, 80]` — log-mel features for one chunk
- `x_lens`: `i64 [batch=1]` — number of frames in this chunk
- `state_0` .. `state_N`: `f32 [various shapes]` — cached hidden states (initialized to zeros on first call)

**Outputs:**
- `encoder_out`: `f32 [batch=1, T', encoder_dim]` — encoded features (T' is subsampled)
- `encoder_out_lens`: `i64 [batch=1]` — output length
- `new_state_0` .. `new_state_N`: `f32 [various shapes]` — updated hidden states (fed back as input on next call)

### Decoder (prediction network)

**Inputs:**
- `y`: `i64 [batch=1, context_size]` — last `context_size` token IDs (typically 2)

**Outputs:**
- `decoder_out`: `f32 [batch=1, 1, decoder_dim]` — prediction embedding

### Joiner (joint network)

**Inputs:**
- `encoder_out`: `f32 [batch=1, 1, encoder_dim]` — single encoder frame
- `decoder_out`: `f32 [batch=1, 1, decoder_dim]` — decoder output

**Outputs:**
- `logit`: `f32 [batch=1, 1, vocab_size]` — log-probabilities over vocabulary

### Greedy Decode Algorithm (per encoder output frame)

```
for each frame t in encoder_out:
    loop:
        joiner_out = joiner(encoder_out[t], decoder(context))
        token = argmax(joiner_out)
        if token == blank_id: break  # move to next frame
        emit(token)
        update context with token
```

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Token loading and mel feature extraction
- [x] Task 2: StreamingAsr struct and ONNX session loading
- [x] Task 3: Transducer greedy decode loop
- [x] Task 4: Sink/Stream implementation and Inference integration

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Token loading and mel feature extraction

**Objective:** Implement two foundation pieces: (1) loading the `tokens.txt` file into a `Vec<String>` mapping token_id → text, and (2) computing 80-dimensional log-mel filterbank features from 16kHz PCM audio, matching the Kaldi-compatible feature extraction used by the Zipformer model.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/asr/streaming/tokens.rs`
- Create: `crates/inference/src/asr/streaming/features.rs`
- Create: `crates/inference/src/asr/streaming/mod.rs`
- Test: `crates/inference/tests/streaming_asr_test.rs`

**Key Decisions / Notes:**

- Token file format: each line is `<token_text> <token_id>`, IDs are 0-based contiguous. Parse into `Vec<String>` indexed by ID.
- Special tokens: `<blk>` (id=0) is the blank token for transducer decoding.
- Mel features: 80 bins, 25ms window (400 samples at 16kHz), 10ms hop (160 samples), pre-emphasis 0.97, Hann window, power spectrum → mel filterbank → log. This matches the Kaldi `fbank` configuration used by all sherpa-onnx models.
- The `▁` prefix on tokens indicates a word boundary — when decoding, replace `▁` with a space to produce readable text.
- Feature extraction is pure Rust — no external dependencies. Use a simple DFT (the window is only 400 samples / 512-point FFT, performance is not a concern).

**Definition of Done:**

- [ ] `load_tokens("tokens.txt")` returns `Vec<String>` with correct mapping (blank at index 0)
- [ ] `load_tokens` returns error for malformed or missing files
- [ ] `compute_features(pcm_i16, sample_rate)` returns `Vec<f32>` of shape `[num_frames, 80]` (flattened)
- [ ] Features computed from a known audio signal match expected dimensions (num_frames = (num_samples - 400) / 160 + 1)
- [ ] All tests pass

**Verify:**

- `cargo test -p inference --test streaming_asr_test -- tokens` — token loading tests pass
- `cargo test -p inference --test streaming_asr_test -- features` — feature extraction tests pass

---

### Task 2: StreamingAsr struct and ONNX session loading

**Objective:** Create the `StreamingAsr` struct that holds three ONNX sessions (encoder, decoder, joiner), the token vocabulary, and mutable decoding state. Implement construction, encoder state initialization (by introspecting ONNX model inputs/outputs to discover state tensor names and shapes), and the `Inference::use_streaming_asr()` factory method.

**Dependencies:** Task 1

**Files:**

- Create: `crates/inference/src/asr/streaming/streaming_asr.rs`
- Modify: `crates/inference/src/asr/streaming/mod.rs`
- Modify: `crates/inference/src/asr/mod.rs`
- Modify: `crates/inference/src/inference.rs`
- Modify: `crates/inference/src/lib.rs`
- Test: `crates/inference/tests/streaming_asr_test.rs`

**Key Decisions / Notes:**

- The `onnx` crate's `Session` does not currently expose input/output metadata (names, shapes). We need to add methods to query this from the ONNX Runtime C API — `SessionGetInputCount`, `SessionGetInputName`, `SessionGetInputTypeInfo` etc. This is a necessary extension to the `onnx` crate.
- State tensors: the encoder has N state inputs (`state_0`, `state_1`, ...) and N corresponding outputs (`new_state_0`, `new_state_1`, ...). On first call, all states are initialized to zero tensors of the correct shape. On subsequent calls, outputs from the previous call become inputs.
- Structure: `StreamingAsr` holds `Arc<Mutex<Session>>` for each of the 3 sessions (same pattern as Kokoro), plus `tokens: Vec<String>`, `context: Vec<i64>` (last `context_size` token IDs, initialized to blank), and `encoder_states: Vec<Value>` (the cached states).
- The `onnx` crate needs a `Session::input_count()`, `Session::input_name(idx)`, `Session::output_count()`, `Session::output_name(idx)`, and `Session::input_shape(idx)` set of methods added.

**Definition of Done:**

- [ ] `onnx::Session` exposes `input_count()`, `input_name(idx)`, `output_count()`, `output_name(idx)`, and `input_shape(idx)` methods
- [ ] `StreamingAsr::new()` loads 3 ONNX sessions and tokens file without error (when given valid files)
- [ ] `StreamingAsr::new()` returns descriptive error for missing/invalid files
- [ ] Encoder initial states are correctly sized zero tensors matching model metadata
- [ ] `Inference::use_streaming_asr()` creates a `StreamingAsr` successfully
- [ ] `StreamingAsr` is `Send`
- [ ] All tests pass

**Verify:**

- `cargo test -p onnx` — onnx crate tests pass (including new metadata methods)
- `cargo test -p inference --test streaming_asr_test -- streaming_asr` — struct creation tests pass

---

### Task 3: Transducer greedy decode loop

**Objective:** Implement the core transducer greedy decode method that takes a chunk of log-mel features, runs the encoder (with state carry), and performs frame-by-frame greedy decoding through the decoder and joiner. Returns newly decoded tokens as text.

**Dependencies:** Task 2

**Files:**

- Modify: `crates/inference/src/asr/streaming/streaming_asr.rs`
- Test: `crates/inference/tests/streaming_asr_test.rs`

**Key Decisions / Notes:**

- The decode method signature: `fn decode_chunk(&mut self, features: &[f32], num_frames: usize) -> Result<String>` — takes flattened mel features and frame count, returns decoded text for this chunk.
- Greedy search with `max_sym_per_frame = 1`: for each encoder output frame, run joiner at most once. If argmax is not blank, emit the token and update context. If blank, move to next frame.
- Token-to-text conversion: look up token ID in `self.tokens`, replace leading `▁` with space, concatenate. Trim leading space from the result.
- Encoder state management: after each `session.run()`, the output states replace `self.encoder_states` for the next chunk.
- All three sessions are called within `spawn_blocking` (they're CPU-bound ONNX inference).
- The decoder's `context_size` is determined from the decoder model's input shape (typically 2). Initialize context to `[blank_id; context_size]`.

**Definition of Done:**

- [ ] `decode_chunk()` runs encoder with correct state carry across consecutive calls
- [ ] Greedy decoding: given joiner outputs where argmax produces `[5, 0, 12, 0, 0]`, decoder emits tokens `[5, 12]` (blanks skipped)
- [ ] Token text for IDs where `tokens[N]="▁hello"` and `tokens[M]="world"` produces `"hello world"` (blank skipped, `▁` → space, leading space trimmed)
- [ ] Multi-chunk test: two consecutive `decode_chunk` calls maintain state (encoder states from chunk 1 are fed as inputs to chunk 2)
- [ ] All tests pass

**Verify:**

- `cargo test -p inference --test streaming_asr_test -- decode` — decode tests pass
- `cargo build -p inference` — no compilation errors

---

### Task 4: Sink/Stream implementation and Inference integration

**Objective:** Implement `Sink<AudioSample>` and `Stream<Item = Result<Transcription>>` for `StreamingAsr`, following the exact pattern used by `Whisper` and `Kokoro`. Audio samples are buffered, features are computed when enough samples accumulate for one chunk, `decode_chunk` is called, and partial/final transcriptions are yielded.

**Dependencies:** Task 3

**Files:**

- Modify: `crates/inference/src/asr/streaming/streaming_asr.rs`
- Modify: `crates/inference/src/asr/mod.rs`
- Modify: `crates/inference/src/lib.rs`
- Test: `crates/inference/tests/streaming_asr_test.rs`

**Key Decisions / Notes:**

- Chunk size: configurable via `with_chunk_samples(n)`, default to 5120 samples (320ms at 16kHz, which produces 16 mel frames — matching the model's default chunk configuration).
- `Sink::start_send()` appends PCM samples to an internal buffer and wakes the stream. Validates sample rate is 16kHz.
- `Stream::poll_next()`: when buffer has enough samples for a chunk, extract them, compute features, call `decode_chunk` via `spawn_blocking`, yield `Transcription::Partial { text, confidence: 1.0 }` with cumulative text. On stream close, flush remaining buffer and yield `Transcription::Final`.
- The cumulative text approach: `StreamingAsr` maintains a running `decoded_text: String` that grows as chunks are processed. Partial results contain the full text decoded so far. Final result contains the complete text.
- Follow the `inflight` future pattern from `whisper.rs:106-141` and `kokoro.rs:107-177`.
- Re-export `StreamingAsr` from `crates/inference/src/asr/mod.rs` and `lib.rs`.

**Definition of Done:**

- [ ] Send 5120 samples (1 chunk) → Stream yields `Transcription::Partial` with text matching `decode_chunk` output
- [ ] Send 10240 samples (2 chunks) → Stream yields 2 `Partial` results, second contains cumulative text from both chunks
- [ ] Close sink with remaining samples → Stream yields `Transcription::Final` with complete cumulative text
- [ ] Send audio at 8kHz → Sink returns error `"requires 16000 Hz audio, got 8000 Hz"`
- [ ] `with_chunk_samples()` configures the chunk size
- [ ] `StreamingAsr` is re-exported from `inference` crate's public API
- [ ] `Inference::use_streaming_asr()` is documented and accessible
- [ ] All tests pass
- [ ] `cargo build -p inference` succeeds with no warnings

**Verify:**

- `cargo test -p inference --test streaming_asr_test` — all streaming ASR tests pass
- `cargo build -p inference` — clean build
- `cargo doc -p inference --no-deps` — documentation builds

## Testing Strategy

- **Unit tests:** Token loading (valid, invalid, empty files), mel feature extraction (known signal → expected dimensions, edge cases), greedy decode logic (mock sessions or known model outputs)
- **Integration tests:** Full pipeline test with actual ONNX model files (when available) — send audio chunks, verify text output
- **Manual verification:** Run with the downloaded sherpa-onnx-streaming-zipformer-en-20M model and a known WAV file, compare output text

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ONNX model state tensor names/shapes vary between model versions | Med | High | Introspect model metadata at runtime rather than hardcoding names; add `Session` metadata API to `onnx` crate |
| Mel feature extraction doesn't match model's training features | Med | High | Follow Kaldi fbank specification exactly (80 bins, 25ms/10ms, pre-emphasis 0.97, Hann window); validate with known test vectors from sherpa-onnx |
| The `onnx` crate doesn't expose session metadata APIs needed for state introspection | High | High | Extend the `onnx` crate's FFI bindings to include `SessionGetInputCount`, `SessionGetInputName`, `SessionGetInputTypeInfo` etc. — these are standard ONNX Runtime C API functions |
| Context size assumption (2) doesn't match some models | Low | Med | Read context_size from decoder model's input shape rather than hardcoding |

## Goal Verification

### Truths (what must be TRUE for the goal to be achieved)

- Audio chunks sent to StreamingAsr produce text tokens incrementally
- The transducer decode loop correctly skips blank tokens and emits non-blank tokens
- Encoder state is preserved across chunks, enabling continuous recognition
- The public API follows the same Sink/Stream pattern as existing Whisper and Kokoro models
- Token text is correctly decoded with word boundaries producing spaces

### Artifacts (what must EXIST to support those truths)

- `crates/inference/src/asr/streaming/mod.rs` — module declaration
- `crates/inference/src/asr/streaming/tokens.rs` — token file loading
- `crates/inference/src/asr/streaming/features.rs` — mel feature extraction
- `crates/inference/src/asr/streaming/streaming_asr.rs` — StreamingAsr struct, decode loop, Sink/Stream impl
- `crates/inference/tests/streaming_asr_test.rs` — tests

### Key Links (critical connections that must be WIRED)

- `Inference::use_streaming_asr()` → creates `StreamingAsr` via `self.onnx_session()` for each of the 3 models
- `StreamingAsr::Sink::start_send()` → buffers PCM → triggers `decode_chunk()` when enough samples
- `decode_chunk()` → `encoder.run()` → `decoder.run()` → `joiner.run()` → token lookup → text
- `StreamingAsr` re-exported from `inference::asr` and `inference` crate root

## Open Questions

- None — the model I/O spec is well-documented from sherpa-onnx/icefall source code.

### Deferred Ideas

- Beam search decoding for better accuracy
- Endpoint detection (silence → utterance boundary)
- Integration with VAD (Silero or energy-based) for automatic speech segmentation
- Streaming the chat experiment with voice input instead of text
