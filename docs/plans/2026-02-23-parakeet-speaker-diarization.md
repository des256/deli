# Parakeet Speaker Diarization Implementation Plan

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

**Goal:** Add streaming speaker diarization using NVIDIA's Sortformer v2 ONNX model, exposed via `Inference::use_parakeet_diar`, following the architecture from the `altunenes/parakeet-rs` reference implementation.

**Architecture:** A new `diar::parakeet` module containing a `Sortformer` struct that processes audio chunks through the ONNX model with streaming state management (speaker cache, FIFO buffer, silence profile). Post-processing converts raw per-frame speaker probabilities into `SpeakerSegment`s with start/end times and speaker IDs.

**Tech Stack:** ONNX Runtime (existing `onnx` crate), no new dependencies.

## Scope

### In Scope

- Mel feature extraction for diarization (time-first layout, no normalization)
- Streaming Sortformer inference with state management
- Smart speaker cache compression (NeMo algorithm)
- Post-processing: median filtering, hysteresis binarization
- `Inference::use_parakeet_diar` API
- Unit tests for features, cache compression, post-processing
- Integration test (gated on model file presence)

### Out of Scope

- Connecting diarization output to the multitalker ASR model's `spk_targets`
- Resampling audio (caller provides 16kHz mono)
- GPU-specific optimizations beyond what ONNX Runtime provides

## Prerequisites

- Model file: `data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx`
- Existing `onnx` crate with `Session`, `Value`, `from_slice`, `zeros`, `extract_tensor`, `tensor_shape`

## Context for Implementer

- **Patterns to follow:** Follow `crates/inference/src/vad/silero.rs` for ONNX session wrapping, state management, and `process()` method pattern. Follow `crates/inference/src/asr/parakeet/` for module organization (mod.rs, tests/).
- **Conventions:** Use `crate::error::{InferError, Result}` for errors. ONNX values via `onnx::Value::from_slice(shape, data)` and `onnx::Value::zeros::<T>(shape)`. Input shapes use `&[usize]` for `from_slice`, `&[i64]` for `zeros`.
- **Key files:**
  - `crates/inference/src/inference.rs` — add `use_parakeet_diar` method
  - `crates/inference/src/lib.rs` — add `pub mod diar;`
  - `crates/inference/src/vad/silero.rs` — pattern for ONNX inference wrapper
  - `crates/inference/src/asr/parakeet/features.rs` — existing mel feature extraction (reference, but diar needs different layout)
- **Gotchas:**
  - Diarization mel features use time-first layout `[B, T, 128]`, not channels-first `[128, T]` like ASR
  - Diarization does NOT apply per-feature normalization (unlike ASR)
  - `onnx::Value::from_slice` takes `&[usize]` shapes; `onnx::Value::zeros` takes `&[i64]` shapes
  - Dynamic tensor shapes: spkcache and fifo start empty (0 time dim) and grow
  - The model's `spkcache_fifo_chunk_preds` output concatenates predictions for [spkcache, fifo, chunk] — must slice by current lengths
- **Domain context:** Sortformer outputs per-frame probabilities for up to 4 speakers. The streaming protocol maintains a FIFO buffer for recent context and a speaker cache for long-range context. When the cache exceeds its limit, a smart compression algorithm retains the most informative frames per speaker.

**Reference implementation:** `github.com/altunenes/parakeet-rs/blob/master/src/sortformer.rs`

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Module structure, types, and mel feature extraction
- [x] Task 2: Core Sortformer streaming inference
- [x] Task 3: Smart cache compression
- [x] Task 4: Post-processing, public API, and integration

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Module structure, types, and mel feature extraction

**Objective:** Create the `diar::parakeet` module skeleton with public types (`SpeakerSegment`, `DiarizationConfig`) and mel feature extraction that produces time-first `[B, T, 128]` features from 16kHz mono audio.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/diar/mod.rs`
- Create: `crates/inference/src/diar/parakeet/mod.rs`
- Create: `crates/inference/src/diar/parakeet/features.rs`
- Create: `crates/inference/src/diar/parakeet/types.rs`
- Create: `crates/inference/src/diar/parakeet/tests/mod.rs`
- Create: `crates/inference/src/diar/parakeet/tests/features_test.rs`
- Modify: `crates/inference/src/lib.rs` — add `pub mod diar;`

**Key Decisions / Notes:**

- **Input format:** `compute_mel_features` takes `&[f32]` audio already normalized to [-1, 1] range (NOT i16 PCM). Unlike ASR's `compute_features` which takes `&[i16]` and divides by 32768.0, diarization skips the i16→f32 conversion since input is already f32. Pre-emphasis is applied directly to the f32 signal.
- Mel features for diarization differ from ASR: time-first layout `[B, T, 128]`, no per-feature normalization, uses `LOG_ZERO_GUARD = 5.96e-8` instead of `1e-10`
- Reuse the DFT-based power spectrum approach from `asr/parakeet/features.rs:126` but output in time-first flat layout
- Constants: N_FFT=512, WIN_LENGTH=400, HOP_LENGTH=160, N_MELS=128, PREEMPH=0.97, SAMPLE_RATE=16000
- `DiarizationConfig` has preset methods: `callhome()` (default), `dihard3()`, `custom(onset, offset)`
- `SpeakerSegment { start: f32, end: f32, speaker_id: usize }`
- `features.rs` provides `compute_mel_filterbank()` and `compute_mel_features()`. Sortformer calls `compute_mel_filterbank()` in `::new()` and stores the result for reuse in `compute_mel_features()`

**Definition of Done:**

- [ ] `compute_mel_features(&[f32], usize) -> Result<(Vec<f32>, usize)>` returns time-first features and frame count
- [ ] Features match expected dimensions: `features.len() == num_frames * 128`
- [ ] `DiarizationConfig::callhome()` returns onset=0.641, offset=0.561
- [ ] `SpeakerSegment` is public and accessible from `crate::diar::parakeet`
- [ ] Tests verify feature dimensions, mel filterbank properties, short audio rejection
- [ ] Sanity test: 440Hz sine wave features have higher energy in lower mel bins than upper bins (catches amplitude scaling bugs)

**Verify:**

- `cargo test -p inference -- diar::parakeet::tests::features_test -q`

### Task 2: Core Sortformer streaming inference

**Objective:** Implement the `Sortformer` struct with ONNX inference, streaming state management (spkcache, fifo, silence profile), and the `streaming_update` method.

**Dependencies:** Task 1

**Files:**

- Create: `crates/inference/src/diar/parakeet/sortformer.rs`
- Modify: `crates/inference/src/diar/parakeet/mod.rs` — add sortformer module
- Create: `crates/inference/src/diar/parakeet/tests/sortformer_test.rs`
- Modify: `crates/inference/src/diar/parakeet/tests/mod.rs` — add test module

**Key Decisions / Notes:**

- Sortformer struct holds: `session: Session`, `config: DiarizationConfig`, streaming constants (chunk_len, fifo_len, spkcache_len), streaming state (spkcache `Vec<f32>`, fifo `Vec<f32>`, fifo_preds `Vec<f32>`, spkcache_preds `Option<Vec<f32>>`, mean_sil_emb `Vec<f32>`, n_sil_frames)
- Use flat `Vec<f32>` with manual indexing (consistent with codebase — no ndarray dependency)
- `streaming_update(chunk_feat: &[f32], chunk_feat_frames: usize, actual_len: usize) -> Result<Vec<f32>>` runs ONNX model and updates state
- ONNX inputs: `chunk` [1, T_chunk, 128], `spkcache` [1, T_cache, 512], `fifo` [1, T_fifo, 512], lengths as i64
- ONNX outputs: extract predictions [1, T_out, 4] and embeddings [1, T_pre, 512]
- Slice output predictions by current lengths: spkcache portion, fifo portion, chunk portion
- Append chunk embeddings to fifo; when fifo exceeds fifo_len, pop excess into spkcache
- Update silence profile from popped frames (running mean of embeddings where total speaker prob < 0.2)
- Constants: SUBSAMPLING=8, EMB_DIM=512, NUM_SPEAKERS=4, FRAME_DURATION=0.08

**Definition of Done:**

- [ ] `Sortformer::new(session, config) -> Result<Self>` initializes with zero state
- [ ] `reset()` zeros all streaming state
- [ ] `streaming_update()` runs ONNX inference and returns chunk predictions
- [ ] FIFO grows and overflows into spkcache correctly
- [ ] Silence profile updates from low-activity frames
- [ ] Unit tests verify state dimensions after streaming_update calls
- [ ] First streaming_update call succeeds with zero-length spkcache/fifo (empty tensor [1, 0, DIM]). If ONNX rejects zero-length tensors, fall back to [1, 1, DIM] filled with zeros and adjust output slicing accordingly.

**Verify:**

- `cargo test -p inference -- diar::parakeet::tests::sortformer_test -q`
- `cargo check -p inference` (no errors)

### Task 3: Smart cache compression

**Objective:** Implement the NeMo-style smart speaker cache compression that retains the most informative frames per speaker when the cache exceeds its limit.

**Dependencies:** Task 2

**Files:**

- Modify: `crates/inference/src/diar/parakeet/sortformer.rs` — add compression methods
- Create: `crates/inference/src/diar/parakeet/tests/compression_test.rs`
- Modify: `crates/inference/src/diar/parakeet/tests/mod.rs` — add test module

**Key Decisions / Notes:**

- `compress_spkcache()`: called when `spkcache.len() > spkcache_len`
- Quality scoring: `get_log_pred_scores(preds) -> scores` — log-likelihood ratio per speaker per frame
- Score manipulation: `disable_low_scores()` sets non-positive scores to NEG_INFINITY; `boost_topk_scores()` boosts top-K frames by additive constant
- Top-K selection: `get_topk_indices()` selects spkcache_len/NUM_SPEAKERS frames per speaker
- Silence frames: reserve `SPKCACHE_SIL_FRAMES_PER_SPK=3` frames per speaker filled with mean silence embedding
- `gather_spkcache()` builds new cache from selected indices + silence frames
- Constants: PRED_SCORE_THRESHOLD=0.25, STRONG_BOOST_RATE=0.75, WEAK_BOOST_RATE=1.5, MIN_POS_SCORES_RATE=0.5, SIL_THRESHOLD=0.2

**Definition of Done:**

- [ ] Cache compression triggers when spkcache exceeds spkcache_len
- [ ] After compression, spkcache length equals spkcache_len
- [ ] Quality scores correctly compute log-likelihood ratios
- [ ] Silence frames are inserted using mean silence embedding
- [ ] Tests verify compression reduces cache to target size while preserving high-scoring frames
- [ ] Alignment test: synthetic cache with embedding[i]=[i,...] and pred[i]=[i,0,0,0] — after compression, retained embeddings and predictions remain paired correctly (catches off-by-one indexing)

**Verify:**

- `cargo test -p inference -- diar::parakeet::tests::compression_test -q`

### Task 4: Post-processing, public API, and integration

**Objective:** Implement median filtering and hysteresis binarization for segment generation, the public `diarize_chunk()` method, `Inference::use_parakeet_diar`, and integration tests.

**Dependencies:** Task 3

**Files:**

- Modify: `crates/inference/src/diar/parakeet/sortformer.rs` — add median_filter, binarize, diarize_chunk
- Modify: `crates/inference/src/diar/parakeet/mod.rs` — pub use exports
- Modify: `crates/inference/src/inference.rs` — add `use_parakeet_diar`
- Create: `crates/inference/src/diar/parakeet/tests/postprocess_test.rs`
- Create: `crates/inference/src/diar/parakeet/tests/integration_test.rs`
- Modify: `crates/inference/src/diar/parakeet/tests/mod.rs` — add test modules

**Key Decisions / Notes:**

- `diarize_chunk(audio_16k_mono: &[f32]) -> Result<Vec<SpeakerSegment>>`: extract features, process in CHUNK_LEN-frame sub-chunks, concatenate predictions, median filter, binarize
- **Chunk boundary handling:** If remaining frames < CHUNK_LEN, zero-pad features to CHUNK_LEN and pass `actual_len` (real frame count) to `streaming_update` so the model knows the true length. This matches the reference implementation's approach. Add a test with short audio (0.5s = 8000 samples) to verify no panic.
- `median_filter(preds, window)`: per-speaker sliding median over time axis
- `binarize(preds) -> Vec<SpeakerSegment>`: hysteresis thresholding with onset/offset, padding, min duration filtering, gap merging
- `Inference::use_parakeet_diar(model_path) -> Result<Sortformer>`: loads ONNX session, creates Sortformer with default config
- Integration test: `#[ignore]` test that loads the real model, processes silence/test audio, checks output is valid

**Definition of Done:**

- [ ] `median_filter` smooths predictions with configurable window size
- [ ] `binarize` produces SpeakerSegments with correct onset/offset thresholding
- [ ] `diarize_chunk` returns segments with start/end times in seconds
- [ ] `Inference::use_parakeet_diar` compiles and creates Sortformer instance
- [ ] `Sortformer` is `Send` (for use across async boundaries)
- [ ] Unit tests verify median filter output, binarization with known inputs
- [ ] Integration test: silence audio returns empty segments, test audio returns segments with valid `start < end` times in seconds and `speaker_id` in 0..3
- [ ] Multi-chunk integration test: call `diarize_chunk` 3 times with 1-second chunks, verify FIFO/cache state is non-empty after first call (state persists across calls)
- [ ] Short audio test: `diarize_chunk` with 0.5s audio (8000 samples) succeeds without panic (partial chunk handling)

**Verify:**

- `cargo test -p inference -- diar::parakeet -q` — all tests pass
- `cargo check -p inference -p wav-asr -p record-asr` — no compilation errors

## Testing Strategy

- **Unit tests:** Feature dimensions, mel filterbank shape, DiarizationConfig presets, cache compression size invariant, median filter correctness, binarization with synthetic probability curves
- **Integration tests:** `#[ignore]` tests that require the model file — load model, process silence (expect no segments), process synthetic audio (expect valid segment structure)
- **Manual verification:** Covered by the `#[ignore]` integration test — no separate CLI needed for this scope

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ONNX model expects different input format than reference | Low | High | Verify model inputs/outputs with Python (already done — shapes confirmed) |
| Performance: O(n²) DFT too slow for long audio | Low | Med | 512-point DFT is fast; real bottleneck is ONNX inference. Can upgrade to FFT later if needed |
| Cache compression numeric instability | Low | Med | Use same constants and algorithm as NeMo reference; test with edge cases (empty cache, single frame) |
| Dynamic tensor shapes cause ONNX errors with empty inputs | Med | Med | Handle zero-length spkcache/fifo by passing empty tensors with shape [1, 0, DIM] (confirmed this works from reference) |

## Goal Verification

### Truths (what must be TRUE for the goal to be achieved)

- Callers can create a Sortformer instance via `inference.use_parakeet_diar(model_path)`
- Audio chunks can be processed via `sortformer.diarize_chunk(audio)` returning speaker segments
- Each segment identifies which speaker (0-3) was active and for how long (start/end in seconds)
- Streaming state persists across `diarize_chunk` calls (no reset between calls)
- The model file `data/parakeet/diar_streaming_sortformer_4spk-v2.1.onnx` is loaded and runs without error

### Artifacts (what must EXIST to support those truths)

- `crates/inference/src/diar/parakeet/sortformer.rs` — Sortformer struct with streaming inference
- `crates/inference/src/diar/parakeet/features.rs` — mel feature extraction
- `crates/inference/src/diar/parakeet/types.rs` — SpeakerSegment, DiarizationConfig
- `crates/inference/src/inference.rs` — `use_parakeet_diar` method

### Key Links (critical connections that must be WIRED)

- `Inference::use_parakeet_diar` → creates `onnx::Session` → passes to `Sortformer::new`
- `Sortformer::diarize_chunk` → `compute_mel_features` → `streaming_update` → post-processing → `Vec<SpeakerSegment>`
- `lib.rs` exports `pub mod diar` → `diar/mod.rs` exports `parakeet` → `parakeet/mod.rs` exports `Sortformer`, `SpeakerSegment`, `DiarizationConfig`

## Open Questions

- None — the reference implementation and model inspection provide complete information.

### Deferred Ideas

- Connect diarization output to multitalker ASR `spk_targets` for speaker-aware transcription
- Add `diarize()` convenience method that resets state and processes full audio (non-streaming)
- Add experiment binary (e.g., `experiments/wav-diar`) for CLI testing
