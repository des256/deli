# Silero VAD ONNX Implementation Plan

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

**Goal:** Implement Silero Voice Activity Detection (VAD) using ONNX inference, placed at `crates/inference/src/vad/`, following existing codebase patterns for ONNX model orchestration.

**Architecture:** A `SileroVad` struct wraps an ONNX `Session` plus internal state tensors (state `[2,1,128]` and context buffer for v5). A `process()` method accepts a single audio frame (`&[f32]`, 512 samples at 16kHz) and returns a speech probability `f32`. The `Inference` struct gains a `use_silero_vad()` factory method. The design mirrors the Sherpa ASR pattern (core struct with ONNX sessions, state carry between calls, factory method on `Inference`).

**Tech Stack:** Rust, ONNX Runtime via the project's `onnx` crate, `base` crate for logging

## Scope

### In Scope

- `SileroVad` struct with ONNX session, state management, and frame-level inference
- Support for Silero VAD v5 model (state `[2,1,128]`, context prepending, `input`/`state`/`sr` → `output`/`stateN` I/O names)
- `process(&mut self, audio_frame: &[f32]) -> Result<f32>` method (returns speech probability 0.0–1.0)
- `reset(&mut self)` method to zero all internal state
- `Inference::use_silero_vad()` factory method
- Module wiring: `pub mod vad` in `lib.rs`, `mod.rs` with re-exports
- Unit tests (Send trait, input validation, reset behavior)
- Integration test (requires ONNX model, `#[ignore]`)

### Out of Scope

- Speech segment detection / timestamp extraction (higher-level logic that consumers build on top of `process()`)
- Automatic model downloading
- Silero VAD v4 support (separate h/c tensors — v5 is current)
- Audio resampling — callers must provide 16kHz mono f32 audio
- Streaming wrapper / Sink+Stream pattern (can be added later if needed)

## Prerequisites

- Silero VAD v5 ONNX model file (`silero_vad.onnx`) — downloadable from https://github.com/snakers4/silero-vad
- Place at `data/silero/silero_vad.onnx` for integration tests

## Context for Implementer

> This section is critical for cross-session continuity. Write it for an implementer who has never seen the codebase.

- **Patterns to follow:**
  - Module structure: Follow `crates/inference/src/tts/pocket/` — `mod.rs` for module declarations + re-exports, `core.rs` equivalent for ONNX logic, separate test file
  - ONNX session usage: Follow `crates/inference/src/asr/sherpa/asrcore.rs:33-76` — create `onnx::Value` with `from_slice`, call `session.run(&inputs, &output_names)`, extract results with `extract_tensor::<f32>()`
  - State carry pattern: Follow `crates/inference/src/asr/sherpa/asrcore.rs:72-73` — replace old state vectors with new outputs each inference step
  - Factory method: Follow `crates/inference/src/inference.rs:149-165` (`use_streaming_asr`) — accept path, create session via `self.onnx_session()`, construct the struct

- **Conventions:**
  - Error type: `crate::error::InferError` with `Result<T> = std::result::Result<T, InferError>`, use `InferError::Runtime(msg)` for runtime errors, `InferError::Onnx` auto-converts from `onnx::OnnxError`
  - Logging: `base::log_info!()`, `base::log_warn!()`, `base::log_error!()`
  - Tests: Place in `tests/` subdirectory, reference via `#[cfg(test)] #[path = "tests/file.rs"] mod test_mod;` in `mod.rs`
  - Integration tests requiring model files use `#[ignore]`

- **Key files:**
  - `crates/onnx/src/session.rs` — `Session` API: `run(&mut self, inputs, output_names)`, `input_count()`, `input_name()`, `input_shape()`
  - `crates/onnx/src/value.rs` — `Value::from_slice::<T>(shape, data)`, `Value::zeros::<T>(shape)`, `extract_tensor::<T>()`, `tensor_shape()`
  - `crates/inference/src/inference.rs` — `Inference` struct with `onnx_session()` helper and model factory methods
  - `crates/inference/src/error.rs` — `InferError` enum, `Result<T>` type alias

- **Gotchas:**
  - `onnx::Value::from_slice` takes `&[usize]` for shape (not `&[i64]`)
  - `Session::run` takes `&[(&str, &Value)]` for inputs and `&[&str]` for output names
  - `Session::run` requires `&mut self` — plan for mutable access
  - `Value::zeros` resolves negative dims (dynamic) to 1, but 0 dims stay as literal 0
  - The ONNX crate is a custom FFI wrapper (not the `ort` crate) — use `onnx::Value`, not `ndarray`

- **Domain context:**
  - Silero VAD v5 model I/O: inputs are `input` (audio frame `[1, N]`), `state` (`[2,1,128]`), `sr` (sample rate `[1]`); outputs are `output` (speech prob `[1,1]`), `stateN` (updated state `[2,1,128]`)
  - Context prepending: v5 prepends 64 samples of context (last 64 samples from previous frame) to the input frame, making effective input `[1, 64+512]` = `[1, 576]`
  - Sample rate: 16000 Hz (16kHz). Frame size: 512 samples (32ms window)
  - Speech probability threshold: typically 0.5 (caller decides, not our concern)

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create VAD module structure and SileroVad core
- [x] Task 2: Wire into Inference factory and lib.rs
- [x] Task 3: Integration test with ONNX model

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create VAD module structure and SileroVad core

**Objective:** Create the `vad` module with `SileroVad` struct that loads the Silero VAD v5 ONNX model, maintains internal state, and processes audio frames to return speech probabilities.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/vad/mod.rs`
- Create: `crates/inference/src/vad/silero.rs`
- Test: `crates/inference/src/vad/tests/silero_vad_test.rs`

**Key Decisions / Notes:**

- `SileroVad` struct holds: `session: Session`, `state: onnx::Value` (shape `[2,1,128]`, f32, zero-initialized), `context: Vec<f32>` (length 64, zero-initialized), `sample_rate: i64` (fixed at 16000)
- `SileroVad::new(session: Session) -> Result<Self>` — takes an already-loaded session, initializes state and context to zeros
- `SileroVad::process(&mut self, audio_frame: &[f32]) -> Result<f32>` — prepends context (last 64 samples) to frame, runs ONNX session, updates state + context, returns probability
- `SileroVad::reset(&mut self)` — zeros state tensor and context buffer
- Input validation: frame must be exactly 512 samples of f32 audio normalized to [-1, 1]
- Follow `asrcore.rs` pattern for ONNX input/output handling (see `crates/inference/src/asr/sherpa/asrcore.rs:33-76`)
- `mod.rs` declares `pub mod silero;` and re-exports `pub use silero::SileroVad;`, plus `#[cfg(test)]` test module reference

**Definition of Done:**

- [ ] All tests pass (unit + compile)
- [ ] No diagnostics errors
- [ ] `SileroVad` is `Send` (verified by compile-time test)
- [ ] `process()` validates frame size is 512 and returns `InferError::Runtime` for wrong sizes
- [ ] `reset()` zeros both state tensor and context buffer
- [ ] Context prepending produces input tensor of shape `[1, 576]` (64 context + 512 frame)
- [ ] State tensor is updated after each `process()` call (integration test verifies state differs after inference, same pattern as `asrcore.rs:72-73`)

**Verify:**

- `cargo test -p inference vad -- --nocapture` — unit tests pass
- `cargo check -p inference` — no errors

### Task 2: Wire into Inference factory and lib.rs

**Objective:** Add `pub mod vad;` to `lib.rs` and add `use_silero_vad()` factory method to the `Inference` struct.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/inference/src/lib.rs`
- Modify: `crates/inference/src/inference.rs`

**Key Decisions / Notes:**

- Add `pub mod vad;` to `lib.rs` alongside existing `pub mod asr;`, `pub mod tts;`, etc.
- Factory method signature: `pub fn use_silero_vad(&self, model_path: impl AsRef<Path>) -> Result<crate::vad::SileroVad, InferError>`
- Implementation: call `self.onnx_session(model_path)?` then `crate::vad::SileroVad::new(session)`
- Follow the pattern of `use_pose_detector` at `inference.rs:48-53` (single model path, simple construction)

**Definition of Done:**

- [ ] All tests pass
- [ ] No diagnostics errors
- [ ] `Inference::use_silero_vad()` compiles and is callable
- [ ] `pub mod vad` is exported from the inference crate

**Verify:**

- `cargo check -p inference` — no errors
- `cargo test -p inference -- --nocapture` — all tests pass

### Task 3: Integration test with ONNX model

**Objective:** Write an integration test that loads the real Silero VAD model, processes a synthetic audio signal, and verifies the model returns valid speech probabilities.

**Dependencies:** Task 2

**Files:**

- Modify: `crates/inference/src/vad/tests/silero_vad_test.rs`

**Key Decisions / Notes:**

- Test is `#[ignore]` (requires ONNX model at `data/silero/silero_vad.onnx`)
- Test `test_silero_vad_integration`:
  1. Create `Inference::cpu()`
  2. Call `inference.use_silero_vad("data/silero/silero_vad.onnx")`
  3. Generate synthetic silence (512 zeros) and synthetic speech (512 samples of sine wave at 440Hz)
  4. Process silence → expect low probability (< 0.5)
  5. Process sine wave for multiple frames → verify probabilities are in range [0.0, 1.0]
  6. Call `reset()`, process silence again → verify state was actually reset (probability similar to first call)
- Test `test_silero_vad_multiple_frames`: Feed 100 frames of silence, verify all probabilities are valid floats in [0.0, 1.0]

**Definition of Done:**

- [ ] All non-ignored tests pass
- [ ] No diagnostics errors
- [ ] Integration test runs successfully when model file is present (`cargo test -p inference silero_vad_integration -- --ignored --nocapture`)
- [ ] Silence frames (512 zeros) produce probabilities < 0.3; all probabilities in [0.0, 1.0] with no NaN/Inf; `reset()` causes probability to return to silence-level values

**Verify:**

- `cargo test -p inference vad -- --nocapture` — unit tests pass
- `cargo test -p inference silero_vad_integration -- --ignored --nocapture` — integration test passes (requires model)

## Testing Strategy

- **Unit tests:** `SileroVad` is `Send`, input frame size validation (reject != 512), reset zeroing behavior
- **Integration tests:** Full ONNX model loading + inference with synthetic audio, multi-frame processing, state reset verification. Marked `#[ignore]` since they require the model file.
- **Manual verification:** Run integration test with `--ignored` flag when model file is available

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Silero VAD v5 I/O names differ from reference | Low | High | Verify input/output names by inspecting model metadata in integration test; document expected names (`input`, `state`, `sr` → `output`, `stateN`) |
| State tensor shape mismatch between v4 and v5 | Low | Med | Target v5 only (state `[2,1,128]`); document v5 requirement in struct docs |
| Context prepending off-by-one | Med | Med | Unit test verifying input tensor shape is `[1, 576]` (64 + 512); compare against reference implementation |

## Goal Verification

> Derived from the plan's goal using goal-backward methodology. The spec-reviewer-goal agent verifies these criteria during verification.

### Truths (what must be TRUE for the goal to be achieved)

- `SileroVad` can be constructed from an ONNX session and processes audio frames returning speech probabilities
- `Inference::use_silero_vad(path)` loads the model and returns a ready-to-use `SileroVad`
- Frame-by-frame processing carries internal state across calls (stateful inference)
- `reset()` restores the VAD to its initial state
- The module is properly wired into the inference crate (`pub mod vad` in `lib.rs`)

### Artifacts (what must EXIST to support those truths)

- `crates/inference/src/vad/mod.rs` — module declaration with re-exports
- `crates/inference/src/vad/silero.rs` — `SileroVad` struct with `new()`, `process()`, `reset()`
- `crates/inference/src/vad/tests/silero_vad_test.rs` — unit tests + integration test
- Modified `crates/inference/src/lib.rs` — `pub mod vad;`
- Modified `crates/inference/src/inference.rs` — `use_silero_vad()` factory method

### Key Links (critical connections that must be WIRED)

- `Inference::use_silero_vad()` → `self.onnx_session()` → `SileroVad::new(session)`
- `SileroVad::process()` → `session.run()` with correct input names (`input`, `state`, `sr`) → extract output (`output`, `stateN`)
- `lib.rs` `pub mod vad;` → `vad/mod.rs` → `pub use silero::SileroVad;`

## Open Questions

- None — the Silero VAD v5 interface is well-documented and the codebase patterns are clear.

### Deferred Ideas

- Speech segment detection (start/end timestamps) as a higher-level wrapper
- Sink/Stream pattern for streaming VAD (similar to PocketTts)
- Support for Silero VAD v4 (older h/c tensor format)
- Configurable sample rate (8kHz support)
