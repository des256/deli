# Quantized Phi 4 LLM Implementation Plan

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

**Goal:** Add a quantized Phi 4 LLM module to `deli-infer`, following the same structural pattern as the existing `pose_detector` and `asr` modules.

**Architecture:** Create an `llm` module containing a `Phi4` struct that wraps `candle-transformers`' `quantized_phi3::ModelWeights` (Phi 4 uses the Phi 3 architecture in candle). The struct loads from GGUF files, exposes an async `forward` method for text generation, and integrates with the `Inference` factory. The `ModelWeights::forward` requires `&mut self` (KV cache), so the model is wrapped in `Arc<Mutex<>>` for thread-safe async use via `spawn_blocking`.

**Tech Stack:** `candle-transformers` 0.9 (quantized_phi3 model, LogitsProcessor, Sampling), `candle-core` 0.9 (GGUF loading, Tensor), `tokenizers` 0.21

## Scope

### In Scope

- New `src/llm/` module directory with `mod.rs` and `phi4.rs`
- `Phi4` struct with `new()` (GGUF + tokenizer loading) and async `forward()` (text generation)
- `Inference::use_phi4()` factory method
- Public re-exports from `lib.rs`
- Unit and integration tests with CUDA feature gating
- `candle-transformers` dependency with cuda feature

### Out of Scope

- Enum-based multi-LLM abstraction (planned for later)
- HuggingFace Hub model downloading (models loaded from local paths)
- Flash attention support
- Streaming token output
- Chat/conversation history management

## Prerequisites

- `candle-transformers = "0.9"` crate available (confirmed: 0.9.2)
- For real model tests: a Phi 4 GGUF file and tokenizer.json in `models/phi-4/`
- CUDA toolkit installed for `--features cuda` tests

## Context for Implementer

- **Patterns to follow:** The `Phi4` struct follows the same pattern as `SpeechRecognizer` in `src/asr/recognizer.rs:12-154` — wraps a model in `Arc`, uses `tokio::task::spawn_blocking` for inference, created via an `Inference` factory method.
- **Conventions:** Modules are directories under `src/` with `mod.rs`. Public types are re-exported from `lib.rs`. The `Inference` struct in `src/inference.rs` has factory methods prefixed with `use_` (e.g., `use_pose_detector`, `use_speech_recognizer`).
- **Key files:**
  - `src/inference.rs` — Factory struct with device management and `use_*` methods
  - `src/error.rs` — `InferError` enum, needs no changes (already has all needed variants)
  - `src/lib.rs` — Module declarations and public re-exports
  - `src/asr/recognizer.rs` — Reference implementation for the async inference pattern
  - `src/pose_detector/detector.rs` — Another reference for the `spawn_blocking` pattern
- **Gotchas:**
  - Phi 4 uses `candle_transformers::models::quantized_phi3::ModelWeights`, NOT `quantized_phi`. The candle example maps `Which::Phi4` to `Phi3::from_gguf(use_flash_attn, ct, reader, device)`.
  - `ModelWeights::forward(&mut self, xs, pos)` takes `&mut self` because of internal KV cache and mask cache. This requires `Mutex` wrapping (unlike pose/ASR models which are `&self`).
  - `ModelWeights::from_gguf` takes `(use_flash_attn: bool, ct: Content, reader: &mut R, device: &Device)` — the first param is `use_flash_attn`, which we'll set to `false`.
  - GGUF loading uses `candle_core::quantized::gguf_file::Content::read(&mut file)`, not safetensors.
  - The `cuda` feature in `Cargo.toml` must also chain through `candle-transformers/cuda`.
- **Domain context:** Quantized models use GGUF format (from llama.cpp ecosystem) rather than safetensors. The model weights are quantized (e.g., Q4_K_M) for reduced memory. Inference is autoregressive: tokenize prompt → forward all tokens → sample next token → repeat until EOS or max length.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: LLM module scaffolding and Phi4 struct with GGUF loading
- [x] Task 2: Async forward method with text generation
- [x] Task 3: Inference factory integration and integration tests

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: LLM module scaffolding and Phi4 struct with GGUF loading

**Objective:** Create the `llm` module directory with `Phi4` struct that loads a quantized Phi 4 model from a GGUF file and a tokenizer.

**Dependencies:** None

**Files:**

- Create: `crates/deli-infer/src/llm/mod.rs`
- Create: `crates/deli-infer/src/llm/phi4.rs`
- Modify: `crates/deli-infer/Cargo.toml` (add `candle-transformers` dep, update `cuda` feature)
- Modify: `crates/deli-infer/src/lib.rs` (add `mod llm` and re-export `Phi4`)
- Test: `crates/deli-infer/tests/phi4_test.rs`

**Key Decisions / Notes:**

- Add `candle-transformers = "0.9"` and `tracing = "0.1"` to `[dependencies]`
- Update `cuda` feature to include `candle-transformers/cuda`: `cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]`
- `Phi4` struct fields: `model: Arc<Mutex<ModelWeights>>`, `tokenizer: Arc<Tokenizer>`, `device: Device`
- `Phi4::new(model_path, tokenizer_path, device) -> Result<Self, InferError>` loads GGUF via `gguf_file::Content::read`, then `ModelWeights::from_gguf(false, content, &mut file, &device)`
- `mod.rs` re-exports `Phi4` from `phi4.rs`
- Tests: verify construction fails with missing file, verify struct is `Send + Sync`

**Definition of Done:**

- [ ] `Phi4::new()` loads GGUF model and tokenizer from file paths
- [ ] `Phi4` is `Send + Sync` (required for async use across `.await` points)
- [ ] Construction error returns `InferError` for missing files
- [ ] `cargo build -p deli-infer` succeeds
- [ ] Unit tests pass for construction and error cases

**Verify:**

- `cargo test -p deli-infer phi4 -- -q` — phi4 tests pass
- `cargo build -p deli-infer` — crate builds without errors

### Task 2: Async forward method with text generation

**Objective:** Implement `Phi4::forward` as an async method that takes a prompt string and generates text using autoregressive sampling.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-infer/src/llm/phi4.rs` (add `forward` method and generation logic)
- Test: `crates/deli-infer/tests/phi4_test.rs` (add forward tests)

**Key Decisions / Notes:**

- Signature: `pub async fn forward(&self, prompt: &str, sample_len: usize) -> Result<String, InferError>`
- Generation flow (inside `spawn_blocking`):
  1. Tokenize `prompt` with `self.tokenizer.encode(prompt, true)`
  2. Create `LogitsProcessor` with `Sampling::ArgMax` (deterministic, simplest default)
  3. Forward all prompt tokens at once: `model.forward(&input_tensor, 0)`
  4. Sample next token from logits
  5. Loop for `sample_len` iterations: forward single token, sample, check for EOS (`<|endoftext|>`)
  6. Decode all generated tokens back to text via tokenizer
- Use `Arc<Mutex<ModelWeights>>` — lock the mutex inside `spawn_blocking`, run the full generation loop, release when done
- EOS token: look up `<|endoftext|>` in tokenizer vocab (same as candle example)
- Clone `Arc`s before moving into `spawn_blocking` closure
- The method resets the KV cache implicitly: `ModelWeights::forward` resets cache masks when it sees `seq_len > 1` for the first call

**Definition of Done:**

- [ ] `forward` takes a prompt string and returns generated text
- [ ] Generation stops at EOS token or `sample_len` limit
- [ ] Method is async and uses `spawn_blocking` for compute
- [ ] Tests verify forward completes without error (with random-weight model via GGUF)
- [ ] Tests verify empty prompt returns error

**Verify:**

- `cargo test -p deli-infer phi4 -- -q` — all phi4 tests pass
- `cargo test -p deli-infer phi4 --features cuda -- -q` — CUDA tests pass (if GPU available)

### Task 3: Inference factory integration and integration tests

**Objective:** Wire `Phi4` into the `Inference` factory and add integration tests including CUDA-gated tests.

**Dependencies:** Task 2

**Files:**

- Modify: `crates/deli-infer/src/inference.rs` (add `use_phi4` method)
- Modify: `crates/deli-infer/tests/phi4_test.rs` (add factory and CUDA tests)
- Modify: `crates/deli-infer/tests/inference_test.rs` (add `use_phi4` signature test)

**Key Decisions / Notes:**

- `Inference::use_phi4(model_path, tokenizer_path) -> Result<Phi4, InferError>` — follows the `use_speech_recognizer` pattern, passes `self.device.clone()`
- CUDA test pattern: `#[cfg(feature = "cuda")]` gated tests that create `Inference::cuda(0)` and verify `use_phi4` works on GPU
- Real model test (gated on file existence): load from `../../models/phi-4/phi-4-q4.gguf` and `../../models/phi-4/tokenizer.json`, run a short forward pass, verify text output is non-empty
- Add a test that creates `Phi4` via the `Inference` factory and verifies error for nonexistent paths

**Definition of Done:**

- [ ] `Inference::use_phi4()` creates a `Phi4` with the correct device
- [ ] `#[cfg(feature = "cuda")]` test verifies CUDA device propagation
- [ ] Integration test with real model (skipped if model files absent) runs forward and gets text
- [ ] All existing tests still pass (no regressions)
- [ ] `cargo test -p deli-infer -- -q` — full test suite passes

**Verify:**

- `cargo test -p deli-infer -- -q` — all tests pass
- `cargo test -p deli-infer --features cuda -- -q` — CUDA tests pass
- `cargo build -p deli-infer --features cuda` — builds with CUDA

## Testing Strategy

- **Unit tests:** `Phi4` construction with missing files (error paths), `Send + Sync` trait bounds, forward with empty prompt validation
- **Integration tests (CPU):** Build a GGUF model fixture in temp dir (or skip if not feasible — GGUF format is complex), test the full `Inference::use_phi4` → `forward` pipeline
- **Integration tests (CUDA):** `#[cfg(feature = "cuda")]` gated tests that verify device propagation and model loading on GPU
- **Real model tests:** Gated on file existence (`models/phi-4/`), run actual inference with a short prompt and verify non-empty output
- **Note on GGUF test fixtures:** Unlike safetensors (which can be created from VarMap), GGUF files require specific metadata and quantized tensor format. Unit tests that need a model will use real GGUF files gated on existence, or test error paths only. The `forward` happy-path test requires a real GGUF file.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| GGUF test fixtures hard to create programmatically | High | Medium | Gate happy-path tests on real model file existence; test error paths without model files |
| `Mutex` contention on model limits throughput | Low | Low | Single inference at a time is acceptable for current use; future optimization can use per-request model cloning or a pool |
| `candle-transformers` 0.9 API differs from GitHub main | Low | High | Pin to `"0.9"` matching existing candle-core/candle-nn versions; verified `from_gguf` signature on GitHub |
| CUDA tests fail in CI without GPU | Medium | Low | All CUDA tests gated behind `#[cfg(feature = "cuda")]`; CI runs without `cuda` feature by default |

## Open Questions

- What default `sample_len` should `forward` use if the caller wants a sensible default? (Currently requires explicit value)
- Should `forward` accept generation parameters (temperature, top_p, etc.) or use fixed defaults? (Current plan: ArgMax/greedy sampling for simplicity; parameters can be added later)
