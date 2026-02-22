# Add Phi3, Llama, and SmolLM2 Model Implementations

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

**Goal:** Add three new LLM model backends — Phi3, Llama (TinyLlama 1.1B), and SmolLM2 (1.7B) — to the inference crate, following the same wrapper pattern as the existing Qwen3 implementation. Each model gets a public struct with `new()` constructor and async `forward()` method, a factory method on `Inference`, and corresponding tests.

**Architecture:** Each model wraps its respective `candle_transformers::models::quantized_*::ModelWeights` struct behind `Arc<Mutex<>>` with a `Tokenizer`, mirroring the `Qwen3` wrapper exactly. The key differences per model are:
- **Phi3**: Uses `quantized_phi3::ModelWeights::from_gguf(use_flash_attn, ct, reader, device)` — extra `bool` param, pass `false`
- **Llama (TinyLlama)**: Uses `quantized_llama::ModelWeights::from_gguf(ct, reader, device)` — same 3-arg signature as Qwen3
- **SmolLM2**: Also uses `quantized_llama::ModelWeights` — SmolLM2 is Llama-architecture and GGUF files use `llama.*` metadata keys

**Tech Stack:** candle-core, candle-transformers (quantized_phi3, quantized_llama), tokenizers, tokio

## Scope

### In Scope

- New wrapper structs: `Phi3`, `Llama`, `SmolLm2` in `crates/inference/src/llm/`
- Factory methods on `Inference`: `use_phi3()`, `use_llama()`, `use_smolm2()`
- Re-exports from `lib.rs`
- Tests matching the Qwen3 test pattern (construction failure, Send+Sync, factory signature, empty prompt, CUDA forward)
- Module registration in `llm/mod.rs`

### Out of Scope

- Downloading model files (user handles this)
- Changes to the chat experiment or testy binary (these stay on Qwen3)
- Streaming/token-by-token generation
- Flash attention support (pass `false` for Phi3)
- Any changes to the existing Qwen3 implementation

## Prerequisites

- candle-transformers 0.9 already provides `quantized_phi3` and `quantized_llama` modules (verified)
- No new crate dependencies needed

## Context for Implementer

- **Pattern to follow:** The exact pattern is in `crates/inference/src/llm/qwen3.rs` — `Arc<Mutex<ModelWeights>>` wrapper with `new()` and `async forward()`. Copy this structure.
- **Conventions:** Each model lives in its own file under `crates/inference/src/llm/`. Public types are re-exported from `lib.rs`. Factory methods go on `Inference` in `inference.rs`.
- **Key files:**
  - `crates/inference/src/llm/qwen3.rs` — the template to follow
  - `crates/inference/src/llm/mod.rs` — module declarations and re-exports
  - `crates/inference/src/inference.rs` — `Inference` struct with factory methods
  - `crates/inference/src/lib.rs` — public re-exports
  - `crates/inference/tests/qwen3_test.rs` — test template
- **KV cache differences:**
  - Qwen3 (candle): has `clear_kv_cache()` — called before each generation
  - Phi3 (candle): KV cache auto-resets when `index_pos == 0` — no explicit clear needed
  - Llama (candle): KV cache auto-resets when `index_pos == 0` — no explicit clear needed
  - Our wrapper always starts with `index_pos = 0`, so auto-reset works. But Phi3/Llama don't expose `clear_kv_cache()` as a public method, so we must not call it.
- **Output tensor shape:** All three models return `(batch, vocab_size)` from `forward()`. With batch=1, `squeeze(0)` gives `(vocab_size,)` for sampling. Same as Qwen3.
- **EOS tokens:** Each model has different special tokens:
  - Phi3: `<|endoftext|>` (EOS), `<|end|>` (end-of-turn)
  - TinyLlama/Llama: `</s>` (EOS)
  - SmolLM2 (Llama arch): `</s>` (EOS), `<|im_end|>` (end-of-turn, if present)
- **Gotchas:**
  - `quantized_phi3::ModelWeights::from_gguf` takes an extra `use_flash_attn: bool` as first arg
  - `quantized_llama::ModelWeights` has no `clear_kv_cache()` — KV resets when `index_pos == 0`
  - SmolLM2 uses `quantized_llama` (NOT `quantized_smollm3` which is for SmolLM3, a different architecture)
  - The Llama `forward` uses `rope_i` (interleaved) vs Qwen3's `rope` — this is internal to candle, no impact on our wrapper

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Phi3 model wrapper
- [x] Task 2: Llama model wrapper (TinyLlama 1.1B)
- [x] Task 3: SmolLM2 model wrapper
- [x] Task 4: Integration — Inference factory methods and re-exports
- [x] Task 5: Tests for all three models

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Phi3 Model Wrapper

**Objective:** Create `Phi3` struct in `crates/inference/src/llm/phi3.rs` wrapping `candle_transformers::models::quantized_phi3::ModelWeights`.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/llm/phi3.rs`

**Key Decisions / Notes:**

- Follow `qwen3.rs` pattern exactly: `Arc<Mutex<ModelWeights>>`, `Arc<Tokenizer>`, `Device`, EOS tokens
- Constructor loads GGUF via `quantized_phi3::ModelWeights::from_gguf(false, ct, reader, &device)` — pass `false` for flash_attn
- EOS token: `<|endoftext|>`, end-of-turn: `<|end|>`
- No `clear_kv_cache()` call needed — Phi3 auto-resets KV when `index_pos == 0`
- `forward()` returns `(batch, vocab_size)` — same squeeze(0) logic as Qwen3

**Definition of Done:**

- [ ] `phi3.rs` exists with `Phi3` struct implementing `Clone`, `Send`, `Sync`
- [ ] `Phi3::new(model_path, tokenizer_path, device)` loads GGUF and validates EOS token
- [ ] `Phi3::forward(prompt, sample_len)` generates text with EOS stopping
- [ ] Compiles without errors

**Verify:**

- `cargo check -p inference` — compiles cleanly

### Task 2: Llama Model Wrapper (TinyLlama 1.1B)

**Objective:** Create `Llama` struct in `crates/inference/src/llm/llama.rs` wrapping `candle_transformers::models::quantized_llama::ModelWeights`.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/llm/llama.rs`

**Key Decisions / Notes:**

- Follow `qwen3.rs` pattern: `Arc<Mutex<ModelWeights>>`, `Arc<Tokenizer>`, `Device`, EOS tokens
- Constructor loads GGUF via `quantized_llama::ModelWeights::from_gguf(ct, reader, &device)` — same 3 args as Qwen3
- EOS token: `</s>` (standard Llama EOS)
- No `clear_kv_cache()` — Llama auto-resets when `index_pos == 0`
- `forward()` returns `(batch, vocab_size)` — same squeeze(0) logic

**Definition of Done:**

- [ ] `llama.rs` exists with `Llama` struct implementing `Clone`, `Send`, `Sync`
- [ ] `Llama::new(model_path, tokenizer_path, device)` loads GGUF and validates EOS token
- [ ] `Llama::forward(prompt, sample_len)` generates text with EOS stopping
- [ ] Compiles without errors

**Verify:**

- `cargo check -p inference` — compiles cleanly

### Task 3: SmolLM2 Model Wrapper

**Objective:** Create `SmolLm2` struct in `crates/inference/src/llm/smolm2.rs` wrapping `candle_transformers::models::quantized_llama::ModelWeights` (SmolLM2 uses Llama architecture).

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/llm/smolm2.rs`

**Key Decisions / Notes:**

- SmolLM2 uses Llama architecture — GGUF files have `llama.*` metadata keys
- Wraps the same `quantized_llama::ModelWeights` as Task 2
- EOS token: `</s>`, optional end-of-turn: `<|im_end|>` (if present in tokenizer)
- Same KV cache behavior as Llama — no `clear_kv_cache()` needed

**Definition of Done:**

- [ ] `smolm2.rs` exists with `SmolLm2` struct implementing `Clone`, `Send`, `Sync`
- [ ] `SmolLm2::new(model_path, tokenizer_path, device)` loads GGUF and validates EOS token
- [ ] `SmolLm2::forward(prompt, sample_len)` generates text with EOS stopping
- [ ] Compiles without errors

**Verify:**

- `cargo check -p inference` — compiles cleanly

### Task 4: Integration — Inference Factory Methods and Re-exports

**Objective:** Wire up all three new models into `Inference` and the public API.

**Dependencies:** Task 1, Task 2, Task 3

**Files:**

- Modify: `crates/inference/src/llm/mod.rs` — add module declarations and re-exports
- Modify: `crates/inference/src/inference.rs` — add `use_phi3()`, `use_llama()`, `use_smolm2()` factory methods
- Modify: `crates/inference/src/lib.rs` — add public re-exports

**Key Decisions / Notes:**

- `mod.rs` pattern: `mod phi3; pub use phi3::Phi3;` (same as existing `mod qwen3; pub use qwen3::Qwen3;`)
- Factory methods follow `use_qwen3` pattern: take `model_path` and `tokenizer_path`, return `Result<Model, InferError>`
- `lib.rs` adds: `pub use llm::{Phi3, Llama, SmolLm2};`

**Definition of Done:**

- [ ] `Inference::use_phi3(model_path, tokenizer_path)` exists and compiles
- [ ] `Inference::use_llama(model_path, tokenizer_path)` exists and compiles
- [ ] `Inference::use_smolm2(model_path, tokenizer_path)` exists and compiles
- [ ] `inference::Phi3`, `inference::Llama`, `inference::SmolLm2` are publicly importable
- [ ] `cargo check -p inference` passes cleanly

**Verify:**

- `cargo check -p inference` — full crate compiles

### Task 5: Tests for All Three Models

**Objective:** Create test files for each model matching the Qwen3 test pattern.

**Dependencies:** Task 4

**Files:**

- Create: `crates/inference/tests/phi3_test.rs`
- Create: `crates/inference/tests/llama_test.rs`
- Create: `crates/inference/tests/smolm2_test.rs`

**Key Decisions / Notes:**

- Mirror `qwen3_test.rs` structure: construction failure test, Send+Sync test, factory signature test, InferError importable test
- Tests that need real model files (forward, empty prompt) use `if !model_path.exists() { return; }` guard — same pattern as Qwen3
- Model data paths:
  - Phi3: `data/phi3/` (model GGUF + tokenizer.json)
  - Llama: `data/llama/` (model GGUF + tokenizer.json)
  - SmolLM2: `data/smolm2/` (model GGUF + tokenizer.json)
- All tests require `cuda` feature (same as Qwen3 tests)

**Definition of Done:**

- [ ] `phi3_test.rs` has construction_failure, send_sync, factory_signature tests
- [ ] `llama_test.rs` has construction_failure, send_sync, factory_signature tests
- [ ] `smolm2_test.rs` has construction_failure, send_sync, factory_signature tests
- [ ] All tests compile and pass: `cargo test -p inference --features cuda`

**Verify:**

- `cargo test -p inference --features cuda` — all tests pass

## Testing Strategy

- **Unit tests (CPU, no model):** Construction failure with fake paths, Send+Sync trait verification, factory method signature, InferError importability — these run without model files
- **Integration tests (CUDA + model):** Forward with real model files, empty prompt validation, CUDA device propagation — these are skipped if model files aren't present
- **Manual verification:** Load each model, run a simple prompt, compare output quality

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SmolLM2 GGUF doesn't use `llama.*` metadata keys | Low | High | SmolLM2 is well-documented as Llama-arch; if metadata differs, add a fallback architecture prefix check in the constructor |
| Phi3 `from_gguf` signature changes in future candle versions | Low | Med | Pin candle-transformers 0.9 (already done in Cargo.toml) |
| EOS tokens vary across model quantizations | Med | Low | Look up EOS token by string; if not found, return clear error message indicating wrong tokenizer |

## Goal Verification

> Derived from the plan's goal using goal-backward methodology.

### Truths (what must be TRUE for the goal to be achieved)

- Users can construct a Phi3 model from GGUF + tokenizer and generate text
- Users can construct a Llama model from GGUF + tokenizer and generate text
- Users can construct a SmolLM2 model from GGUF + tokenizer and generate text
- All three models follow the same public API pattern as Qwen3 (new + forward)
- All three models are accessible through the `Inference` factory pattern

### Artifacts (what must EXIST to support those truths)

- `crates/inference/src/llm/phi3.rs` — Phi3 wrapper struct with new() and forward()
- `crates/inference/src/llm/llama.rs` — Llama wrapper struct with new() and forward()
- `crates/inference/src/llm/smolm2.rs` — SmolLm2 wrapper struct with new() and forward()
- `crates/inference/src/inference.rs` — use_phi3(), use_llama(), use_smolm2() factory methods
- `crates/inference/tests/phi3_test.rs` — Phi3 tests
- `crates/inference/tests/llama_test.rs` — Llama tests
- `crates/inference/tests/smolm2_test.rs` — SmolLm2 tests

### Key Links (critical connections that must be WIRED)

- `Inference::use_phi3()` → `Phi3::new()` → `quantized_phi3::ModelWeights::from_gguf()`
- `Inference::use_llama()` → `Llama::new()` → `quantized_llama::ModelWeights::from_gguf()`
- `Inference::use_smolm2()` → `SmolLm2::new()` → `quantized_llama::ModelWeights::from_gguf()`
- `lib.rs` re-exports `Phi3`, `Llama`, `SmolLm2` from `llm` module

## Open Questions

- None — the pattern is well-established and all three candle model backends are verified to exist.

### Deferred Ideas

- Add a unified `Llm` trait to abstract over model backends for easier swapping
- Add streaming token generation
- Add model comparison benchmark tooling
