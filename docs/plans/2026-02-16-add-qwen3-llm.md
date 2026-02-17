# Add Qwen3 LLM Implementation Plan

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

**Goal:** Replace the Phi 4 LLM module with Qwen3-8B (Q4_K_M quantized GGUF) for text generation. Phi 4 (14B, 8.4GB) is too large; Qwen3-8B is smaller and has native support in candle-transformers 0.9.2.

**Architecture:** Create a `Qwen3` struct in `crates/deli-infer/src/llm/qwen3.rs` using `candle_transformers::models::quantized_qwen3::ModelWeights`. The existing `Phi4` module/struct will be replaced. The public API pattern (constructor + async `forward`) stays identical.

**Tech Stack:** `candle-transformers` 0.9 (`quantized_qwen3::ModelWeights`), `tokenizers` 0.21. Model files at `models/qwen3/`.

## Scope

### In Scope

- Create `crates/deli-infer/src/llm/qwen3.rs` with `Qwen3` struct
- Update `mod.rs` re-export: `Phi4` → `Qwen3`
- Update `lib.rs` re-export: `Phi4` → `Qwen3`
- Rename factory method `use_phi4` → `use_qwen3` in `inference.rs`
- Delete `phi4.rs`
- Create `qwen3_test.rs`, delete `phi4_test.rs`
- Update `inference_test.rs` references
- Run CUDA tests with real model

### Out of Scope

- Chat template formatting (ChatML wrapping)
- Downloading models automatically
- Removing old phi-4 model files from `models/`

## Prerequisites

- Model files available at `models/qwen3/` (GGUF file + `tokenizer.json`)
- CUDA toolkit installed for `--features cuda` tests

## Context for Implementer

- **Patterns to follow:** The existing `Phi4` struct at `crates/deli-infer/src/llm/phi4.rs` is the template. The new `Qwen3` struct follows the exact same pattern: constructor loads GGUF + tokenizer, async `forward` runs generation in `spawn_blocking`.
- **Conventions:** All inference structs are `Clone` (via `Arc<Mutex<>>` for the model). Factory methods on `Inference` follow the `use_<name>` pattern (see `inference.rs:27` for `use_pose_detector`, `inference.rs:40` for `use_phi4`).
- **Key files:**
  - `crates/deli-infer/src/llm/phi4.rs` — current LLM impl, will be replaced
  - `crates/deli-infer/src/llm/mod.rs` — module re-exports
  - `crates/deli-infer/src/lib.rs` — crate-level re-exports
  - `crates/deli-infer/src/inference.rs` — `Inference` struct with factory methods
  - `crates/deli-infer/src/error.rs` — `InferError` enum used for all errors
  - `crates/deli-infer/tests/phi4_test.rs` — existing test patterns
  - `crates/deli-infer/tests/inference_test.rs` — factory signature test
- **Gotchas:**
  - `quantized_qwen3::ModelWeights::from_gguf` takes 3 args `(ct, reader, device)` — NOT 4 like phi3 (no `use_flash_attn` param)
  - `phi4.rs` has uncommitted changes (added `<|end|>` token lookup) from a previous session. These are irrelevant and will be superseded when phi4.rs is deleted.
  - Qwen3's `forward` returns logits only for the last token position (shape `[batch, vocab_size]`), same as phi3. After `squeeze(0)`, this becomes `[vocab_size]` for `LogitsProcessor::sample`.
  - Qwen3 handles tied embeddings internally (falls back to `token_embd.weight` if `output.weight` missing in GGUF).
- **Domain context:** Qwen3-8B uses ChatML special tokens. EOS tokens are likely `<|endoftext|>` (end of text) and `<|im_end|>` (end of assistant turn). Must verify from actual `models/qwen3/tokenizer.json` during implementation.

## Feature Inventory

### Files Being Replaced

| Old File | Functions/Classes | Mapped to Task |
| --- | --- | --- |
| `crates/deli-infer/src/llm/phi4.rs` | `Phi4` struct, `Phi4::new()`, `Phi4::forward()` | Task 1 |
| `crates/deli-infer/src/llm/mod.rs` | `mod phi4; pub use phi4::Phi4;` | Task 1 |
| `crates/deli-infer/src/lib.rs` | `pub use llm::Phi4;` | Task 1 |
| `crates/deli-infer/src/inference.rs` | `use_phi4()` factory method | Task 1 |
| `crates/deli-infer/tests/phi4_test.rs` | All 7 test functions | Task 2 |
| `crates/deli-infer/tests/inference_test.rs` | `test_use_phi4_signature()` | Task 2 |

### Feature Mapping Verification

- [x] All old files listed above
- [x] All functions/classes identified
- [x] Every feature has a task number
- [x] No features accidentally omitted

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create Qwen3 module and update all references
- [x] Task 2: Create tests and verify with real model on CUDA

**Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create Qwen3 module and update all references

**Objective:** Create the `Qwen3` struct using `quantized_qwen3::ModelWeights` and update all module re-exports and factory methods.

**Dependencies:** None

**Files:**

- Create: `crates/deli-infer/src/llm/qwen3.rs`
- Modify: `crates/deli-infer/src/llm/mod.rs`
- Modify: `crates/deli-infer/src/lib.rs`
- Modify: `crates/deli-infer/src/inference.rs`
- Delete: `crates/deli-infer/src/llm/phi4.rs`

**Key Decisions / Notes:**

- Copy structure from `phi4.rs` but change:
  - Import: `use candle_transformers::models::quantized_qwen3::ModelWeights;` (was `quantized_phi3`)
  - `from_gguf` call: `ModelWeights::from_gguf(content, &mut file, &device)` — 3 args (remove the `false` first arg)
  - Struct name: `Qwen3` (was `Phi4`)
  - Doc comments: reference "Qwen3 8B" (was "Phi 4 Mini Instruct")
  - EOS token: read `models/qwen3/tokenizer.json` to find correct token names. Expected: `<|endoftext|>` for EOS, `<|im_end|>` for end-of-turn
  - Token lookup policy: primary EOS token (e.g. `<|endoftext|>`) is REQUIRED — fail at construction if absent (wrong tokenizer). End-of-turn token (e.g. `<|im_end|>`) is OPTIONAL — use `Option<u32>` and only check during generation if present
- In `mod.rs`: change to `mod qwen3; pub use qwen3::Qwen3;`
- In `lib.rs`: change to `pub use llm::Qwen3;`
- In `inference.rs`: rename `use_phi4` → `use_qwen3`, update return type to `crate::Qwen3`

**Definition of Done:**

- [ ] `qwen3.rs` exists with `Qwen3` struct using `quantized_qwen3::ModelWeights`
- [ ] `from_gguf` called with 3 args (ct, reader, device)
- [ ] EOS token names match Qwen3 tokenizer (verified from `models/qwen3/tokenizer.json`)
- [ ] `phi4.rs` deleted
- [ ] `mod.rs` re-exports `Qwen3`
- [ ] `lib.rs` re-exports `Qwen3`
- [ ] `inference.rs` has `use_qwen3` method returning `Result<crate::Qwen3, InferError>`
- [ ] `cargo build -p deli-infer` succeeds

**Verify:**

- `cargo build -p deli-infer` — builds without errors

### Task 2: Create tests and verify with real model on CUDA

**Objective:** Create the test file for Qwen3, update inference_test.rs, and verify real model inference works on CUDA.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-infer/tests/qwen3_test.rs`
- Modify: `crates/deli-infer/tests/inference_test.rs`
- Delete: `crates/deli-infer/tests/phi4_test.rs`

**Key Decisions / Notes:**

- Mirror test structure from `phi4_test.rs` (7 test functions):
  1. `test_qwen3_construction_fails_for_missing_file` — CPU, no model needed
  2. `test_qwen3_send_sync` — compile-time check
  3. `test_inference_factory_signature` — tests `use_qwen3` exists
  4. `test_forward_empty_prompt_rejected` — verifies InferError type importable
  5. `test_forward_with_real_model` — CPU real model test (skip if files missing)
  6. `test_forward_empty_prompt_validation` — CPU, requires real model
  7. `test_cuda_device_propagation` — `#[cfg(feature = "cuda")]`
  8. `test_forward_with_cuda` — `#[cfg(feature = "cuda")]`, real model on GPU
- Model paths: `../../models/qwen3/<gguf-filename>` and `../../models/qwen3/tokenizer.json` — check `ls models/qwen3/` for actual GGUF filename
- In `inference_test.rs`: rename `test_use_phi4_signature` → `test_use_qwen3_signature`, update to call `use_qwen3`

**Definition of Done:**

- [ ] `qwen3_test.rs` exists with all test functions referencing `Qwen3` and `use_qwen3`
- [ ] `phi4_test.rs` deleted
- [ ] `inference_test.rs` updated: `test_use_qwen3_signature` calls `inference.use_qwen3()`
- [ ] `cargo test -p deli-infer --test qwen3_test -- -q` passes (CPU tests)
- [ ] `cargo test -p deli-infer --test inference_test -- -q` passes
- [ ] `cargo test -p deli-infer --test qwen3_test --features cuda -- -q` passes with real model
- [ ] Real model forward produces non-empty text output on CUDA

**Verify:**

- `cargo test -p deli-infer --test qwen3_test -- -q` — CPU tests pass
- `cargo test -p deli-infer --test inference_test -- -q` — inference tests pass
- `cargo test -p deli-infer --test qwen3_test --features cuda -- -q` — CUDA tests pass
- `cargo build -p deli-infer --features cuda` — builds cleanly

## Testing Strategy

- **Unit tests (CPU, no model):** Construction failure for missing files, Send+Sync bounds, factory method signature, InferError importability
- **Integration tests (CPU, real model):** Load `models/qwen3/` GGUF on CPU, run forward, verify non-empty output. Gracefully skip if model files not present.
- **Integration tests (CUDA, real model):** Load `models/qwen3/` GGUF on GPU, run forward, verify non-empty output. Use `match Inference::cuda(0)` pattern for graceful skip.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Qwen3 GGUF file incompatible with candle 0.9.2 | Low | High | candle-transformers 0.9.2 has dedicated `quantized_qwen3` module; read GGUF metadata keys `qwen3.*` |
| EOS token names differ from expected | Medium | Low | Read actual `models/qwen3/tokenizer.json` before hardcoding token names; fall back to common alternatives |
| Model file not present at `models/qwen3/` during testing | Medium | Low | All real-model tests check file existence and skip gracefully with a message |

## Open Questions

- What is the actual GGUF filename in `models/qwen3/`? (Need to `ls` during implementation)
- What are the exact EOS/end-of-turn token names in Qwen3's tokenizer? (Will read from `tokenizer.json`)
