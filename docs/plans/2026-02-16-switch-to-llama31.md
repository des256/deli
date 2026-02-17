# Switch LLM to Llama 3.1 8B Instruct Implementation Plan

Created: 2026-02-16
Status: PENDING
Approved: No
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

**Goal:** Replace the Phi 4 LLM module with Llama 3.1 8B Instruct. The original Phi 4 (14B, 8.4GB) was too large; Phi 4 Mini was tried but has tied embeddings incompatible with candle's loader. Llama 3.1 8B uses candle's `quantized_llama::ModelWeights` which has separate `output.weight` (no tied embedding issue).

**Architecture:** Llama 3.1 uses `candle_transformers::models::quantized_llama::ModelWeights`. Key differences from the current phi3 loader:
- `from_gguf(ct, reader, device)` — 3 args (no `use_flash_attn` param)
- GGUF metadata prefix: `llama.*` instead of `phi3.*`
- Separate Q/K/V/O attention projections (not fused `attn_qkv`)
- `forward(&mut self, x, index_pos)` — same signature as phi3

**Tech Stack:** `candle-transformers` 0.9 (`quantized_llama::ModelWeights`), `tokenizers` 0.21. Model files at `models/llama31/`.

## Research Findings (from previous session)

### Why NOT Phi 4 Mini
- Phi 4 Mini Instruct uses tied input/output embeddings ("shared input and output embedding")
- candle's `quantized_phi3::ModelWeights` expects separate `output.weight` tensor
- Loading fails: "cannot find tensor info for output.weight"

### Llama 3.1 Compatibility with candle
- `quantized_llama::ModelWeights` confirmed in candle-transformers 0.9.2
- `from_gguf(ct, reader, device)` reads `llama.*` metadata keys
- `forward(&mut self, x, index_pos) -> Result<Tensor>` — same interface
- Separate `output.weight` tensor — no tied embeddings issue
- GQA supported via `head_count_kv`

### Llama 3.1 Stop Tokens (NEEDS VERIFICATION)
- Likely uses `<|eot_id|>` for end-of-turn and `<|end_of_text|>` for EOS
- Must verify against actual tokenizer.json in `models/llama31/`

## Scope

### In Scope

- Rename `src/llm/phi4.rs` → `src/llm/llama31.rs`
- Rename struct `Phi4` → `Llama31`
- Switch model loader from `quantized_phi3::ModelWeights` to `quantized_llama::ModelWeights`
- Update `from_gguf` call (3 args instead of 4)
- Update EOS/stop token lookup for Llama 3.1 tokenizer
- Rename `mod.rs` re-export: `Phi4` → `Llama31`
- Rename factory method `use_phi4` → `use_llama31` in `inference.rs`
- Update `lib.rs` re-exports
- Rename and update test file `phi4_test.rs` → `llama31_test.rs`
- Update model paths to `models/llama31/`
- Run CUDA tests with real model

### Out of Scope

- Chat template formatting
- Downloading models automatically
- Removing old model files

## Prerequisites

- Model files available at `models/llama31/` (user confirmed)
- CUDA toolkit installed for `--features cuda` tests

## Context for Implementer

- **Current implementation:** `crates/deli-infer/src/llm/phi4.rs` contains `Phi4` struct using `quantized_phi3::ModelWeights`
- **Key change:** `quantized_llama::ModelWeights::from_gguf` takes 3 args `(ct, reader, device)` — NOT 4 (no `use_flash_attn` param)
- **Module structure:** `src/llm/mod.rs` re-exports, `src/lib.rs` declares mod + re-exports, `src/inference.rs` has `use_phi4` factory
- **phi4.rs has uncommitted changes** from Task 1 of old plan (added `<|end|>` token) — these will be superseded by the rename
- **Stop tokens:** Check `models/llama31/tokenizer.json` for actual EOS token names. Llama 3.1 Instruct typically uses `<|eot_id|>` (end of turn) and `<|end_of_text|>` (end of text)
- **Gotchas:**
  - The `quantized_llama` forward method uses `Option<(Tensor, Tensor)>` for KV cache (different from phi3's `KvCache`), but this is internal — our wrapper code just calls `model.forward(&input, pos)` which is the same interface

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [ ] Task 1: Rename module files and struct from Phi4 to Llama31
- [ ] Task 2: Switch model loader to quantized_llama and update forward()
- [ ] Task 3: Update tests, factory method, and verify with real model on CUDA

**Total Tasks:** 3 | **Completed:** 0 | **Remaining:** 3

## Implementation Tasks

### Task 1: Rename module files and struct from Phi4 to Llama31

**Objective:** Rename the LLM module from phi4 to llama31 throughout the codebase.

**Dependencies:** None

**Files:**

- Rename: `crates/deli-infer/src/llm/phi4.rs` → `crates/deli-infer/src/llm/llama31.rs`
- Modify: `crates/deli-infer/src/llm/mod.rs` (update module name and re-export)
- Modify: `crates/deli-infer/src/lib.rs` (update re-export from `Phi4` to `Llama31`)
- Modify: `crates/deli-infer/src/inference.rs` (rename `use_phi4` to `use_llama31`, update type)

**Key Decisions / Notes:**

- `git mv` is NOT needed since we're not committing — just create new file, delete old
- In `llama31.rs`: rename struct `Phi4` → `Llama31`, keep all methods
- In `mod.rs`: change `mod phi4; pub use phi4::Phi4;` → `mod llama31; pub use llama31::Llama31;`
- In `lib.rs`: update `pub use llm::Phi4;` → `pub use llm::Llama31;`
- In `inference.rs`: rename method and update return type

**Definition of Done:**

- [ ] `phi4.rs` deleted, `llama31.rs` exists with `Llama31` struct
- [ ] `mod.rs` re-exports `Llama31`
- [ ] `lib.rs` re-exports `Llama31`
- [ ] `inference.rs` has `use_llama31` method
- [ ] `cargo build -p deli-infer` succeeds

**Verify:**

- `cargo build -p deli-infer` — builds without errors

### Task 2: Switch model loader to quantized_llama and update forward()

**Objective:** Replace `quantized_phi3::ModelWeights` with `quantized_llama::ModelWeights` and update the `from_gguf` call and EOS token handling.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-infer/src/llm/llama31.rs`

**Key Decisions / Notes:**

- Change import: `use candle_transformers::models::quantized_llama::ModelWeights;` (was `quantized_phi3`)
- Change `from_gguf` call: `ModelWeights::from_gguf(content, &mut file, &device)` — 3 args, NO `false` first arg
- Check `models/llama31/tokenizer.json` for EOS tokens — update the token name lookup from `<|endoftext|>` to whatever Llama 3.1 uses (likely `<|end_of_text|>` or `<|eot_id|>`)
- Keep the optional end-of-turn token pattern from Task 1 of old plan
- Update doc comments to reference Llama 3.1 8B Instruct

**Definition of Done:**

- [ ] Uses `quantized_llama::ModelWeights` import
- [ ] `from_gguf` called with 3 args
- [ ] EOS token names match Llama 3.1 tokenizer
- [ ] `cargo build -p deli-infer` succeeds

**Verify:**

- `cargo build -p deli-infer` — builds without errors

### Task 3: Update tests, factory method, and verify with real model on CUDA

**Objective:** Rename test file, update all test references, and run real model inference on CUDA.

**Dependencies:** Task 2

**Files:**

- Delete: `crates/deli-infer/tests/phi4_test.rs`
- Create: `crates/deli-infer/tests/llama31_test.rs`
- Modify: `crates/deli-infer/tests/inference_test.rs` (if it references `use_phi4`)

**Key Decisions / Notes:**

- Copy test structure from phi4_test.rs, rename all `Phi4` → `Llama31`, `use_phi4` → `use_llama31`
- Update model paths to `../../models/llama31/` — need to know actual GGUF filename
- Check `ls models/llama31/` for actual filenames
- Remove `#[ignore]` from real model tests (model files available)
- Run with `--features cuda` for CUDA verification

**Definition of Done:**

- [ ] `phi4_test.rs` deleted, `llama31_test.rs` exists
- [ ] All tests reference `Llama31` struct and `use_llama31` method
- [ ] Model paths point to `models/llama31/`
- [ ] `cargo test -p deli-infer --test llama31_test -- -q` passes
- [ ] `cargo test -p deli-infer --test llama31_test --features cuda -- -q` passes with real model
- [ ] Real model forward produces non-empty text output

**Verify:**

- `cargo test -p deli-infer --test llama31_test -- -q` — unit tests pass
- `cargo test -p deli-infer --test llama31_test --features cuda -- -q` — CUDA tests pass
- `cargo build -p deli-infer --features cuda` — builds cleanly

## Testing Strategy

- **Unit tests (CPU, no model):** Construction failure, Send+Sync bounds, factory signature, InferError importability
- **Integration tests (CUDA, real model):** Load `models/llama31/` GGUF on GPU, run forward, verify non-empty output
- **CUDA test pattern:** Use `match Inference::cuda(0)` with graceful skip if no GPU

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Llama 3.1 GGUF format incompatible with candle 0.9 | Low | High | candle's quantized example is Llama-based; most tested architecture |
| EOS token names differ from expected | Medium | Low | Read actual tokenizer.json before hardcoding token names |
| inference_test.rs references use_phi4 | Medium | Low | Grep for all Phi4/phi4 references and update |

## Open Questions

- What are the actual filenames in `models/llama31/`? (GGUF file name, tokenizer.json)
- What are the exact EOS token names in Llama 3.1's tokenizer?
