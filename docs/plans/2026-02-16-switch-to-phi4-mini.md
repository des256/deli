# Switch to Phi 4 Mini Instruct Implementation Plan

Created: 2026-02-16
Status: PENDING
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

**Goal:** Switch the Phi 4 LLM module from the full Phi 4 (14B, ~8.4GB quantized) to Phi 4 Mini Instruct (3.8B, ~2.5GB quantized) for resource-constrained target systems.

**Architecture:** Phi 4 Mini Instruct uses `Phi3ForCausalLM` architecture (`"model_type": "phi3"` in config.json), which means candle's existing `quantized_phi3::ModelWeights` loads it directly — no new model code needed. The changes are: (1) add `<|end|>` as an additional stop token for instruct-model turn completion, (2) update documentation and test paths to reference the new model, (3) rename the model directory from `phi-4/` to `phi-4-mini/`.

**Tech Stack:** Same as before — `candle-transformers` 0.9 (`quantized_phi3::ModelWeights`), `tokenizers` 0.21. Model source: `bartowski/microsoft_Phi-4-mini-instruct-GGUF` on HuggingFace (Q4_K_M recommended, 2.49GB).

## Research Findings

### Size Comparison

| Model | Params | Q4 GGUF Size | Context Length |
| --- | --- | --- | --- |
| Phi 4 (current) | 14B | ~8.4GB | 16K |
| **Phi 4 Mini Instruct** | **3.8B** | **2.49GB (Q4_K_M)** | **128K** |

### Architecture Compatibility

Phi 4 Mini Instruct config.json confirms:
- `"model_type": "phi3"` and `"architectures": ["Phi3ForCausalLM"]`
- GGUF metadata uses `phi3.*` prefix — matches `quantized_phi3::ModelWeights::from_gguf`
- GQA: `num_attention_heads=24`, `num_key_value_heads=8` — candle handles via `head_count_kv`
- Partial rotary: `partial_rotary_factor=0.75` → GGUF sets `phi3.rope.dimension_count=96` — candle reads this
- 200K vocab (vs ~100K for Phi 4 full) — handled by embedding size in GGUF

### Tokenizer Compatibility

- EOS token: `<|endoftext|>` (token id 199999) — **same** token name as current code looks up
- Additional stop token needed: `<|end|>` (token id 200020) — marks end of assistant turn in instruct format. Without this, the model will generate beyond the assistant response into fake user turns.
- Tokenizer class: `GPT2Tokenizer` (loaded from `tokenizer.json` as before)

### GGUF Sources

Available from `bartowski/microsoft_Phi-4-mini-instruct-GGUF` on HuggingFace:
```bash
huggingface-cli download bartowski/microsoft_Phi-4-mini-instruct-GGUF \
  --include "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf" --local-dir models/phi-4-mini/
```

Tokenizer from the original model:
```bash
huggingface-cli download microsoft/Phi-4-mini-instruct \
  --include "tokenizer.json" --local-dir models/phi-4-mini/
```

## Scope

### In Scope

- Add `<|end|>` stop token to `forward()` generation loop in `phi4.rs`
- Update doc comments in `phi4.rs` to reference Phi 4 Mini Instruct
- Update test model paths from `models/phi-4/` to `models/phi-4-mini/`
- Update test GGUF filename references

### Out of Scope

- Changing the struct name `Phi4` (it's still the Phi 4 family)
- Changing the factory method name `use_phi4` (API stability)
- Adding chat template formatting (users compose prompts themselves)
- Downloading models automatically (models loaded from local paths)
- Removing the old Phi 4 model files (user manages model directory)

## Prerequisites

- Model files already available at `models/phi-4-mini/`:
  - `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` (2.49GB)
  - `tokenizer.json`
- CUDA toolkit installed — all real model tests must run with `--features cuda`

## Context for Implementer

- **Patterns to follow:** The `forward()` method in `phi4.rs:84-173` already has the generation loop with EOS detection at line 141. The change is adding a second stop token lookup and check.
- **Key files:**
  - `crates/deli-infer/src/llm/phi4.rs` — The Phi4 struct and forward method
  - `crates/deli-infer/tests/phi4_test.rs` — Integration tests with model file paths
- **Gotchas:**
  - The `<|end|>` token may not exist in all tokenizer files (e.g., non-instruct Phi models). The code should handle this gracefully — look it up but treat it as optional (only stop on it if found in vocab).
  - The `Phi4` struct and `use_phi4` factory method name stay the same for API stability — Phi 4 Mini IS a Phi 4 model.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add `<|end|>` stop token and update phi4.rs documentation
- [ ] Task 2: Update test paths and verify with real model

**Total Tasks:** 2 | **Completed:** 1 | **Remaining:** 1

## Implementation Tasks

### Task 1: Add `<|end|>` stop token and update phi4.rs documentation

**Objective:** Modify the generation loop in `forward()` to also stop at the `<|end|>` instruct turn-end token, and update doc comments to reference Phi 4 Mini Instruct.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-infer/src/llm/phi4.rs`
- Test: `crates/deli-infer/tests/phi4_test.rs`

**Key Decisions / Notes:**

- Look up `<|end|>` token in the tokenizer vocab alongside `<|endoftext|>`. Store it as `Option<u32>` since non-instruct tokenizers may not have it.
- In the generation loop (line 139-162), add a check: `if next_token == eos_token || Some(next_token) == end_token { break; }`
- Update the module doc comment (lines 9-16) and `forward()` doc comment to mention Phi 4 Mini Instruct compatibility.
- Write a unit test that verifies the `Phi4` struct is still `Send + Sync` (existing test).

**Definition of Done:**

- [ ] `forward()` stops generation at both `<|endoftext|>` and `<|end|>` tokens
- [ ] `<|end|>` token lookup is optional (no error if token not in vocab)
- [ ] Doc comments reference Phi 4 Mini Instruct as the target model
- [ ] `cargo build -p deli-infer` succeeds
- [ ] Existing phi4 tests still pass

**Verify:**

- `cargo test -p deli-infer phi4 -- -q` — all phi4 tests pass
- `cargo build -p deli-infer` — crate builds without errors

### Task 2: Update test paths and verify with real model

**Objective:** Update all test references from `models/phi-4/phi-4-q4.gguf` to `models/phi-4-mini/` paths for the new smaller model.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-infer/tests/phi4_test.rs`

**Key Decisions / Notes:**

- Change model path from `../../models/phi-4/phi-4-q4.gguf` to `../../models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` (matching bartowski's naming convention)
- Change tokenizer path from `../../models/phi-4/tokenizer.json` to `../../models/phi-4-mini/tokenizer.json`
- Update print messages that reference model file locations
- Remove `#[ignore]` from real model tests since model files are available
- Run real model tests with CUDA to verify end-to-end inference

**Definition of Done:**

- [ ] All test paths reference `models/phi-4-mini/` directory
- [ ] GGUF filename matches `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf`
- [ ] `cargo test -p deli-infer phi4 -- -q` — unit tests pass
- [ ] `cargo test -p deli-infer phi4 --features cuda -- -q` — CUDA real model tests pass
- [ ] Real model forward produces non-empty text output

**Verify:**

- `cargo test -p deli-infer phi4 -- -q` — unit tests pass
- `cargo test -p deli-infer phi4 --features cuda -- -q` — CUDA tests pass with real model
- `cargo build -p deli-infer --features cuda` — builds cleanly with CUDA

## Testing Strategy

- **Unit tests (CPU, no model):** Construction failure, Send+Sync bounds, factory signature, InferError importability — all unchanged
- **Integration tests (CUDA, real model):** Load `models/phi-4-mini/` GGUF on GPU, run forward with a short prompt, verify non-empty text output. These tests run with `--features cuda`.
- **CUDA test pattern:** Use `match Inference::cuda(0)` with graceful skip if no GPU available at runtime

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `<|end|>` token not in all tokenizers | Medium | Low | Lookup is `Option<u32>` — gracefully ignored if absent |
| Phi 4 Mini GGUF has incompatible metadata | Very Low | High | Verified: config.json shows `model_type: phi3`, same arch as current code |
| RoPE frequency base differs for long contexts | Low | Low | Only affects sequences > 4096 tokens; short prompts use standard base 10000 |
| Bartowski GGUF naming changes | Low | Low | Test paths are easily updated; model path is a user parameter, not hardcoded |

## Open Questions

- None — architecture compatibility confirmed, tokenizer verified, GGUF sources identified.
