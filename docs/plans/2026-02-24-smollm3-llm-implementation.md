# SmolLM3 LLM Implementation Plan

Created: 2026-02-24
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

**Goal:** Build a new ONNX-based LLM implementation `Smollm3` in `inference/src/llm/smollm3/` with a `forward` method that starts autoregressive text generation and an async `recv` method for streaming generated tokens one at a time.

**Architecture:** `Smollm3` wraps an ONNX `Session` and HuggingFace `Tokenizer`. The ONNX model uses KV cache — the first forward pass sends the full tokenized prompt with empty past key-value tensors, and subsequent passes send only the new token with updated KV cache. `forward(text)` tokenizes the input, spawns a `tokio::task::spawn_blocking` that runs the autoregressive KV-cached generation loop, and sends decoded token strings through a `tokio::sync::mpsc` channel. `recv()` is an async method that receives the next token from the channel.

**Tech Stack:** Rust, ONNX Runtime (via the `onnx` crate), HuggingFace `tokenizers` crate, `tokio` for async channel-based streaming.

## Scope

### In Scope

- `Smollm3` struct with `new()`, `forward()`, `recv()`
- Autoregressive generation loop with KV cache and greedy decoding (argmax)
- Dynamic discovery of ONNX model input/output names at construction time
- Streaming token output via tokio mpsc channel
- Module wiring into `llm/mod.rs` and `Inference` facade
- Unit and integration tests

### Out of Scope

- Chat template / conversation formatting (future work: combining sentences)
- Sampling strategies (top-p, temperature) — greedy only for now
- Sentence combining and further processing (explicitly deferred)

## Prerequisites

- ONNX Runtime installed on the system (already required by other modules)
- Model files present at `data/smollm3/model_int8.onnx` (+ `model_int8.onnx_data`) and `data/smollm3/tokenizer.json`

## Context for Implementer

- **Patterns to follow:** Follow `vad/silero.rs:30-44` for ONNX session wrapping and `vad/silero.rs:54-112` for `session.run()` usage (create `Value` tensors, call `session.run()`, extract results). Follow `asr/sherpa/sherpa.rs:135-158` for `tokio::task::spawn_blocking` with `Arc<Mutex<>>` pattern. **Note:** The `llama`, `phi3`, `qwen3`, and `smollm2` modules declared in `llm/mod.rs` do NOT exist yet — they are empty declarations. Do NOT reference them.
- **Conventions:** Module structure uses `mod.rs` + named file (e.g., `smollm3/mod.rs` + `smollm3/smollm3.rs`). Tests live in `tests/` subdirectory, wired via `#[cfg(test)] #[path = "tests/..."] mod ...;`. Integration tests requiring model files use `#[ignore]`.
- **Key files:**
  - `crates/onnx/src/session.rs` — `Session::run()`, `input_count()`, `input_name()`, `output_name()`, `input_shape()` APIs
  - `crates/onnx/src/value.rs` — `Value::from_slice()`, `Value::zeros()`, `Value::extract_tensor()`
  - `crates/inference/src/error.rs` — `InferError` enum, `Result<T>` type alias
  - `crates/inference/src/inference.rs` — `Inference` facade with `onnx_session()` helper
  - `crates/inference/src/vad/silero.rs` — Reference ONNX usage pattern (create tensors, run, extract, update state)
  - `crates/inference/src/asr/sherpa/sherpa.rs` — Reference async spawn_blocking pattern
- **Gotchas:**
  - **KV cache is required.** Despite `config.json` having `use_cache: false`, the ONNX export includes KV cache inputs/outputs. See `data/smollm3/README.md` lines 113-143 for the Python generation loop. Empty KV cache tensors have shape `[1, num_kv_heads, 0, head_dim]` (zero-length cache dim).
  - The model has external data (`model_int8.onnx_data`) alongside the main `.onnx` file. ONNX Runtime loads this automatically when co-located.
  - EOS token ID is 128012 (`<|im_end|>`). Generation should stop when EOS is produced.
  - `Session::run()` takes `&mut self`, so the session must be behind `Arc<Mutex<>>` for spawn_blocking.
  - Model config: `num_hidden_layers=36`, `num_key_value_heads=4`, `hidden_size=2048`, `num_attention_heads=16`, so `head_dim = 2048/16 = 128`.
  - **Model inputs** (3 + 72 = 75 total): `input_ids` (i64 `[1, seq_len]`), `attention_mask` (i64 `[1, total_len]`), `position_ids` (i64 `[1, seq_len]`), + `past_key_values.{0..35}.key` and `past_key_values.{0..35}.value` (f32 `[1, 4, cache_len, 128]`).
  - **Model outputs** (1 + 72 = 73 total): `logits` (f32 `[1, seq_len, 128256]`), + 72 present KV value tensors.
  - **Generation with KV cache:** First pass: `input_ids=[full_prompt]`, `position_ids=[0..N-1]`, empty KV cache → logits + KV cache. Subsequent passes: `input_ids=[new_token]`, `position_ids=[N]`, updated KV cache → logits + updated KV cache. Only pass the new single token, not the full sequence.
  - Input/output names should be discovered dynamically via `session.input_name(i)` / `session.output_name(i)` at construction time, not hardcoded.
- **Domain context:** SmolLM3 is a causal language model with GQA (grouped-query attention). Autoregressive generation with KV cache: the first pass computes attention for all prompt tokens and populates the cache. Each subsequent pass only computes attention for the new token, reusing cached keys/values for previous positions. This is O(n) per token instead of O(n²).

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Smollm3 module structure, struct, new() with I/O discovery, and module wiring
- [x] Task 2: Autoregressive generation with KV cache, forward() and recv()
- [x] Task 3: Integration test with real ONNX model

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Smollm3 module structure, struct, new() with I/O discovery, and module wiring

**Objective:** Create the `smollm3` module directory, define the `Smollm3` struct with constructor that dynamically discovers ONNX model I/O names, and wire it into the LLM module and Inference facade.

**Dependencies:** None

**Files:**

- Create: `crates/inference/src/llm/smollm3/mod.rs`
- Create: `crates/inference/src/llm/smollm3/smollm3.rs`
- Modify: `crates/inference/src/llm/mod.rs`
- Modify: `crates/inference/src/inference.rs`
- Create: `crates/inference/src/llm/smollm3/tests/smollm3_test.rs`

**Key Decisions / Notes:**

- `Smollm3` struct holds:
  - `session: Arc<Mutex<Session>>`
  - `tokenizer: Arc<tokenizers::Tokenizer>` (Arc because Tokenizer may not impl Clone; needed for spawn_blocking)
  - `eos_token_id: u32` (128012)
  - `max_tokens: usize` (default 512)
  - `num_layers: usize` (discovered from model I/O count)
  - `num_kv_heads: usize` (from config: 4)
  - `head_dim: usize` (from config: 128)
  - `input_names: Vec<String>` (all input names discovered from session)
  - `output_names: Vec<String>` (all output names discovered from session)
  - `rx: Option<tokio::sync::mpsc::Receiver<Result<String>>>`
- `new(session: Session, tokenizer_path: impl AsRef<Path>) -> Result<Self>`:
  1. Load tokenizer from file, wrap in `Arc`
  2. Discover all input names via `session.input_count()` + `session.input_name(i)`, store sorted
  3. Discover all output names via `session.output_count()` + `session.output_name(i)`, store sorted
  4. Derive `num_layers` from KV cache input count: `(input_count - 3) / 2` (3 non-KV inputs: input_ids, attention_mask, position_ids)
  5. Wrap session in `Arc<Mutex<>>`
- `max_tokens` defaults to 512, configurable via `with_max_tokens(n)` builder method
- Include stub `forward()` returning `Ok(())` and stub `recv()` returning `None` so the full API compiles
- Wire into `llm/mod.rs` as `pub(crate) mod smollm3; pub use smollm3::Smollm3;`
- Add `use_smollm3(model_path, tokenizer_path)` to `Inference` facade. This is ONNX-based: call `self.onnx_session(model_path)` to create the session, then `Smollm3::new(session, tokenizer_path)`. Follow `use_kokoro` pattern (`inference.rs:112-120`), NOT Candle-based patterns.

**Definition of Done:**

- [ ] `Smollm3` struct compiles with all fields
- [ ] `Smollm3::new()` loads tokenizer, discovers I/O names from session, wraps session in Arc<Mutex<>>
- [ ] `num_layers` is correctly derived from discovered input names
- [ ] `Smollm3` is re-exported from `llm` module
- [ ] `Inference::use_smollm3()` creates a Smollm3 instance via `self.onnx_session()` + `Smollm3::new()`
- [ ] Stub `forward()` and `recv()` methods compile
- [ ] Unit test: `Smollm3` is `Send`
- [ ] `cargo build -p inference` succeeds with no errors

**Verify:**

- `cargo build -p inference 2>&1 | tail -5` — compiles without errors
- `cargo test -p inference smollm3 -q 2>&1 | tail -5` — unit tests pass

### Task 2: Autoregressive generation with KV cache, forward() and recv()

**Objective:** Implement the `forward()` method that starts autoregressive text generation with KV-cached ONNX inference, and the async `recv()` method that streams generated tokens.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/inference/src/llm/smollm3/smollm3.rs`
- Modify: `crates/inference/src/llm/smollm3/tests/smollm3_test.rs`

**Key Decisions / Notes:**

- `forward(&mut self, text: &str) -> Result<()>`:
  1. Tokenize `text` using tokenizer → `Vec<u32>` token IDs
  2. Create `tokio::sync::mpsc::channel(32)` — sender to blocking task, receiver to `self.rx`
  3. Clone `Arc<Mutex<Session>>`, `Arc<Tokenizer>`, `eos_token_id`, `max_tokens`, input/output names, num_layers, num_kv_heads, head_dim
  4. `tokio::task::spawn_blocking` the generation closure:
     - **Initial state:** Build `input_ids: Vec<i64>` from tokenized IDs (full prompt), `attention_mask: Vec<i64>` (all 1s, length = prompt length), `position_ids: Vec<i64>` (0..prompt_len-1). Create 72 empty KV cache tensors: `Value::from_slice::<f32>(&[1, num_kv_heads, 0, head_dim], &[])` (zero-length cache dim)
     - **Loop (max_tokens iterations):**
       1. Create `input_ids` tensor `[1, seq_len]`, `attention_mask` tensor `[1, total_len]`, `position_ids` tensor `[1, seq_len]`
       2. Build inputs: `[("input_ids", &ids_val), ("attention_mask", &mask_val), ("position_ids", &pos_val)]` + KV cache inputs matched by discovered name
       3. Build output names from discovered names
       4. `session.lock().unwrap().run(&inputs, &output_names)` → outputs
       5. Extract logits from first output, argmax on last position → `next_token_id`
       6. Check `next_token_id == eos_token_id` → break
       7. Decode token: `tokenizer.decode(&[next_token_id as u32], true)` (skip_special_tokens=true since EOS is already checked before decoding)
       8. Send decoded text through channel. If `tx.send()` returns `Err` (receiver dropped, e.g., caller called forward() again), break the loop early
       9. Update for next iteration: `input_ids = vec![next_token_id]` (single token), extend `attention_mask` by 1, `position_ids = vec![current_pos + 1]`, update KV cache from outputs
     - On completion (EOS, max_tokens, or send error), sender drops → `recv()` returns `None`

- `recv(&mut self) -> Option<Result<String>>`:
  - `self.rx.as_mut()?.recv().await`
  - Returns `None` when generation is complete (channel closed)

- Argmax: iterate `logits[last_pos * vocab_size..(last_pos+1) * vocab_size]`, find index of max f32 value
- KV cache tensor management: After each `session.run()`, the output contains 72 present KV tensors. These become the inputs for the next iteration. Store them in a `Vec<Value>` indexed to match the discovered output names (excluding "logits").

**Definition of Done:**

- [ ] `forward()` tokenizes input and spawns generation task with KV cache
- [ ] First pass sends full prompt + empty KV cache; subsequent passes send single token + updated cache
- [ ] Generation loop produces tokens via greedy argmax decoding
- [ ] `recv()` returns generated tokens one at a time as `Option<Result<String>>`
- [ ] Generation stops at EOS token (128012) or max_tokens limit
- [ ] Channel drops cleanly when generation completes (success or error), causing recv() to return None
- [ ] If tx.send() fails (receiver dropped), generation loop breaks early
- [ ] `cargo build -p inference` succeeds with no errors

**Verify:**

- `cargo build -p inference 2>&1 | tail -5` — compiles without errors
- `cargo test -p inference smollm3 -q 2>&1 | tail -5` — unit tests pass

### Task 3: Integration test with real ONNX model

**Objective:** Write integration tests that load the real SmolLM3 ONNX model and tokenizer, verify model I/O, and run forward inference with streaming recv().

**Dependencies:** Task 2

**Files:**

- Modify: `crates/inference/src/llm/smollm3/tests/smollm3_test.rs`

**Key Decisions / Notes:**

- **Model I/O verification test** (`#[test] #[ignore]`): Load ONNX session, verify input count is 75 (3 base + 72 KV cache), verify output count is 73 (1 logits + 72 KV). Check that input names include `input_ids`, `attention_mask`, `position_ids`. Check output names include `logits`. If names differ, test fails with actual-vs-expected.
- **Forward/recv integration test** (`#[tokio::test] #[ignore]`):
  1. Create `Inference::cpu()`
  2. Call `use_smollm3("../../data/smollm3/model_int8.onnx", "../../data/smollm3/tokenizer.json")`
  3. Set `with_max_tokens(20)` to keep test fast
  4. Call `forward("Hello")`
  5. Loop `recv()` collecting tokens, print each token with `eprint!` for visibility, break on `None`
  6. Assert: at least 3 tokens were generated
  7. Assert: concatenated output is non-empty, >= 5 characters
  8. Assert: recv() returns None (channel cleanup)

**Definition of Done:**

- [ ] Model I/O verification test confirms expected input/output count and names
- [ ] Integration test loads real model and tokenizer successfully
- [ ] `forward("Hello")` followed by `recv()` loop produces at least 3 tokens with non-empty output >= 5 chars
- [ ] `recv()` returns `None` after generation completes, proving channel cleanup
- [ ] All tests pass when run with `--ignored` flag and model files present

**Verify:**

- `cargo test -p inference smollm3 -q 2>&1 | tail -5` — non-ignored tests pass
- `cargo test -p inference smollm3 -q -- --ignored 2>&1 | tail -20` — integration tests pass (requires model)

## Runtime Environment

- **Manual verification:** Run `cargo test -p inference test_smollm3 -- --ignored --nocapture` to see generated tokens printed to stderr. The integration test prints each token as it arrives.
- This is a library crate, not a service — no running process, port, or health check applies.

## Testing Strategy

- Unit tests: `Smollm3` is `Send`, struct construction compiles
- Integration tests (`#[ignore]`): Model I/O count and name verification, full forward/recv generation loop with real model and KV cache
- Manual verification: Run integration tests with `--ignored --nocapture` to see streaming output

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| ONNX model I/O names differ from expected | Low | High | All I/O names are discovered dynamically at construction time via `session.input_name(i)` / `session.output_name(i)`. No hardcoded names. Task 3 includes verification test asserting expected counts and key names. |
| KV cache tensor management is error-prone (72 tensors per step) | Med | High | KV cache inputs/outputs are matched by their discovered names. The generation loop stores present KV values in a `Vec<Value>` indexed consistently with output names. Task 3 integration test verifies multi-step generation works end-to-end. |
| Tokenizer decode produces unexpected output for individual tokens | Med | Low | EOS check happens BEFORE decoding (compare token_id == eos_token_id). Decode uses `skip_special_tokens=true` for clean output. If decode returns empty string for a non-EOS token, send `Err(InferError::TokenizerError)` through channel. |
| Generation task continues after caller drops receiver | Low | Low | Generation loop checks `tx.send()` result; if it returns `Err` (receiver dropped), breaks the loop immediately to avoid wasted compute. |

## Goal Verification

> Derived from the plan's goal using goal-backward methodology. The spec-reviewer-goal agent verifies these criteria during verification.

### Truths (what must be TRUE for the goal to be achieved)

- `Smollm3::new()` successfully creates an instance from an ONNX session and tokenizer path
- `new()` discovers all model I/O names dynamically from the session
- `forward(text)` starts autoregressive generation with KV cache from the given text
- `recv()` yields generated tokens one at a time asynchronously
- `recv()` returns `None` when generation is complete
- Generation stops at EOS token or max_tokens limit
- The module is accessible via `inference::llm::Smollm3` and `Inference::use_smollm3()`

### Artifacts (what must EXIST to support those truths)

- `crates/inference/src/llm/smollm3/mod.rs` — module root with re-exports and test wiring
- `crates/inference/src/llm/smollm3/smollm3.rs` — `Smollm3` struct with `new()`, `forward()`, `recv()`, KV cache management
- `crates/inference/src/llm/smollm3/tests/smollm3_test.rs` — unit + integration tests

### Key Links (critical connections that must be WIRED)

- `llm/mod.rs` declares `pub(crate) mod smollm3` and `pub use smollm3::Smollm3`
- `inference.rs` has `use_smollm3()` that creates a `Smollm3` via `self.onnx_session()` + `Smollm3::new()`
- `forward()` → `tokio::task::spawn_blocking` → `session.run()` with KV cache → `mpsc::Sender::send()` → `recv()` reads from `mpsc::Receiver`
- KV cache outputs from step N become KV cache inputs for step N+1

## Open Questions

- None — requirements are clear for this initial implementation.

### Deferred Ideas

- Chat template formatting (Jinja template in `data/smollm3/chat_template.jinja`)
- Sampling strategies (temperature=0.6, top_p=0.95 from generation_config.json)
- Sentence combining and further processing (mentioned by user as future work)
