# Kokoro TTS via ONNX in deli-infer Implementation Plan

Created: 2026-02-17
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
> **Worktree:** No — working directly on current branch

## Summary

**Goal:** Add ONNX Runtime support to `deli-infer` via the `ort` crate, extend the `Inference` struct with ONNX session creation capability, and implement a Kokoro TTS module that converts text to a `Tensor<i16>` of audio.

**Architecture:** The `Inference` struct gets a second member tracking the ONNX device configuration (CPU vs CUDA ordinal). A new `tts` module contains the `Kokoro` struct which owns an `ort::Session`, a vocab map, and voice style data loaded from an NPY file. Phonemization uses `espeak-ng` via direct C FFI (`libespeak-ng.so`) — we declare the few needed `extern "C"` functions inline (no bindgen, no sys crate). The espeak-ng data files are at `models/espeak-ng-data/`. The `speak` method is async via `spawn_blocking`, matching the pattern used by `SpeechRecognizer::transcribe`. Both phonemization (FFI, not async-safe) and ONNX inference (CPU-bound) run inside the same `spawn_blocking` closure.

**Tech Stack:** `ort` 2.0.0-rc.10 for ONNX inference, `espeak-ng` FFI for IPA phonemization (with clause terminator support), `ndarray` for tensor construction (ort 2.0's required API surface).

## Scope

### In Scope

- Add `ort` (and `ndarray`) dependency to `deli-infer`
- Add ONNX device member to `Inference`, configure in `cpu()` and `cuda()`
- Add `Inference::onnx_session(path)` helper to create ONNX sessions with the right execution providers
- Create `src/tts/` module with `Kokoro` struct
- `Kokoro::speak(&self, text: &str) -> Result<Tensor<i16>>` (async, via spawn_blocking)
- `Inference::use_kokoro(model_path, voice_path)` factory method
- NPY voice file loading with magic byte validation + header skip
- IPA phonemization via espeak-ng C FFI with clause terminator handling
- Vocab mapping (IPA characters → token IDs, from reference implementation)

### Out of Scope

- GPU-accelerated ONNX inference testing with real CUDA hardware (CUDA EP configuration is implemented and compiles correctly, but actual GPU inference requires CUDA hardware — CPU path is verified in tests)
- Streaming/chunked TTS output
- Other TTS models besides Kokoro
- Speed parameter (hardcoded to 1.0, matching reference default)

## Prerequisites

- `libespeak-ng.so` shared library installed (confirmed at `/usr/local/lib/libespeak-ng.so`)
- espeak-ng data files at `models/espeak-ng-data/` (confirmed, includes `en_dict`, `phondata`, `phonindex`, `phontab`, etc.)
- Model file at `models/kokoro/kokoro-v1.0.onnx` (confirmed, 326MB)
- Voice file at `models/kokoro/bf_emma.npy` (confirmed, 522KB, shape [510, 1, 256] f32)

## Context for Implementer

- **Patterns to follow:** The `SpeechRecognizer` in `crates/deli-infer/src/asr/recognizer.rs` is the closest analog — it's an async inference wrapper that uses `spawn_blocking` for CPU-bound work. Follow its structure for `Kokoro`. Both phonemization (FFI) and ONNX inference run inside the same `spawn_blocking` closure.
- **Conventions:** Module structure follows `asr/` pattern: `mod.rs` re-exports the public type, internal files are `pub(crate)`. Error handling uses `crate::error::{InferError, Result}`. Factory methods on `Inference` follow `use_<name>()` naming (`inference.rs:27-46`).
- **Key files:**
  - `crates/deli-infer/src/inference.rs` — `Inference` struct, `cpu()`, `cuda()`, `use_*()` methods
  - `crates/deli-infer/src/error.rs` — `InferError` enum, `Result` type alias
  - `crates/deli-infer/src/lib.rs` — module declarations and public re-exports
  - `crates/deli-infer/src/asr/recognizer.rs` — reference for async inference pattern
  - `~/droid/deep/src/onnx/kokoro.rs` — reference Kokoro implementation (**uses older ort 1.x API — NOT directly portable, see ort 2.0 API Translation below**)
- **Gotchas:**
  - **ort 2.0 API is NOT compatible with the reference.** The reference uses ort 1.x with `DynTensor`, `Allocator`, raw pointers, and unsafe slice operations. ort 2.0 uses `ndarray::Array` for all tensor I/O. See "ort 2.0 API Translation" section below for the exact mapping.
  - **NPY loading:** The reference skips via `align_to::<f32>()` then `[32..]` (128 bytes / 4 = 32 f32 words). We validate the magic bytes and parse via `chunks_exact(4).map(f32::from_le_bytes)` instead — safer and portable.
  - **Style indexing bounds:** The voice style is indexed as `style[256 * token_count .. 256 * token_count + 256]`. The voice file has 510 style vectors, so `token_count` must be clamped to 509 max. The reference does NO bounds checking and will panic on long text.
  - **ort::init():** Called once in `Inference::cpu()` and `Inference::cuda()` constructors via `std::sync::OnceLock`. NOT called in `onnx_session()` — avoids repeated calls and potential warnings from ort 2.0.
  - The vocab map has specific IPA character → i64 mappings (94 entries). These must match the reference exactly.
  - **espeak-ng FFI is NOT thread-safe.** The `espeak_Initialize` / `espeak_TextToPhonemesWithTerminator` / `espeak_Terminate` functions use global state. All calls must be serialized. Since our `speak()` runs inside `spawn_blocking`, and espeak is initialized once in `Kokoro::new`, concurrent `speak()` calls need a `Mutex` around the phonemization FFI calls.
  - **espeak-ng data path:** The reference uses `b"data/espeak-ng-data\0"` as the path argument to `espeak_Initialize`. We use `models/espeak-ng-data` (confirmed present). This path must be a null-terminated C string.
  - **espeak_TextToPhonemesWithTerminator loop:** The function processes one clause at a time and advances the text pointer. The reference only calls it once (only gets the first clause). We should loop until the text pointer is exhausted to handle multi-sentence input properly.
- **Domain context:** Kokoro TTS converts text → IPA phonemes (via espeak-ng FFI) → token IDs (via vocab) → ONNX inference (model takes tokens + style + speed tensors) → f32 audio samples → i16 PCM. The model outputs at 24kHz sample rate. The f32→i16 conversion is `(sample * 32768.0).clamp(-32768.0, 32767.0) as i16`.

### espeak-ng FFI Declaration

We declare only the functions and constants we need — no bindgen, no sys crate:

```rust
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

// Audio output mode
const ESPEAK_AUDIO_OUTPUT_SYNCHRONOUS: c_int = 2;
// Error codes
const ESPEAK_EE_OK: c_int = 0;
// Text encoding
const ESPEAK_CHARS_AUTO: c_int = 0;
// Phoneme mode
const ESPEAK_PHONEMES_IPA: c_int = 2;

// Clause terminator constants (from reference kokoro.rs:26-37)
const CLAUSE_INTONATION_FULL_STOP: i32 = 0x00000000;
const CLAUSE_INTONATION_COMMA: i32 = 0x00001000;
const CLAUSE_INTONATION_QUESTION: i32 = 0x00002000;
const CLAUSE_INTONATION_EXCLAMATION: i32 = 0x00003000;
const CLAUSE_TYPE_CLAUSE: i32 = 0x00040000;
const CLAUSE_TYPE_SENTENCE: i32 = 0x00080000;
const CLAUSE_PERIOD: i32 = 40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE;
const CLAUSE_COMMA: i32 = 20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE;
const CLAUSE_QUESTION: i32 = 40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE;
const CLAUSE_EXCLAMATION: i32 = 45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE;
const CLAUSE_COLON: i32 = 30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE;
const CLAUSE_SEMICOLON: i32 = 30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE;

#[link(name = "espeak-ng")]
extern "C" {
    fn espeak_Initialize(
        output: c_int,
        buflength: c_int,
        path: *const c_char,
        options: c_int,
    ) -> c_int;

    fn espeak_SetVoiceByName(name: *const c_char) -> c_int;

    fn espeak_TextToPhonemesWithTerminator(
        textptr: *mut *const c_void,
        textmode: c_int,
        phonememode: c_int,
        terminator: *mut c_int,
    ) -> *const c_char;

    fn espeak_Terminate() -> c_int;
}
```

### ort 2.0 API Translation

The reference implementation (`~/droid/deep/src/onnx/kokoro.rs`) uses ort 1.x. Here's the exact translation to ort 2.0:

| Reference (ort 1.x) | ort 2.0 Equivalent |
|---|---|
| `DynTensor` | Not needed — use `ndarray::Array` directly |
| `Allocator::new(&session, MemoryInfo::new(..))` | Not needed — sessions manage memory internally |
| `DynTensor::from_array(allocator, array)` | Pass `ndarray::CowArray` or `ArrayView` directly to `session.run()` |
| `session.run(vec![(Cow::from("tokens"), tokens_tensor)])` | `session.run(ort::inputs!["tokens" => tokens_array, "style" => style_array, "speed" => speed_array]?)` |
| `outputs[0].try_extract_tensor::<f32>()` | `outputs["audio"].try_extract_tensor::<f32>()` or index-based extraction |

**Input tensor construction (ort 2.0):**
```rust
use ndarray::{Array1, Array2};
let tokens = Array2::<i64>::from_shape_vec((1, ids.len()), /* padded token ids */)?;
let style = Array2::<f32>::from_shape_vec((1, 256), style_slice.to_vec())?;
let speed = Array1::<f32>::from_vec(vec![1.0]);
let outputs = session.run(ort::inputs!["tokens" => tokens, "style" => style, "speed" => speed]?)?;
let audio = outputs[0].try_extract_tensor::<f32>()?;
```

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add `ort` dependency and ONNX session support to `Inference`
- [x] Task 2: Create `tts` module with espeak-ng FFI phonemization and vocab
- [x] Task 3: Implement `Kokoro` struct with async `speak` method

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add `ort` dependency and ONNX session support to `Inference`

**Objective:** Add the `ort` crate to `deli-infer`, add an ONNX device member to the `Inference` struct, and provide a method to create ONNX sessions configured for the right device.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-infer/Cargo.toml`
- Modify: `crates/deli-infer/src/inference.rs`
- Modify: `crates/deli-infer/src/error.rs`
- Test: `crates/deli-infer/tests/inference_test.rs`

**Key Decisions / Notes:**

- Add `ort = "2.0.0-rc.10"` and `ndarray = "0.16"` to dependencies. Set `default-features = false` for ort, add feature `"ndarray"`. Add ort's `cuda` feature behind deli-infer's existing `cuda` feature flag.
- Add a private enum `OnnxDevice` with variants `Cpu` and `Cuda(usize)` to `inference.rs`.
- Add field `onnx_device: OnnxDevice` to `Inference`.
- In `cpu()`: set `onnx_device: OnnxDevice::Cpu`. Also call `ort::init().commit().ok()` via `std::sync::OnceLock` to ensure one-time global initialization.
- In `cuda(ordinal)`: set `onnx_device: OnnxDevice::Cuda(ordinal)`. Same OnceLock ort init.
- Use a module-level `static ORT_INIT: OnceLock<()>` and a helper `fn ensure_ort_init()` that calls `ORT_INIT.get_or_init(|| { ort::init().commit().ok(); })`. Call `ensure_ort_init()` in both `cpu()` and `cuda()`.
- Add method `pub fn onnx_session(&self, model_path: impl AsRef<Path>) -> Result<ort::session::Session>` that builds a session using `ort::Session::builder()?.commit_from_file(model_path)` with the right execution providers based on `self.onnx_device`. For CPU: `CPUExecutionProvider::default()`. For CUDA: `CUDAExecutionProvider::default().with_device_id(ordinal)` followed by CPU fallback.
- Add `InferError::Onnx(String)` variant and `From<ort::Error>` impl.

**Definition of Done:**

- [ ] `Inference::cpu()` creates an instance with `onnx_device == OnnxDevice::Cpu`, verified by test assertion
- [ ] `Inference::cuda(0)` creates an instance with `onnx_device == OnnxDevice::Cuda(0)`, verified by `cfg(feature = "cuda")` test
- [ ] `onnx_session("nonexistent.onnx")` returns `Err(InferError::Onnx(_))` containing "not found" or similar message
- [ ] `InferError::Onnx` variant exists and `Display` impl works (verified by `format!("{}", err)` in test)
- [ ] Calling `onnx_session()` multiple times does not panic (OnceLock ensures single ort init)
- [ ] All existing tests still pass (`cargo test -p deli-infer`)

**Verify:**

- `cargo test -p deli-infer -- inference_test -q` — inference tests pass
- `cargo check -p deli-infer` — no compile errors

### Task 2: Create `tts` module with espeak-ng FFI phonemization and vocab

**Objective:** Create the `tts/` module structure with IPA phonemization via espeak-ng C FFI (with clause terminator handling) and the Kokoro vocab mapping.

**Dependencies:** None (parallel with Task 1)

**Files:**

- Create: `crates/deli-infer/src/tts/mod.rs`
- Create: `crates/deli-infer/src/tts/phonemize.rs`
- Create: `crates/deli-infer/src/tts/vocab.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add `pub mod tts;`)
- Modify: `crates/deli-infer/Cargo.toml` (add `build.rs` or cargo link directive for `libespeak-ng`)
- Test: `crates/deli-infer/tests/tts_phonemize_test.rs`
- Test: `crates/deli-infer/tests/tts_vocab_test.rs`

**Key Decisions / Notes:**

- **Linking:** Add a `build.rs` to `deli-infer` that prints `cargo:rustc-link-lib=espeak-ng` (dynamic linking). The library is at `/usr/local/lib/libespeak-ng.so`.
- `phonemize.rs`: Contains the FFI declarations (see "espeak-ng FFI Declaration" in Context), espeak initialization, and phonemization logic.
  - `pub(crate) fn espeak_init(data_path: &str) -> Result<()>` — calls `espeak_Initialize(SYNCHRONOUS, 0, data_path_cstr, 0)`, then `espeak_SetVoiceByName("en-us")`. Returns `InferError::Runtime` on failure. Called once during `Kokoro::new`.
  - `pub(crate) fn phonemize(text: &str) -> Result<String>` — synchronous function (called inside `spawn_blocking`). Creates `CString` from text, then loops calling `espeak_TextToPhonemesWithTerminator` until text_ptr is exhausted. Each iteration: reads phonemes from returned `*const c_char` via `CStr::from_ptr` (NOT `CString::from_raw` — espeak owns the memory), appends to output string. After each call, checks terminator and appends appropriate punctuation (`.`, `?`, `!`, `, `, `: `, `; `) matching reference behavior (kokoro.rs:235-242). Loop ends when the returned phoneme string is empty or text_ptr reaches end.
  - **IMPORTANT:** Do NOT use `CString::from_raw` on the returned pointer — espeak-ng owns that memory (it's a static internal buffer). Use `CStr::from_ptr` instead to borrow it.
  - **Thread safety:** espeak-ng FFI uses global state. Wrap phonemize calls in a `static ESPEAK_MUTEX: Mutex<()>` to prevent concurrent access.
- `vocab.rs`: Contains `pub(crate) fn vocab() -> HashMap<char, i64>` returning the exact 94-entry mapping from the reference implementation (`~/droid/deep/src/onnx/kokoro.rs:79-194`). Using `HashMap<char, i64>` — avoids per-character String allocation during tokenization. Also contains `pub(crate) fn tokenize(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64>` that converts an IPA string to token IDs: `phonemes.chars().filter_map(|c| vocab.get(&c).copied()).collect()`.
- `mod.rs`: Re-exports the public `Kokoro` type (added in Task 3). For now, declares the submodules.

**Definition of Done:**

- [ ] `espeak_init("models/espeak-ng-data")` succeeds without error
- [ ] `phonemize("Hello world.")` returns a non-empty IPA string ending with `.` (clause terminator appended)
- [ ] `phonemize("How are you?")` returns IPA string ending with `?`
- [ ] `tokenize` converts known IPA characters to correct token IDs: `assert_eq!(vocab.get(&'a'), Some(&43))`, `assert_eq!(vocab.get(&' '), Some(&16))`
- [ ] `vocab()` returns exactly 94 entries: `assert_eq!(vocab.len(), 94)`
- [ ] Unknown characters are silently skipped (matching reference behavior)
- [ ] `cargo check -p deli-infer` links successfully against `libespeak-ng`

**Verify:**

- `cargo test -p deli-infer -- tts_phonemize -q` — phonemize tests pass
- `cargo test -p deli-infer -- tts_vocab -q` — vocab tests pass

### Task 3: Implement `Kokoro` struct with async `speak` method

**Objective:** Implement the `Kokoro` struct that loads the ONNX model and voice style, and provides `async fn speak(&self, text: &str) -> Result<Tensor<i16>>`.

**Dependencies:** Task 1, Task 2

**Files:**

- Create: `crates/deli-infer/src/tts/kokoro.rs`
- Modify: `crates/deli-infer/src/tts/mod.rs` (add `Kokoro` re-export)
- Modify: `crates/deli-infer/src/inference.rs` (add `use_kokoro` method)
- Modify: `crates/deli-infer/src/lib.rs` (add `Kokoro` to public exports)
- Test: `crates/deli-infer/tests/tts_kokoro_test.rs`

**Key Decisions / Notes:**

- `Kokoro` struct fields:
  - `session: ort::session::Session` — the loaded ONNX model
  - `vocab: HashMap<char, i64>` — IPA → token ID mapping
  - `style: Vec<f32>` — voice style data loaded from NPY (130560 f32s = 510 × 256)
- **NPY loading in `Kokoro::new`:**
  - Read all bytes from voice file
  - Validate `bytes[0..6] == b"\x93NUMPY"`. If not, return `InferError::Runtime("Invalid NPY file: missing \\x93NUMPY magic bytes")`
  - Skip 128-byte header (fixed for NPY format v1.0)
  - Parse remaining bytes as little-endian f32: `bytes[128..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>()`
  - Validate length == 130560 (510 × 256). If not, return `InferError::Runtime(format!("Voice file wrong size: expected 130560 f32s, got {}", len))`
- **Constructor `Kokoro::new(session, voice_path, espeak_data_path)`:**
  - Takes a pre-built `ort::session::Session` (created by `Inference::onnx_session`)
  - Loads NPY voice file (with validation as above)
  - Calls `espeak_init(espeak_data_path)` to initialize espeak-ng with the data directory
  - Builds vocab via `vocab::vocab()`
- **`speak(&self, text: &str) -> Result<Tensor<i16>>`:**
  - Async method. Entire pipeline runs in `spawn_blocking` (both FFI phonemization and ONNX inference are sync/CPU-bound).
  - Inner pipeline: `phonemize(text)` → `tokenize(phonemes, &vocab)` → pad token IDs with 0 at start and end → **clamp style index:** `let style_idx = padded_ids.len().min(509)` → extract `style[style_idx * 256 .. (style_idx + 1) * 256]` → build ndarray input tensors → `session.run(ort::inputs![...])` → extract f32 output → convert to i16 via `(sample * 32768.0).clamp(-32768.0, 32767.0) as i16` → wrap in `Tensor::new(vec![num_samples], samples)`
  - **Style index clamping:** Before indexing into style data, `let style_idx = padded_ids.len().min(509)`. This prevents panic when text produces more than 509 tokens.
  - **Input tensors (ort 2.0 ndarray API):**
    ```rust
    let tokens = Array2::<i64>::from_shape_vec((1, padded_ids.len()), padded_ids)?;
    let style = Array2::<f32>::from_shape_vec((1, 256), style_slice.to_vec())?;
    let speed = Array1::<f32>::from_vec(vec![1.0]);
    let outputs = session.run(ort::inputs!["tokens" => tokens, "style" => style, "speed" => speed]?)?;
    ```
  - **Output extraction (ort 2.0):** `outputs[0].try_extract_tensor::<f32>()` returns an `ArrayView`. Iterate to convert f32 → i16.
  - **Return type:** `Tensor::new(vec![num_samples], i16_samples)` — wraps `Vec<i16>` in deli-base `Tensor` with shape `[N]`.
- `Inference::use_kokoro(&self, model_path, voice_path, espeak_data_path) -> Result<Kokoro>`:
  - Calls `self.onnx_session(model_path)` then `Kokoro::new(session, voice_path, espeak_data_path)`
- `Kokoro` must be `Send + Sync` for use with `spawn_blocking`. `ort::Session` is `Send + Sync`. `HashMap` and `Vec<f32>` are `Send + Sync`. This works naturally (espeak mutex is module-level static, not a field).
- Integration test uses the real model at `models/kokoro/kokoro-v1.0.onnx` and voice at `models/kokoro/bf_emma.npy`. Mark with `#[ignore]` so CI doesn't need the model files — run with `cargo test -p deli-infer -- --ignored tts_kokoro`.

**Definition of Done:**

- [ ] `Kokoro::new()` successfully loads `models/kokoro/kokoro-v1.0.onnx` and `models/kokoro/bf_emma.npy` and inits espeak: verified by `assert!(result.is_ok())` in integration test
- [ ] `speak("Hello world")` returns `Ok(Tensor { shape: [N], data: Vec<i16> })` where N > 0: verified by `assert!(tensor.shape[0] > 0 && tensor.data.len() > 0)`
- [ ] f32→i16 conversion uses `.clamp(-32768.0, 32767.0)` before cast to prevent overflow
- [ ] Style index is clamped: `style_idx = token_count.min(509)` — code contains this check
- [ ] Loading corrupted NPY (wrong magic bytes) returns `InferError::Runtime` containing "Invalid NPY file"
- [ ] `Inference::use_kokoro()` returns `Ok(Kokoro)`: verified by integration test
- [ ] `Kokoro` is `Send + Sync`: verified by `fn assert_send_sync<T: Send + Sync>() {}; assert_send_sync::<Kokoro>();` in test
- [ ] Integration test writes audio to `/tmp/kokoro_output.raw` for manual verification

**Verify:**

- `cargo test -p deli-infer -- tts_kokoro -q` — unit tests pass (Send+Sync check, NPY validation)
- `cargo test -p deli-infer -- --ignored tts_kokoro -q` — integration test with real model passes
- `cargo check -p deli-infer` — clean build

## Runtime Environment

**Audio Output Verification:**

```bash
# Run integration test and save audio
cargo test -p deli-infer -- --ignored tts_kokoro --nocapture

# The integration test writes raw PCM to /tmp/kokoro_output.raw
# Play 24kHz mono i16 PCM:
ffplay -f s16le -ar 24000 -ac 1 /tmp/kokoro_output.raw
# or
aplay -f S16_LE -r 24000 -c 1 /tmp/kokoro_output.raw
```

**Expected:** Clear speech output "Hello world" in Emma voice at 24kHz.

## Testing Strategy

- **Unit tests:** Phonemization (espeak FFI output with clause terminators), vocab mapping (94 entries, spot-check key mappings), tokenization (IPA string → ID sequence, unknown char skipping), NPY validation (magic bytes, wrong size)
- **Integration tests:** Full `speak("Hello world")` pipeline with real model files (marked `#[ignore]` — run with `cargo test -p deli-infer -- --ignored tts_kokoro`)
- **Manual verification:** Play the output audio written to `/tmp/kokoro_output.raw` with ffplay/aplay (see Runtime Environment)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `libespeak-ng.so` not installed | Low | High | Build fails at link time with clear error. Document in Prerequisites that `libespeak-ng-dev` (or equivalent) must be installed. |
| espeak-ng FFI thread safety | Med | High | Use `static ESPEAK_MUTEX: Mutex<()>` in `phonemize.rs`. All FFI calls (init, phonemize) acquire this mutex. Concurrent `speak()` calls serialize on phonemization but run ONNX inference in parallel. |
| `ort` 2.0 API differs from reference | Med | High | Do NOT port reference code directly. Use ort 2.0 ndarray API as documented in "ort 2.0 API Translation" section above. Key: `ort::inputs![]` macro, `ndarray::Array` for tensors, no `DynTensor`/`Allocator`. |
| NPY file corrupted or wrong format | Low | Med | Validate magic bytes `\x93NUMPY` at `bytes[0..6]`. Validate data length == 130560 f32s after header skip. Return `InferError::Runtime("Invalid NPY file: ...")` with specific message on failure. |
| Style index OOB on long text (>509 tokens) | Med | High | Clamp before indexing: `let style_idx = padded_ids.len().min(509)`. This is in `speak()` before style slice extraction. Prevents panic — long text still infers but uses style vector 509 for the style selection. |
| `ort::init()` called multiple times | Low | Low | Use `std::sync::OnceLock` in `inference.rs`. `ensure_ort_init()` calls `ort::init().commit().ok()` exactly once. No repeated init warnings or panics. |
| espeak-ng data path not found | Low | High | `espeak_init()` returns error code if data path is invalid. Map to `InferError::Runtime("espeak-ng initialization failed — check that espeak-ng-data exists at {path}")`. |

## Open Questions

- None — task is well-specified by user requirements and reference implementation.
