# Replace ort with Custom ONNX Runtime FFI Wrapper

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

**Goal:** Replace the `ort` crate (and its transitive `ort-sys` dependency) with a thin, custom FFI wrapper crate (`crates/onnx`) that links directly against the system-installed `libonnxruntime.so` at compile time, eliminating build script incompatibilities on Jetson.

**Architecture:** A new `crates/onnx` crate provides raw FFI bindings to the onnxruntime C API vtable (`OrtApi`) plus a minimal safe wrapper (Env, Session, Value, Error). The `inference` crate replaces all `ort` usage with `onnx`. Linking is handled by a build.rs that finds the system library via `ONNXRUNTIME_DIR` env var, pkg-config, or standard system paths.

**Tech Stack:** Rust FFI (`extern "C"`), onnxruntime C API (vtable accessed via `OrtGetApiBase`), compile-time dynamic linking (`-l onnxruntime`).

## Scope

### In Scope

- New `crates/onnx` crate: build.rs, raw FFI types, safe wrapper (Env, Session, Value, Error)
- CUDA execution provider support behind `cuda` feature flag
- Refactor `crates/inference`: replace all `ort` imports/usage with `onnx`
- Update workspace member detection (automatic via `crates/*` glob)

### Out of Scope

- Changes to `candle-*` based models (Whisper, Qwen3, PoseDetector) — they don't use ort
- TensorRT execution provider (only CPU + CUDA needed)
- `ndarray` interop (current code doesn't use it for I/O, only ort's own Tensor type)
- Changing the onnxruntime version installed on the Jetson
- Static linking against `libonnxruntime.a` (Jetson provides `.so` only)

## Prerequisites

- System-installed `libonnxruntime.so` (with headers or known API version)
- On Jetson: onnxruntime from JetPack or built from source (typically in `/usr/lib` or `/usr/local/lib`)
- On dev machine: onnxruntime installed to a discoverable location, or `ONNXRUNTIME_DIR` env var set
- For CUDA feature: onnxruntime built with CUDA support

## Context for Implementer

- **Patterns to follow:** The crate structure follows other `crates/*` members. See `crates/base/` for a minimal crate example.
- **Conventions:** Edition 2024, compact use statements, `base::log_*` macros (not `log` crate). Error types use `From` impls for ergonomic `?` propagation.
- **Key files the implementer must read:**
  - `crates/inference/src/inference.rs` — Session creation, `onnx_session()` method, `ensure_ort_init()`
  - `crates/inference/src/tts/kokoro.rs` — Tensor creation (`OrtTensor::from_array`), session run (`ort::inputs!`), output extraction (`try_extract_tensor::<f32>()`)
  - `crates/inference/src/error.rs` — `From<ort::Error>` impl
- **Gotchas:**
  - The `OrtApi` struct is a C vtable (sequence of function pointers). Functions are accessed by their positional index. The order is defined in `onnxruntime_c_api.h` and is ABI-stable across versions (new functions appended, never reordered).
  - CUDA EP uses a standalone C function (`OrtSessionOptionsAppendExecutionProvider_CUDA`) not part of the OrtApi vtable — it's linked directly from `libonnxruntime.so`.
  - `kokoro.rs` holds `Session` in `Arc<Mutex<Session>>` — the safe wrapper's `Session` must be `Send`.
  - The `ort::inputs!` macro creates named input bindings. Our replacement uses a `Vec<(&str, &Value)>` or similar.
- **Domain context:** The OrtApi is obtained via `OrtGetApiBase() → GetApi(version)`. All operations go through function pointers in this vtable. Opaque types (`OrtEnv`, `OrtSession`, `OrtValue`, etc.) are heap-allocated by the C library and freed via corresponding `Release*` functions.

## Feature Inventory

### Files Being Replaced/Modified

| Old File | Functions/APIs Used | Mapped to Task |
|----------|-------------------|----------------|
| `crates/inference/Cargo.toml` | `ort` dependency + `cuda` feature | Task 5 |
| `crates/inference/src/inference.rs` | `ort::init().commit()`, `ort::session::Session::builder()`, `ort::execution_providers::{CPU,CUDA}ExecutionProvider` | Task 5 |
| `crates/inference/src/tts/kokoro.rs` | `ort::session::Session`, `ort::value::Tensor::from_array()`, `ort::inputs!`, `try_extract_tensor::<f32>()` | Task 5 |
| `crates/inference/src/error.rs` | `From<ort::Error> for InferError` | Task 5 |

### Feature Mapping Verification

- [x] All old files listed above
- [x] All functions/APIs identified
- [x] Every feature has a task number
- [x] No features accidentally omitted

## Progress Tracking

- [x] Task 1: Create `crates/onnx` crate with build script and FFI bindings
- [x] Task 2: Implement safe wrapper — Error, Env, API initialization
- [x] Task 3: Implement safe wrapper — Session and SessionBuilder
- [x] Task 4: Implement safe wrapper — Value (tensor I/O) and Session::run
- [x] Task 5: Refactor `inference` crate to use `onnx` instead of `ort`

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create `crates/onnx` crate with build script and FFI bindings

**Objective:** Create the new crate with a build.rs that discovers and links the system onnxruntime, plus raw FFI type definitions for the onnxruntime C API.

**Dependencies:** None

**Files:**

- Create: `crates/onnx/Cargo.toml`
- Create: `crates/onnx/build.rs`
- Create: `crates/onnx/src/ffi.rs`
- Create: `crates/onnx/src/lib.rs` (stub with `mod ffi;`)

**Key Decisions / Notes:**

- `build.rs` search order: `ONNXRUNTIME_DIR` env → `ONNXRUNTIME_LIB_DIR` env → pkg-config → common system paths (`/usr/lib`, `/usr/local/lib`, `/usr/lib/aarch64-linux-gnu`)
- Emits `cargo:rustc-link-lib=onnxruntime` and `cargo:rustc-link-search=native=<path>`
- The `OrtApi` is represented as a raw pointer to a vtable array: `struct Api(*const *const ())`. Functions are accessed by known index and transmuted to the correct signature. This avoids defining the full 280+ field struct while remaining correct (all fields are pointer-sized function pointers in `#[repr(C)]`).
- Define opaque types: `OrtEnv`, `OrtSession`, `OrtSessionOptions`, `OrtValue`, `OrtStatus`, `OrtMemoryInfo`, `OrtAllocator`, `OrtRunOptions`, `OrtTensorTypeAndShapeInfo`
- Define enums: `OrtLoggingLevel`, `ONNXTensorElementDataType`, `OrtErrorCode`, `GraphOptimizationLevel`
- Single extern function: `OrtGetApiBase() -> *const OrtApiBase`
- CUDA EP function behind `#[cfg(feature = "cuda")]`: `OrtSessionOptionsAppendExecutionProvider_CUDA(options, device_id) -> *mut OrtStatus`
- OrtApi function indices must match onnxruntime_c_api.h (verified against v1.17.0): CreateStatus=0, GetErrorCode=1, GetErrorMessage=2, CreateEnv=3, CreateSession=7, Run=9, CreateSessionOptions=10, SetSessionGraphOptimizationLevel=23, SetIntraOpNumThreads=24, SessionGetInputCount=30, SessionGetOutputCount=31, SessionGetInputName=36, SessionGetOutputName=37, and the Release*/tensor functions at higher indices (to be counted from the full header during implementation)
- Cargo.toml: edition 2024, no external dependencies (only `libc` for `c_char`/`c_int` if needed, or use `std::ffi`)

**Definition of Done:**

- [ ] `cargo check -p onnx` succeeds when `ONNXRUNTIME_DIR` is set to a valid onnxruntime installation
- [ ] Build script prints `cargo:rustc-link-lib=onnxruntime` and a valid search path
- [ ] FFI types compile and opaque types are correctly repr(C)
- [ ] No unsafe code outside of `ffi.rs`
- [ ] All unit tests pass

**Verify:**

- `ONNXRUNTIME_DIR=/usr/local cargo check -p onnx` — crate compiles
- `cargo test -p onnx -- --nocapture` — unit tests pass

### Task 2: Implement safe wrapper — Error, Env, API initialization

**Objective:** Implement the error type that wraps OrtStatus, the global API/Env initialization, and the public init function.

**Dependencies:** Task 1

**Files:**

- Create: `crates/onnx/src/error.rs`
- Modify: `crates/onnx/src/lib.rs`

**Key Decisions / Notes:**

- `OnnxError` stores an error code (`OrtErrorCode`) and message (`String`), extracted from `OrtStatus*` via the OrtApi vtable (GetErrorCode, GetErrorMessage), then the status is released (ReleaseStatus).
- Helper function `check_status(api, status_ptr)` converts null → Ok, non-null → Err(OnnxError).
- Global initialization: `init()` calls `OrtGetApiBase() → GetApi(ORT_API_VERSION)`, then `CreateEnv(WARNING, "onnx")`. Stores the Api pointer and OrtEnv pointer in `static OnceLock`. Returns `&'static Api` handle used by Session/Value APIs.
- `ORT_API_VERSION` constant set to 17 (matching onnxruntime 1.17, common on Jetson). Will negotiate downward if the runtime is older — `GetApi` returns null for unsupported versions, which we detect and report.
- `Drop` for Env calls `ReleaseEnv` — but since it's in a `OnceLock`, it lives for `'static` (which is fine, onnxruntime expects a single long-lived env).

**Definition of Done:**

- [ ] `OnnxError` implements `Display`, `Debug`, `std::error::Error`
- [ ] `init()` returns a handle usable for session creation
- [ ] Double-calling `init()` is safe (OnceLock)
- [ ] All unit tests pass

**Verify:**

- `cargo test -p onnx` — all tests pass

### Task 3: Implement safe wrapper — Session and SessionBuilder

**Objective:** Implement SessionBuilder for configuring and creating ONNX sessions, with CPU and CUDA execution provider support.

**Dependencies:** Task 2

**Files:**

- Create: `crates/onnx/src/session.rs`
- Modify: `crates/onnx/src/lib.rs` (add `mod session; pub use session::*;`)

**Key Decisions / Notes:**

- `SessionBuilder` wraps `*mut OrtSessionOptions`. Created via `CreateSessionOptions`. Has methods: `with_cpu()`, `with_cuda(device_id)` (behind `#[cfg(feature = "cuda")]`), `with_optimization_level()`, `with_intra_threads()`.
- `commit_from_file(path) -> Result<Session>` calls `CreateSession(env, path, options)` and releases options.
- `Session` wraps `*mut OrtSession`. Implements `Drop` calling `ReleaseSession`.
- `Session` must be `Send` (used in `Arc<Mutex<Session>>` by kokoro). `OrtSession` is thread-safe for `Run` with external synchronization, which the `Mutex` provides.
- The `session_builder()` function on the init handle (or a free function) creates a `SessionBuilder`.
- `SessionBuilder` `Drop` calls `ReleaseSessionOptions` to handle cleanup on error paths.

**Definition of Done:**

- [ ] `Session` is `Send`
- [ ] `SessionBuilder` supports CPU and CUDA (feature-gated) execution providers
- [ ] Creating a session from a non-existent file returns an appropriate error
- [ ] All unit tests pass

**Verify:**

- `cargo test -p onnx` — all tests pass

### Task 4: Implement safe wrapper — Value (tensor I/O) and Session::run

**Objective:** Implement tensor creation from typed data, session execution with named inputs, and output tensor data extraction.

**Dependencies:** Task 3

**Files:**

- Create: `crates/onnx/src/value.rs`
- Modify: `crates/onnx/src/session.rs` (add `run` method)
- Modify: `crates/onnx/src/lib.rs` (add `mod value; pub use value::*;`)

**Key Decisions / Notes:**

- `Value` wraps `*mut OrtValue`. `Drop` calls `ReleaseValue`.
- `Value::from_slice::<T>(shape: &[usize], data: &[T]) -> Result<Value>` — creates an input tensor using `CreateTensorWithDataAsOrtValue`. The data must outlive the Value, OR we copy into an owned buffer. For our use case (kokoro creates boxed slices), we'll own the data: `Value` holds a `Box<[u8]>` backing buffer alongside the `OrtValue*`. This ensures the data lives as long as the Value.
- Type mapping: `f32 → FLOAT`, `i64 → INT64`, `f64 → DOUBLE`, etc. Use a sealed `TensorElement` trait.
- `Value::extract_tensor::<T>() -> Result<&[T]>` — calls `GetTensorMutableData` and returns a slice. Validates element type matches `T` via `GetTensorTypeAndShape` + `GetTensorElementType`.
- `Session::run(inputs: &[(&str, &Value)], output_names: &[&str]) -> Result<Vec<Value>>` — allocates output `OrtValue*` slots as null, calls `OrtApi::Run`, wraps outputs. Converts input/output names to CStrings. Outputs are owned by the caller (wrapped in `Value` with `Drop`).
- `Value` must be `Send` (held across await points in kokoro).

**Definition of Done:**

- [ ] `Value::from_slice::<f32>` and `Value::from_slice::<i64>` create valid tensors
- [ ] `Session::run` executes a model and returns output values
- [ ] `Value::extract_tensor::<f32>` returns correct output data
- [ ] `Value` is `Send`
- [ ] All unit tests pass

**Verify:**

- `cargo test -p onnx` — all tests pass

### Task 5: Refactor `inference` crate to use `onnx` instead of `ort`

**Objective:** Replace all `ort` usage in the inference crate with the new `onnx` crate. Remove the `ort` dependency entirely.

**Dependencies:** Task 4

**Files:**

- Modify: `crates/inference/Cargo.toml`
- Modify: `crates/inference/src/inference.rs`
- Modify: `crates/inference/src/tts/kokoro.rs`
- Modify: `crates/inference/src/error.rs`

**Key Decisions / Notes:**

- **Cargo.toml:** Remove `ort` dependency. Add `onnx = { path = "../onnx" }`. Update `cuda` feature: replace `"ort/cuda"` with `"onnx/cuda"`. Remove `tls-native` and `download-binaries` (no longer needed).
- **inference.rs:**
  - Replace `ort::init().commit()` with `onnx::init()`
  - Replace `OrtSession::builder()?.with_execution_providers([...]).commit_from_file(path)` with `onnx::session_builder().with_cpu().commit_from_file(path)` (and `.with_cuda(ordinal)` for CUDA)
  - The `onnx_session()` method return type changes from `ort::session::Session` to `onnx::Session`
  - `ensure_ort_init()` becomes `ensure_onnx_init()` calling `onnx::init()`
- **kokoro.rs:**
  - Replace `ort::session::Session` with `onnx::Session`
  - Replace `ort::value::Tensor::from_array((shape, data))` with `onnx::Value::from_slice::<i64>(&shape, &data)` (for tokens) and `from_slice::<f32>` (for style/speed)
  - Replace `ort::inputs![...]` with a `Vec<(&str, &Value)>` passed to `session.run(inputs, &["audio"])`
  - Replace `outputs[0].try_extract_tensor::<f32>()` with `outputs[0].extract_tensor::<f32>()`
  - The `(_shape, output_data)` destructure changes to just `output_data` (a `&[f32]`)
- **error.rs:** Replace `From<ort::Error>` with `From<onnx::OnnxError>`

**Definition of Done:**

- [ ] No remaining imports from `ort` anywhere in the `inference` crate
- [ ] `cargo check -p inference` compiles without errors
- [ ] `cargo check -p inference --features cuda` compiles without errors
- [ ] Existing unit tests pass: `cargo test -p inference`
- [ ] The `test_kokoro_is_send` and `test_implements_sink_and_stream` tests pass
- [ ] No diagnostics errors

**Verify:**

- `cargo test -p inference` — all non-ignored tests pass
- `cargo check -p inference --features cuda` — CUDA path compiles

## Testing Strategy

- **Unit tests:** Error formatting, type assertions (Send, Sink, Stream), build script logic. These don't need onnxruntime installed.
- **Integration tests:** Existing inference tests (voice style loading, type assertions) should pass unchanged. The `#[ignore]` integration tests (requiring model files + GPU) verify end-to-end on Jetson.
- **Manual verification:** Build on Jetson with `ONNXRUNTIME_DIR` pointing to system install. Run `test_kokoro_integration` with model files.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OrtApi vtable index mismatch | Med | High | Verify indices against onnxruntime_c_api.h v1.17.0 header during implementation. Add a smoke test that calls `GetVersionString` to confirm the vtable is correct. |
| Jetson onnxruntime version too old for API v17 | Low | High | `GetApi(17)` returns null if unsupported. Detect this and return a clear error message with the installed version string. |
| Data lifetime issue with Value backing buffers | Med | High | Value owns its data buffer. The OrtValue references this buffer. Drop order: release OrtValue first, then drop buffer. Enforce via struct field ordering (Rust drops fields in declaration order). |
| CUDA EP function not exported from non-CUDA builds | Low | Med | CUDA EP is behind `cfg(feature = "cuda")`. If the feature is enabled but the library lacks CUDA, the linker will fail at compile time with a clear error about the missing symbol. |
| Memory leak if Session::run fails mid-execution | Low | Med | Output OrtValue pointers are initialized to null. On error, only non-null outputs need releasing. The cleanup loop handles this. |

## Goal Verification

### Truths (what must be TRUE for the goal to be achieved)

- The `ort` and `ort-sys` crates are no longer dependencies of the workspace
- The `inference` crate compiles and its tests pass using the new `onnx` crate
- Building on Jetson succeeds when `ONNXRUNTIME_DIR` points to the system onnxruntime
- CUDA execution provider is available behind the `cuda` feature flag
- Kokoro TTS synthesis produces audio output (same behavior as before the migration)

### Artifacts (what must EXIST to support those truths)

- `crates/onnx/Cargo.toml` — crate manifest with `cuda` feature
- `crates/onnx/build.rs` — finds and links system onnxruntime
- `crates/onnx/src/ffi.rs` — raw OrtApi vtable access, opaque types, enums
- `crates/onnx/src/error.rs` — `OnnxError` type wrapping OrtStatus
- `crates/onnx/src/lib.rs` — `init()`, module re-exports
- `crates/onnx/src/session.rs` — `SessionBuilder`, `Session`
- `crates/onnx/src/value.rs` — `Value` (tensor creation + extraction)
- Modified `crates/inference/Cargo.toml` — `onnx` dep instead of `ort`
- Modified `crates/inference/src/inference.rs` — uses `onnx::*`
- Modified `crates/inference/src/tts/kokoro.rs` — uses `onnx::*`
- Modified `crates/inference/src/error.rs` — `From<onnx::OnnxError>`

### Key Links (critical connections that must be WIRED)

- `onnx::init()` → `OrtGetApiBase() → GetApi()` → stores Api + Env globally
- `onnx::session_builder().commit_from_file(path)` → `OrtApi::CreateSession` → returns `onnx::Session`
- `onnx::Value::from_slice()` → `OrtApi::CreateTensorWithDataAsOrtValue` → tensor usable as input
- `Session::run(inputs, outputs)` → `OrtApi::Run` → `Vec<Value>` outputs
- `Value::extract_tensor::<f32>()` → `OrtApi::GetTensorMutableData` → `&[f32]` slice
- `inference::Inference::onnx_session()` → `onnx::session_builder()...commit_from_file()`
- `kokoro.rs` synthesis → `session.run([tokens, style, speed], ["audio"])` → f32 samples → i16 PCM

## Open Questions

- Exact onnxruntime version installed on the target Jetson (affects `ORT_API_VERSION` choice). Plan defaults to v17 (onnxruntime 1.17) which is widely available on JetPack 5.x+.
- Whether `ONNXRUNTIME_DIR` is already set in the Jetson build environment or needs to be configured.
