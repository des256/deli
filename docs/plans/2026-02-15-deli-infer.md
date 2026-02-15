# deli-infer: Multi-Backend Deep Learning Inference Crate

Created: 2026-02-15
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: Yes

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

**Goal:** Add a `deli-infer` crate providing a pluggable backend system for deep learning model inference, starting with an ONNX backend via the `ort` crate. Also add a `Tensor<T>` type to `deli-math` as the shared tensor representation.

**Architecture:** Trait-based backend abstraction (`Backend` + `Session` traits) with compile-time feature flags controlling which backends are compiled in, and a `BackendRegistry` providing runtime selection via `registry.get("onnx")` when multiple backends are available. The ONNX backend wraps the `ort` crate, converting between `deli-math::Tensor<T>` and `ort`'s ndarray-based tensors internally. A `Device` enum abstracts hardware selection (CPU, CUDA, TensorRT) across backends.

**Tech Stack:** Rust (edition 2024), `ort` 2.0 for ONNX Runtime, `ndarray` 0.17 (internal to ONNX backend only)

## Scope

### In Scope

- `Tensor<T>` type in `deli-math` (shape + flat data, with convenience constructors)
- `deli-infer` crate with `Backend` and `Session` traits
- `BackendRegistry` for runtime backend selection among compiled-in backends
- `Device` enum for hardware abstraction (CPU, CUDA, TensorRT)
- ONNX backend behind `onnx` feature flag
- Conversion between `Tensor<T>` and ndarray (internal to ONNX backend)
- Error types for inference failures
- Integration tests with a small ONNX model

### Out of Scope

- High-level model APIs (speech recognition, face recognition) — future work on top of this
- Non-ONNX backends (TensorFlow, PyTorch) — future backends plug into the same traits
- Model downloading/caching
- Async inference
- Training

## Platform Support

| Platform | Architecture | ONNX Runtime | Notes |
|----------|-------------|-------------|-------|
| Linux x86-64 | x86_64-unknown-linux-gnu | Pre-built binaries via `ort` | Primary dev platform |
| Linux arm64 | aarch64-unknown-linux-gnu | Pre-built binaries via `ort` | Raspberry Pi 4/5, NVIDIA Jetson |
| macOS arm64 | aarch64-apple-darwin | Pre-built binaries via `ort` | Apple Silicon dev machines |

The `ort` crate automatically downloads pre-built ONNX Runtime binaries for these targets. For platforms without pre-built binaries, users can set `ORT_LIB_LOCATION` to point to a system-installed ONNX Runtime.

## Prerequisites

- ONNX Runtime libraries available (ort crate downloads them automatically for supported platforms)
- A small ONNX test model for integration tests (we'll generate one or use a minimal one)
- For arm64 cross-compilation testing: `rustup target add aarch64-unknown-linux-gnu`

## Context for Implementer

- **Patterns to follow:** The workspace uses `crates/*` with `edition = "2024"`. See `crates/deli-math/Cargo.toml` for the minimal crate setup pattern. Integration tests go in `tests/` directories (e.g., `crates/deli-math/tests/vec3_tests.rs`).
- **Conventions:** Types are simple structs with public fields. No builder patterns in existing code. Errors are enums implementing `Display` + `Error`. No external dependencies in existing crates except the derive macro.
- **Key files:**
  - `Cargo.toml` (workspace root) — members = `["crates/*"]`, resolver = "2"
  - `crates/deli-math/src/lib.rs` — pub mod + pub use pattern for types
  - `crates/deli-codec/src/lib.rs` — error enum pattern with Display impl
- **Gotchas:** The workspace uses edition 2024 (Rust 1.93). The `ort` crate requires `ndarray` 0.17+ to avoid version conflicts.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add Tensor type to deli-math
- [x] Task 2: Create deli-infer crate with core traits, registry, and error types
- [x] Task 3: Implement ONNX backend
- [x] Task 4: Integration test with real ONNX model

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add Tensor type to deli-math

**Objective:** Add a generic `Tensor<T>` struct to `deli-math` that stores shape and flat data, usable as the common tensor representation across all inference backends.

**Dependencies:** None

**Files:**

- Create: `crates/deli-math/src/tensor.rs`
- Modify: `crates/deli-math/src/lib.rs` (add `pub mod tensor; pub use tensor::Tensor;`)
- Create: `crates/deli-math/tests/tensor_tests.rs`

**Key Decisions / Notes:**

- `Tensor<T>` has two public fields: `shape: Vec<usize>` and `data: Vec<T>`
- Provide `Tensor::new(shape, data)` constructor that validates shape vs data length using `checked_mul` to detect overflow; returns `TensorError::ShapeOverflow` on overflow, `TensorError::ShapeMismatch` on length mismatch
- Provide `Tensor::zeros(shape)` where `T: Default + Clone`
- Provide `Tensor::from_scalar(value)` for 0-d tensors
- Implement `Debug`, `Clone`, `PartialEq`
- Add `ndim()`, `len()`, `is_empty()`, `get(indices)` helper methods
- `new()` returns `Result<Tensor<T>, TensorError>` when shape doesn't match data length
- Note: Tensor owns its data. Model weights are stored internally by ort, not in Tensor. Tensor is used for inference inputs/outputs which are typically small-to-medium size

**Definition of Done:**

- [ ] `Tensor::new()` validates shape product using `checked_mul`, returns error on overflow or mismatch
- [ ] `Tensor::zeros()` creates a zero-filled tensor of given shape
- [ ] `ndim()`, `len()`, `is_empty()` return correct values
- [ ] Overflow test: `Tensor::new(vec![usize::MAX, 2], vec![])` returns `TensorError::ShapeOverflow`
- [ ] All tests pass with `cargo test -p deli-math`
- [ ] No compiler warnings

**Verify:**

- `cargo test -p deli-math -- tensor -q` — tensor tests pass
- `cargo check -p deli-math` — no warnings

---

### Task 2: Create deli-infer crate with core traits, registry, and error types

**Objective:** Create the `deli-infer` crate with the core `Backend` and `Session` traits, `BackendRegistry` for runtime backend selection, `Device` enum, and error types. No backend implementations yet — just the public API surface.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-infer/Cargo.toml`
- Create: `crates/deli-infer/src/lib.rs`
- Create: `crates/deli-infer/src/device.rs`
- Create: `crates/deli-infer/src/error.rs`
- Create: `crates/deli-infer/src/backend.rs`
- Create: `crates/deli-infer/src/registry.rs`
- Create: `crates/deli-infer/tests/api_tests.rs`

**Key Decisions / Notes:**

- `Device` enum: `Cpu`, `Cuda { device_id: i32 }`, `TensorRt { device_id: i32, fp16: bool }`
- `Backend` trait:
  ```
  pub trait Backend {
      fn name(&self) -> &str;
      fn load_model(&self, model: ModelSource, device: Device)
          -> Result<Box<dyn Session>, InferError>;
  }
  ```
- `Session` trait (uses `&mut self` because ort caches internal allocations during inference):
  ```
  pub trait Session {
      fn run(&mut self, inputs: &[(&str, Tensor<f32>)])
          -> Result<HashMap<String, Tensor<f32>>, InferError>;
      fn input_names(&self) -> &[String];
      fn output_names(&self) -> &[String];
  }
  ```
- `BackendRegistry` for runtime backend selection:
  ```
  pub struct BackendRegistry { backends: HashMap<String, Box<dyn Backend>> }
  impl BackendRegistry {
      fn new() -> Self;
      fn register(&mut self, backend: Box<dyn Backend>);
      fn get(&self, name: &str) -> Option<&dyn Backend>;
      fn list(&self) -> Vec<&str>;
  }
  ```
- Convenience function: `pub fn create_registry() -> BackendRegistry` that auto-registers all backends enabled via feature flags (ONNX when `onnx` feature is on)
- `ModelSource` enum: `File(PathBuf)`, `Memory(Vec<u8>)` — how to load the model
- `InferError` enum: `BackendError(String)`, `ShapeMismatch { expected, got }`, `UnsupportedDevice(Device)`, `ModelLoad(String)`, `UnsupportedDtype(String)` (for non-f32 model outputs), `InvalidInput { name, expected_names }` (for wrong input names)
- Cargo.toml: depend on `deli-math`, define `onnx` feature (empty for now, wired in Task 3)
- Tests: verify Device Display/Debug, error Display, ModelSource construction, BackendRegistry register/get/list with a mock backend

**Definition of Done:**

- [ ] `Backend` and `Session` traits are defined and public
- [ ] `BackendRegistry` supports register, get, and list operations
- [ ] `create_registry()` returns an empty registry (no backends yet)
- [ ] `Device` enum has `Cpu`, `Cuda`, `TensorRt` variants
- [ ] `InferError` implements `Display` and `Error`
- [ ] `ModelSource` has `File` and `Memory` variants
- [ ] `cargo check -p deli-infer` compiles with no warnings
- [ ] Unit tests pass for registry, error display, device construction

**Verify:**

- `cargo test -p deli-infer -q` — all tests pass
- `cargo check -p deli-infer` — no warnings

---

### Task 3: Implement ONNX backend

**Objective:** Implement the `Backend` and `Session` traits for ONNX Runtime via the `ort` crate, behind the `onnx` feature flag.

**Dependencies:** Task 2

**Files:**

- Create: `crates/deli-infer/src/onnx.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add `#[cfg(feature = "onnx")] pub mod onnx;`)
- Modify: `crates/deli-infer/Cargo.toml` (add `ort` and `ndarray` deps under `onnx` feature)
- Create: `crates/deli-infer/tests/onnx_tests.rs`

**Key Decisions / Notes:**

- `OnnxBackend` struct implements `Backend`
  - `name()` returns `"onnx"`
  - `load_model()` maps `Device` to ort execution providers: `Cpu` → default, `Cuda` → `ep::CUDA`, `TensorRt` → `ep::TensorRT`
  - Returns `OnnxSession` wrapped in `Box<dyn Session>`
- `OnnxSession` struct wraps `ort::session::Session`, implements `Session`
  - `run()` validates input names against `input_names()` before calling ort; returns `InferError::InvalidInput` with expected names if mismatched
  - Converts `Tensor<f32>` → `ndarray::ArrayD<f32>` → ort input, runs inference, converts ort output → `Tensor<f32>`
  - Conversion: `Tensor { shape, data }` → `ndarray::ArrayD::from_shape_vec(shape, data)`
  - Reverse: extract shape and flat vec from ndarray output; explicitly check output dtype is f32, return `InferError::UnsupportedDtype` if not
  - For CUDA/TensorRT: if EP initialization fails, return `InferError::UnsupportedDevice` with a helpful message instead of panicking
- Wire into `create_registry()`: when `onnx` feature is enabled, `create_registry()` registers `OnnxBackend` automatically
- Cargo.toml feature wiring:
  ```toml
  [features]
  onnx = ["dep:ort", "dep:ndarray"]

  [dependencies]
  ort = { version = "2", optional = true }
  ndarray = { version = "0.17", optional = true }
  ```
- For CUDA/TensorRT features, add optional pass-through features:
  ```toml
  cuda = ["onnx", "ort/cuda"]
  tensorrt = ["onnx", "ort/tensorrt"]
  ```
- Tests: use `#[cfg(feature = "onnx")]` guard. Create a minimal ONNX model (identity or add) using a Python script in `crates/deli-infer/tests/fixtures/`, or generate one at test time. Simplest approach: include a pre-built minimal .onnx file as a test fixture.

**Definition of Done:**

- [ ] `OnnxBackend` implements `Backend` trait
- [ ] `OnnxSession` implements `Session` trait
- [ ] `create_registry()` includes ONNX backend when `onnx` feature is enabled
- [ ] Tensor↔ndarray conversion works correctly (shape and data preserved)
- [ ] `Device::Cpu` creates a working session that runs inference
- [ ] Tests pass with `cargo test -p deli-infer --features onnx`
- [ ] Code compiles without `onnx` feature (`cargo check -p deli-infer`)

**Verify:**

- `cargo test -p deli-infer --features onnx -q` — ONNX tests pass
- `cargo check -p deli-infer` — compiles without onnx feature (no unconditional ort imports)

---

### Task 4: Integration test with real ONNX model

**Objective:** Add an end-to-end integration test that loads a real (small) ONNX model, runs inference through the full `deli-infer` API, and verifies output correctness.

**Dependencies:** Task 3

**Files:**

- Create: `crates/deli-infer/tests/fixtures/generate_test_model.py` (Python script to generate test .onnx file)
- Create: `crates/deli-infer/tests/fixtures/test_add.onnx` (generated fixture)
- Create: `crates/deli-infer/tests/integration_tests.rs`

**Key Decisions / Notes:**

- Generate a minimal ONNX model: an "add" model that takes two float tensors and returns their sum. Use Python `onnx` library to create it (a ~20 line script). Check the generated .onnx fixture into the repo so tests work without Python. The Python script is provided for regeneration only (requires `pip install onnx numpy`).
- Integration test flow:
  1. `OnnxBackend::new()` (or `OnnxBackend` struct literal)
  2. `backend.load_model(ModelSource::File("tests/fixtures/test_add.onnx"), Device::Cpu)`
  3. Create two `Tensor<f32>` inputs
  4. `session.run(&[("X", tensor_a), ("Y", tensor_b)])`
  5. Verify output tensor data equals element-wise sum
- Also test `ModelSource::Memory` by reading the .onnx file to bytes and loading from memory
- Test error cases: wrong input name, shape mismatch

**Definition of Done:**

- [ ] Python script generates a valid add model .onnx file
- [ ] Integration test loads model from file and from memory
- [ ] Inference produces correct element-wise sum
- [ ] Error cases (bad input name) return appropriate `InferError`
- [ ] All tests pass with `cargo test -p deli-infer --features onnx`

**Verify:**

- `python3 crates/deli-infer/tests/fixtures/generate_test_model.py` — generates .onnx file
- `cargo test -p deli-infer --features onnx -q` — all tests pass (including integration)

## Testing Strategy

- **Unit tests:** Tensor validation (Task 1), error formatting, Device construction (Task 2), tensor↔ndarray conversion (Task 3)
- **Integration tests:** Full inference pipeline with real ONNX model (Task 4)
- **Manual verification:** `cargo check -p deli-infer` without features to ensure clean conditional compilation

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `ort` crate download of ONNX Runtime binaries fails in CI/offline | Medium | Medium | Tests requiring ONNX are behind `--features onnx`. For offline builds, set `ORT_LIB_LOCATION` to a system-installed ONNX Runtime. Default `cargo test` without features still runs trait/tensor tests. |
| ndarray version conflict | Low | High | Pin `ndarray = "0.17"` explicitly; ort 2.0 requires 0.17+. Document in Cargo.toml comment. |
| ONNX Runtime binaries unavailable for specific arm64 platform | Low | High | The `ort` crate provides pre-built aarch64-linux binaries. For unsupported platforms, users can build ONNX Runtime from source and set `ORT_LIB_LOCATION`. Document this in the crate README. |
| CUDA/TensorRT not available on test machines | High | Low | Integration tests use `Device::Cpu` only. CUDA/TensorRT paths are compile-checked but not runtime-tested in CI. |
| Edition 2024 compatibility with ort | Low | Medium | ort 2.0 supports recent Rust editions. Verify with `cargo check` early in Task 3. |

## Open Questions

- None at this time. Architecture decisions are settled.

### Deferred Ideas

- Async inference API (`async fn run(...)`)
- Model caching / download manager
- Higher-level model APIs (speech recognition, face recognition, object detection)
- Additional backends (TensorFlow Lite, candle, tract)
- Quantization helpers
- Batch inference utilities
