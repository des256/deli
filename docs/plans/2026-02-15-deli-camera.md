# deli-camera Crate Implementation Plan

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

**Goal:** Create a `deli-camera` crate with a `Camera` trait and V4L2 backend. The async `recv()` method captures MJPEG frames via V4L2, decodes them using `deli-image`, and returns `Tensor<u8>` in HWC layout.

**Architecture:** A `Camera` trait defines the async interface (`recv() -> Tensor<u8>`). A `V4l2Camera` struct implements it using the `v4l` crate for V4L2 capture. Internally, a dedicated thread runs the blocking V4L2 mmap stream loop and sends decoded frames over a `tokio::sync::mpsc` channel. The public `recv()` awaits the channel receiver. MJPEG is requested as the pixel format; each raw frame is decoded to RGB `Tensor<u8>` using `deli_image::decode_image`.

**Tech Stack:** `v4l` (V4L2 bindings, behind `v4l2` feature flag), `tokio` (async runtime + mpsc channel), `deli-image` (JPEG decode), `deli-base` (Tensor)

## Scope

### In Scope

- `Camera` trait with async `recv(&mut self) -> Result<Tensor<u8>, CameraError>`
- `V4l2Camera` struct implementing the trait
- Configuration: device path, resolution, frame rate
- MJPEG capture format with automatic decode to `Tensor<u8>` (HWC, RGB)
- Error types for device, stream, and decode failures
- Unit tests (with mocked frame data for decode path) and doc tests

### Out of Scope

- Raspberry Pi camera backend (planned for later)
- Camera controls (exposure, white balance, focus, etc.)
- Multiple simultaneous cameras
- Format negotiation beyond MJPEG
- Video recording or encoding
- GUI or display

## Prerequisites

- `deli-image` crate available (already implemented and verified)
- `deli-base` crate provides `Tensor<T>` and `TensorError`
- Linux system with V4L2 support for integration testing

## Context for Implementer

> This section is critical for cross-session continuity. Write it for an implementer who has never seen the codebase.

- **Patterns to follow:** Follow `deli-image` crate structure — `src/lib.rs` re-exports, `src/error.rs` for error enum with `From` impls, `src/types.rs` for data types. See `crates/deli-image/src/error.rs:1-30` for the error pattern.
- **Conventions:** Workspace uses `crates/*` layout, Rust edition 2024, no async runtime exists yet. All crates use `deli-base` for `Tensor<T>`. Image tensors use HWC layout `[height, width, channels]`.
- **Key files:**
  - `crates/deli-base/src/tensor.rs` — `Tensor<T>` struct with `new(shape, data)`, `shape`, `data` fields
  - `crates/deli-image/src/lib.rs` — `decode_image(&[u8]) -> Result<DecodedImage, ImageError>` decodes any image format
  - `crates/deli-image/src/types.rs` — `DecodedImage` enum (U8/U16/F32 variants wrapping `Tensor<T>`)
- **Gotchas:**
  - `decode_image` returns `DecodedImage` enum, not `Tensor<u8>` directly. Must match on the `U8` variant.
  - MJPEG frames from V4L2 are standard JPEG data — `decode_image` handles them directly.
  - V4L2 `MmapStream::with_buffers` borrows the device — the stream must live alongside the device.
  - The `v4l` crate's `CaptureStream::next()` returns `(&[u8], Metadata)` — the buffer is borrowed and only valid until the next `next()` call, so data must be copied.
- **Domain context:** V4L2 (Video for Linux 2) is the Linux kernel API for video capture. Devices appear at `/dev/video0`, `/dev/video1`, etc. MJPEG is a compressed format where each frame is an independent JPEG — this minimizes USB bandwidth and CPU-side decode is fast via `deli-image`.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Error types and Camera trait
- [x] Task 2: V4l2Camera configuration and device opening
- [x] Task 3: Frame capture thread and async recv

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Error types and Camera trait

**Objective:** Define the `CameraError` enum and the `Camera` trait with async `recv`.

**Dependencies:** None

**Files:**

- Create: `crates/deli-camera/Cargo.toml`
- Create: `crates/deli-camera/src/lib.rs`
- Create: `crates/deli-camera/src/error.rs`
- Create: `crates/deli-camera/src/traits.rs`
- Test: `crates/deli-camera/tests/error_tests.rs`
- Test: `crates/deli-camera/tests/trait_tests.rs`

**Key Decisions / Notes:**

- `CameraError` variants: `Device(String)` (V4L2 device errors), `Stream(String)` (streaming errors), `Decode(deli_image::ImageError)` (JPEG decode failures), `Channel(String)` (internal mpsc channel closed)
- `From<io::Error>` and `From<ImageError>` impls for ergonomic `?` usage
- `Camera` trait:
  ```rust
  pub trait Camera {
      async fn recv(&mut self) -> Result<Tensor<u8>, CameraError>;
  }
  ```
  Uses native async trait syntax (Rust edition 2024 supports this without `async-trait` crate)
- `Cargo.toml` dependencies: `deli-base`, `deli-image`, `tokio` (features: `rt`, `sync`). `v4l` is an **optional dependency** behind `v4l2` feature flag.
  ```toml
  [features]
  default = []
  v4l2 = ["dep:v4l"]

  [dependencies]
  v4l = { version = "0.14", optional = true }
  ```
- All V4L2-specific code (`v4l2.rs` module, `V4l2Camera` struct) is gated with `#[cfg(feature = "v4l2")]`
- Trait tests: verify a mock implementation compiles and can be used polymorphically

**Definition of Done:**

- [ ] `CameraError` has `Device`, `Stream`, `Decode`, `Channel` variants
- [ ] `From<std::io::Error>` converts to `CameraError::Device`
- [ ] `From<ImageError>` converts to `CameraError::Decode`
- [ ] `Camera` trait has async `recv(&mut self) -> Result<Tensor<u8>, CameraError>`
- [ ] Tests verify error conversions and trait is object-safe enough for concrete use

**Verify:**

- `cargo build -p deli-camera` — crate compiles
- `cargo test -p deli-camera -- error` — error tests pass
- `cargo test -p deli-camera -- trait` — trait tests pass

### Task 2: V4l2Camera configuration and device opening

**Objective:** Create `V4l2Camera` struct with a builder/config that opens a V4L2 device and sets MJPEG format.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-camera/src/config.rs`
- Create: `crates/deli-camera/src/v4l2.rs` (gated with `#[cfg(feature = "v4l2")]`)
- Modify: `crates/deli-camera/src/lib.rs` (add module declarations and re-exports, V4L2 module conditional)
- Test: `crates/deli-camera/tests/config_tests.rs`

**Key Decisions / Notes:**

- `CameraConfig` struct with fields:
  - `device: String` (default: `"/dev/video0"`)
  - `width: u32` (default: `640`)
  - `height: u32` (default: `480`)
  - `fps: u32` (default: `30`)
  - `buffer_count: u32` (default: `4`)
- `CameraConfig` provides `Default` impl and builder-style methods
- `V4l2Camera` struct holds config and will be completed in Task 3
- `V4l2Camera::new(config: CameraConfig) -> Result<Self, CameraError>`:
  - Opens V4L2 device at `config.device`
  - Sets format to MJPEG at requested resolution via `Capture::set_format`
  - Sets frame rate via `Capture::set_params`
  - Stores the configured device for Task 3's streaming
- Config tests verify defaults and builder methods (no hardware needed)
- Device open is tested only by verifying the error path (opening a non-existent device returns `CameraError::Device`)

**Definition of Done:**

- [ ] `CameraConfig` has `device`, `width`, `height`, `fps`, `buffer_count` fields with defaults
- [ ] `CameraConfig` builder methods return `Self` for chaining
- [ ] `V4l2Camera::new(config)` opens device and sets MJPEG format
- [ ] Opening a non-existent device path returns `CameraError::Device`
- [ ] Config default tests pass

**Verify:**

- `cargo build -p deli-camera --features v4l2` — compiles with V4L2 feature
- `cargo build -p deli-camera` — compiles without V4L2 feature (trait + config only)
- `cargo test -p deli-camera -- config` — config tests pass

### Task 3: Frame capture thread and async recv

**Objective:** Implement the background capture thread that reads V4L2 frames, decodes MJPEG to `Tensor<u8>`, and sends them through a tokio mpsc channel. Implement `Camera::recv` for `V4l2Camera`.

**Dependencies:** Task 2

**Files:**

- Modify: `crates/deli-camera/src/v4l2.rs` (add streaming logic and `Camera` impl)
- Modify: `crates/deli-camera/src/lib.rs` (update re-exports)
- Test: `crates/deli-camera/tests/recv_tests.rs`

**Key Decisions / Notes:**

- `V4l2Camera::new()` (from Task 2) creates the device. A separate `start(&mut self)` method or lazy-start on first `recv` spawns the capture thread. Prefer lazy-start: first `recv()` call spawns the thread if not already running.
- Capture thread:
  1. Creates `MmapStream::with_buffers(&device, Type::VideoCapture, buffer_count)`
  2. Loops: `CaptureStream::next(&mut stream)` → copies `&[u8]` to `Vec<u8>` → `deli_image::decode_image(&data)` → extracts `Tensor<u8>` from `DecodedImage::U8` variant → sends via `tokio::sync::mpsc::Sender`
  3. On error: sends error through a separate error mechanism or logs and continues
  4. Stops when sender is dropped (receiver dropped) or a stop signal is sent
- `V4l2Camera` holds: `config`, `device: Option<v4l::Device>`, `receiver: Option<mpsc::Receiver<Result<Tensor<u8>, CameraError>>>`, `thread_handle: Option<JoinHandle<()>>`
- `Camera::recv` impl: ensures thread is started, then `self.receiver.recv().await`
- Channel capacity: `config.buffer_count as usize` (bounded channel, backpressure if consumer is slow)
- `Drop` impl: drop the receiver to signal the thread, then join the thread handle
- Tests: test the decode pipeline in isolation (create a fake MJPEG frame → decode → verify tensor shape). Integration test with real hardware is manual-only (documented but not in CI).

**Definition of Done:**

- [ ] Background thread captures frames via V4L2 mmap streaming
- [ ] Each MJPEG frame is decoded to `Tensor<u8>` with HWC layout `[height, width, 3]`
- [ ] `Camera::recv` returns decoded frames from the channel
- [ ] Channel uses bounded capacity (backpressure)
- [ ] Drop impl signals thread to stop and joins it
- [ ] Decode pipeline test passes (synthetic JPEG → decode → verify tensor)

**Verify:**

- `cargo build -p deli-camera --features v4l2` — compiles
- `cargo test -p deli-camera --features v4l2` — all tests pass
- `cargo test -p deli-camera --features v4l2 -- recv` — recv-specific tests pass

## Testing Strategy

- **Unit tests:** Error conversions, config defaults/builder, decode pipeline with synthetic JPEG data
- **Integration tests:** Manual only — requires a real V4L2 camera. Document a manual test procedure in the crate's README or as an ignored test (`#[ignore]`)
- **No mocking of V4L2:** The V4L2 device interaction is thin (open, set format, stream). Tests focus on the decode pipeline and error handling, not on simulating the kernel API.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| V4L2 device doesn't support MJPEG | Low | High | Check format after `set_format` — if the device changed it to something else, return `CameraError::Device("MJPEG not supported")` |
| `decode_image` returns non-U8 variant | Low | Medium | If `DecodedImage` is not `U8`, return `CameraError::Decode` with a descriptive message — JPEG always decodes to U8 but guard against it |
| Channel full causes frame drops | Medium | Low | Use bounded channel with `try_send` — if full, drop oldest frame (log warning). Consumer controls pace. |
| Thread panic on V4L2 error | Low | High | Wrap thread body in catch_unwind or propagate errors through the channel. Send the error as `Err(CameraError::Stream(...))` rather than panicking. |

## Open Questions

- None — all design decisions resolved.

### Deferred Ideas

- Raspberry Pi camera backend (libcamera)
- Camera controls (exposure, white balance, gain)
- Multiple camera support
- YUYV format support with software conversion
- Frame timestamp propagation
