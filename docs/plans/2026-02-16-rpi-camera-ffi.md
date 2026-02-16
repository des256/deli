# RPi Camera FFI Implementation Plan

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

**Goal:** Add an `RPiCamera` backend to `deli-camera` that uses the `libcamera` crate (safe Rust bindings for the libcamera C++ library) to capture frames on Raspberry Pi 5 hardware.

**Architecture:** Use the existing `libcamera` crate (v0.6.0) as an optional dependency behind an `rpicam` feature flag. `RPiCamera` follows the same background-thread + channel pattern as `V4l2Camera`: a dedicated thread runs the libcamera capture loop, decodes MJPEG frames via `deli-image`, and sends `Tensor<u8>` results over a `tokio::mpsc` channel. Integration tests are skipped since libcamera is not available on the dev host; unit tests use compile-time feature gating.

**Tech Stack:** `libcamera` crate (v0.6.0), `deli-image` for MJPEG decoding, `tokio::sync::mpsc` for async frame delivery.

## Scope

### In Scope

- Add `rpicam` feature to `deli-camera/Cargo.toml` with `libcamera` as optional dependency
- Implement `RPiCamera` struct in `crates/deli-camera/src/rpicam.rs`
- `RPiCamera` implements the `Camera` trait (`async fn recv`)
- Capture loop using libcamera's `CameraManager`, `FrameBufferAllocator`, and request-based capture API
- MJPEG capture with decoding via `deli-image::decode_image`, with YUYV fallback and manual YUV→RGB conversion
- Feature-gated module and re-export in `lib.rs`
- Unit tests for `RPiCamera` construction logic (feature-gated, testing error paths with unavailable hardware)

### Out of Scope

- Integration tests against real libcamera hardware (not available on dev host)
- RPi vendor-specific controls (`vendor_rpi` feature of the libcamera crate) — can be added later
- Non-RGB pixel formats beyond MJPEG and YUYV (e.g., NV12, NV21)
- Camera hot-plug detection
- Multiple simultaneous streams

## Prerequisites

- The `libcamera` system library (v0.4.0+) must be installed on the target RPi 5 and accessible via `pkg-config` for compilation
- The `libcamera` crate (v0.6.0) will be added as an optional dependency — it only compiles when the `rpicam` feature is enabled

## Context for Implementer

- **Patterns to follow:** The `V4l2Camera` implementation at `crates/deli-camera/src/v4l2.rs` is the template. It uses a background `std::thread` for capture, `tokio::sync::mpsc` for frame delivery, `deli_image::decode_image` for MJPEG→Tensor conversion, and `Drop` for cleanup.
- **Conventions:** Feature-gated modules use `#[cfg(feature = "...")]` on both the `pub mod` declaration in `lib.rs` and the `pub use` re-export. Builder-pattern config via `CameraConfig`. Errors use the `CameraError` enum.
- **Key files:**
  - `crates/deli-camera/src/lib.rs` — module declarations and re-exports
  - `crates/deli-camera/src/v4l2.rs` — reference implementation to mirror
  - `crates/deli-camera/src/traits.rs` — the `Camera` trait definition
  - `crates/deli-camera/src/config.rs` — `CameraConfig` with builder methods
  - `crates/deli-camera/src/error.rs` — `CameraError` enum
  - `crates/deli-camera/Cargo.toml` — feature flags and dependencies
- **Verified API surface (from `libcamera` crate v0.6.0 source and `jpeg_capture.rs` example):**
  - `CameraManager::new() -> io::Result<Self>` — returns `std::io::Error` which is already covered by `From<std::io::Error> for CameraError`. No additional error conversion needed.
  - `cameras.get(0) -> Option<Camera>` — camera selection by index
  - `cam.acquire() -> io::Result<ActiveCamera>` — exclusive lock
  - `cam.generate_configuration(&[StreamRole::ViewFinder]) -> Option<CameraConfiguration>`
  - `cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT_MJPEG)` — MJPEG format via `PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0)`
  - `FrameBufferAllocator::new(&cam)`, `alloc.alloc(&stream) -> io::Result<Vec<FrameBuffer>>`
  - `MemoryMappedFrameBuffer::new(buf) -> io::Result<Self>` — wraps DMA buffer for readable `&[u8]` access via `.data()` method
  - `cam.on_request_completed(move |req| { tx.send(req).unwrap(); })` — callback takes `move` closure, uses `std::sync::mpsc::Sender` to bridge to capture thread
  - `cam.start(None)`, `cam.queue_request(req)`, `request.reuse(ReuseFlag::ReuseBuffers)`
- **Gotchas:**
  - libcamera uses a request-completed callback model, not a blocking read loop like V4L2. The capture thread must queue requests and wait for completion via `std::sync::mpsc`. The `jpeg_capture.rs` example demonstrates the exact pattern: create `std::sync::mpsc::channel`, pass `tx` into `on_request_completed` via `move` closure, receive completed requests on the capture thread via `rx.recv()`.
  - `CameraConfig.device()` returns a path like `/dev/video0` which is V4L2-specific. For libcamera, the camera is selected by index (first camera). The `device` field in `CameraConfig` is ignored for `RPiCamera`. Add a `log::warn!` if device is not the default, and document this in `RPiCamera::new()` doc comment.
  - `MemoryMappedFrameBuffer` wraps a `FrameBuffer` to provide `&[u8]` access via `.data()`. Buffer data is valid for the lifetime of the `MemoryMappedFrameBuffer`. In the capture loop, MJPEG data must be copied (via `.to_vec()`) before the request is reused, since reuse invalidates the buffer contents. This mirrors the V4l2Camera pattern where `frame_data.to_vec()` copies before the next `CaptureStream::next()` call.
  - The `libcamera` crate's build script uses `pkg-config` to find the system library. If libcamera is not installed, compilation of the `rpicam` feature will fail at link time — this is expected and acceptable.
  - Pixel formats are specified via `PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0)` for MJPEG and `PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0)` for YUYV. After `set_pixel_format` + `cfgs.validate()`, check if the pixel format was adjusted — if it was, the camera doesn't support that format. Try the next format in the fallback chain.
- **Domain context:** The Raspberry Pi 5 uses the libcamera stack instead of direct V4L2 access for its camera modules (IMX219, IMX477, etc.). libcamera handles ISP pipeline configuration internally.

## Runtime Environment

> This section documents manual verification on RPi 5 hardware. Not applicable on the dev host.

- **Target hardware:** Raspberry Pi 5 with camera module (IMX219, IMX477, or compatible) connected to CSI port
- **System dependencies:**
  ```bash
  sudo apt install libcamera-dev libcamera-tools
  ```
- **Verify camera detected:** `libcamera-hello --list-cameras`
- **Build with rpicam feature:**
  ```bash
  cargo build -p deli-camera --features rpicam
  ```
- **Run rpicam tests:**
  ```bash
  cargo test -p deli-camera --features rpicam
  ```
- **Manual verification:**
  1. Modify `experiments/camera-viewer/Cargo.toml` to use `features = ["rpicam"]` instead of `features = ["v4l2"]`
  2. Modify `experiments/camera-viewer/src/camera.rs` to use `RPiCamera::new()` instead of `V4l2Camera::new()`
  3. Run: `cargo run -p camera-viewer`
  4. Expected: Window displays camera feed at configured resolution and FPS

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add rpicam feature and libcamera dependency
- [x] Task 2: Implement RPiCamera struct with Camera trait
- [x] Task 3: Feature-gate module in lib.rs and add unit tests

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add rpicam feature and libcamera dependency

**Objective:** Add the `rpicam` Cargo feature to `deli-camera` that enables the `libcamera` crate as an optional dependency.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-camera/Cargo.toml`

**Key Decisions / Notes:**

- Add `rpicam = ["dep:libcamera"]` to `[features]`
- Add `libcamera = { version = "0.6", optional = true }` to `[dependencies]`
- Do NOT add `rpicam` to `default` features — it requires libcamera system library

**Definition of Done:**

- [ ] `rpicam` feature exists in `Cargo.toml`
- [ ] `libcamera` is listed as optional dependency
- [ ] `cargo check -p deli-camera` succeeds (without rpicam feature)

**Verify:**

- `cargo check -p deli-camera` (without `--features rpicam`) — default build succeeds without libcamera installed

### Task 2: Implement RPiCamera struct with Camera trait

**Objective:** Create `rpicam.rs` implementing `RPiCamera` that captures MJPEG frames via libcamera and delivers them as `Tensor<u8>` through the `Camera` trait.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-camera/src/rpicam.rs`

**Key Decisions / Notes:**

- Follow the `V4l2Camera` architecture: `new(config) -> Result<Self>`, lazy `ensure_started()`, background `capture_loop()`, `Drop` cleanup
- **Format negotiation in `new()`:** Try MJPEG first. If the camera doesn't support MJPEG (validate returns Invalid or adjusts format), fall back to YUYV. Store the negotiated format (an enum `CaptureFormat { Mjpeg, Yuyv }`) in the struct so the capture loop knows which decode path to use.
- **YUYV→RGB conversion:** Implement a `yuyv_to_rgb(data: &[u8], width: u32, height: u32) -> Vec<u8>` function in `rpicam.rs`. YUYV (YUV 4:2:2) packs as `[Y0, U, Y1, V, ...]` — each pair of pixels shares U and V. Convert to RGB using standard BT.601 coefficients: `R = Y + 1.402*(V-128)`, `G = Y - 0.344*(U-128) - 0.714*(V-128)`, `B = Y + 1.772*(U-128)`. Output is `[R, G, B, R, G, B, ...]` with 3 bytes per pixel.
- The capture loop:
  1. Create `CameraManager`, acquire first camera
  2. Configure stream with negotiated format (MJPEG or YUYV) at requested resolution
  3. Allocate frame buffers with `FrameBufferAllocator`
  4. Memory-map buffers with `MemoryMappedFrameBuffer`
  5. Create requests, attach buffers
  6. Set `on_request_completed` callback to send completed requests through `std::sync::mpsc`
  7. Start camera, queue all requests
  8. In loop: receive completed request → read frame data from buffer → decode based on format:
     - MJPEG: `deli_image::decode_image(&data)` → extract `Tensor<u8>`
     - YUYV: `yuyv_to_rgb(&data, width, height)` → construct `Tensor::new(vec![h, w, 3], rgb_data)`
  9. Send `Tensor<u8>` over `tokio::sync::mpsc` → reuse request and re-queue
- The `device` field in `CameraConfig` is ignored — libcamera selects cameras by index (uses first available camera). Document this clearly.
- Use `try_send` to drop frames when channel is full (same as V4l2Camera)
- `camera_index` could be derived from config's device path in the future, but for now use camera index 0

**Definition of Done:**

- [ ] `RPiCamera` struct has `new(CameraConfig) -> Result<Self, CameraError>` constructor
- [ ] `RPiCamera` implements `Camera` trait (correct type signature for `async fn recv`)
- [ ] `ensure_started()` method spawns background capture thread on first call
- [ ] `Drop` implementation drops receiver and joins thread handle
- [ ] Format negotiation tries MJPEG first, falls back to YUYV if MJPEG unavailable
- [ ] `yuyv_to_rgb()` conversion function produces correct RGB output (3 bytes/pixel, BT.601 coefficients)
- [ ] Capture loop decodes MJPEG via `deli_image::decode_image` or converts YUYV via `yuyv_to_rgb`
- [ ] Code structurally mirrors the `V4l2Camera` pattern (background thread, `tokio::mpsc` channel, `try_send` for backpressure)
- [ ] `RPiCamera::new()` doc comment warns that `config.device()` is ignored; always uses first available camera
- [ ] MJPEG buffer data is copied before request reuse (no use-after-free)

**Verify:**

- If libcamera-dev is installed: `cargo check -p deli-camera --features rpicam` succeeds
- If libcamera-dev is NOT installed: verification is deferred to RPi 5 hardware (see Runtime Environment section). Code review confirms structural correctness.

### Task 3: Feature-gate module in lib.rs and add unit tests

**Objective:** Wire up the `rpicam` module in `lib.rs` with proper feature gating and add unit tests.

**Dependencies:** Task 2

**Files:**

- Modify: `crates/deli-camera/src/lib.rs`
- Create: `crates/deli-camera/tests/rpicam_tests.rs`

**Key Decisions / Notes:**

- Add `#[cfg(feature = "rpicam")] pub mod rpicam;` and `#[cfg(feature = "rpicam")] pub use rpicam::RPiCamera;` in `lib.rs`, mirroring the v4l2 pattern
- Tests are feature-gated with `#[cfg(feature = "rpicam")]` — they will only run when the feature is enabled (i.e., on a system with libcamera installed)
- Test `RPiCamera::new()` with invalid/unavailable camera — should return `CameraError::Device` (feature-gated)
- Test `yuyv_to_rgb()` with known input/output pairs — pure logic, feature-gated because the function is in the rpicam module
- Since we can't run integration tests on the dev host, the test file tests error paths and conversion logic only (similar to `v4l2_tests.rs` which tests `test_v4l2_camera_invalid_device`)

**Definition of Done:**

- [ ] `rpicam` module is conditionally compiled in `lib.rs`
- [ ] `RPiCamera` is re-exported under `#[cfg(feature = "rpicam")]`
- [ ] `rpicam_tests.rs` exists with feature-gated tests for error paths and `yuyv_to_rgb` conversion
- [ ] `cargo check -p deli-camera` still succeeds without `rpicam` feature
- [ ] `cargo test -p deli-camera` passes (default features, no rpicam tests run)

**Verify:**

- `cargo check -p deli-camera` (without `--features rpicam`) — default build succeeds without libcamera
- `cargo test -p deli-camera` (without `--features rpicam`) — existing tests pass, rpicam tests are compiled out

## Testing Strategy

- **Unit tests:** Feature-gated error-path tests in `tests/rpicam_tests.rs` (e.g., invalid device returns `CameraError::Device`). Pure-logic test for `yuyv_to_rgb` conversion (does NOT require libcamera — this test can run on any host). Feature-gated construction tests only compile+run when `rpicam` feature is active.
- **Integration tests:** Skipped — libcamera is not available on the dev host. On an actual RPi 5, one would enable the `rpicam` feature and run `cargo test -p deli-camera --features rpicam`.
- **Manual verification:** On RPi 5 hardware, run the `camera-viewer` experiment modified to use `RPiCamera` instead of `V4l2Camera`.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `libcamera` crate API changes in future versions | Low | Med | Pin to `0.6` in Cargo.toml; the crate follows semver |
| libcamera system library not available during CI builds | High | Low | Feature is optional and not in `default`; CI runs `cargo check` and `cargo test` without `rpicam` feature |
| Neither MJPEG nor YUYV supported by camera | Very Low | High | Format negotiation tries MJPEG first, then YUYV. If both fail, return `CameraError::Device("no supported pixel format (tried MJPEG and YUYV)")`. RPi camera modules support both through libcamera ISP. |
| libcamera request-callback threading model is more complex than V4L2's blocking read | Med | Med | Follow the proven `jpeg_capture.rs` example pattern from `libcamera-rs`; use `std::sync::mpsc` to bridge callback to capture thread. Verification: code review confirms pattern matches the example (structural check on dev host). Functional verification deferred to RPi 5 hardware per Runtime Environment section. |

## Open Questions

- None — all design decisions resolved.

### Deferred Ideas

- Support for additional pixel formats beyond MJPEG and YUYV (e.g., NV12, NV21)
- RPi vendor-specific controls via `libcamera`'s `vendor_rpi` feature (exposure, gain, AWB)
- Camera selection by index or model name instead of always using the first camera. Note: libcamera identifies cameras by string ID (e.g., `\_SB_.PCI0-2:1.0-0c45:2690`), not by device path. A future camera selection API would need to map between CameraConfig's device path and libcamera's camera ID system.
