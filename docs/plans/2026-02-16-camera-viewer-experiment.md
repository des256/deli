# Camera-Viewer Experiment Implementation Plan

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

**Goal:** Create an experiment with two executables: `camera` (captures frames via deli-camera and broadcasts via deli-com SenderServer) and `viewer` (receives frames via deli-com ReceiverClient and displays in a window).

**Architecture:** A single Cargo crate at `experiments/camera-viewer/` with two `[[bin]]` targets (`camera` and `viewer`) and a shared library (`src/lib.rs`) defining a `Frame` struct that derives `Codec`. The `camera` binary opens a V4L2 camera, binds a `SenderServer<Frame>` on a TCP port, and loops: capture frame → broadcast. The `viewer` binary connects a `ReceiverClient<Frame>` to the camera's address, and loops: receive frame → convert to ARGB → display in a `minifb` window.

**Tech Stack:** Rust (edition 2024), deli-camera (V4L2), deli-com (SenderServer/ReceiverClient), deli-codec (Codec derive), deli-base (logging), minifb (window display), tokio (async runtime).

## Scope

### In Scope

- `Vec<T: Codec>` implementation in deli-codec (needed for Frame's data field)
- `Frame` struct with `width: u32, height: u32, data: Vec<u8>` deriving `Codec`
- `camera` binary: opens camera, binds SenderServer, broadcasts frames in a loop
- `viewer` binary: connects ReceiverClient, receives frames, displays in minifb window
- Logging via deli-base (init_stdout_logger)

### Out of Scope

- Multiple camera support
- Frame compression or encoding
- Configuration file support
- Graceful reconnection on viewer disconnect

## Prerequisites

- `deli-camera` crate with V4L2 support (exists)
- `deli-com` crate with SenderServer/ReceiverClient (exists)
- `deli-codec` crate with Codec derive macro (exists)
- `deli-base` crate with logging (exists)

## Context for Implementer

- **Patterns to follow:** `experiments/camera-view/src/main.rs` — existing camera + minifb display pattern. `experiments/camera-pose/src/main.rs` — similar but with inference.
- **Conventions:** Edition 2024, experiments are workspace members at `experiments/*`, tests in `tests/` directory with `_tests.rs` suffix. Logging via `deli_base::log` macros with `init_stdout_logger()` at binary startup.
- **Key files:**
  - `crates/deli-com/src/sender.rs` — `SenderServer<T: Codec>` with `bind(addr)` and `send(&T)` and `client_count()`
  - `crates/deli-com/src/receiver.rs` — `ReceiverClient<T: Codec>` with `connect(addr)` and `recv()`
  - `crates/deli-camera/src/lib.rs` — `Camera` trait with `recv()` returning `Tensor<u8>`
  - `crates/deli-codec/src/lib.rs` — `Codec` trait with `encode`/`decode`, derive macro available
  - `crates/deli-codec/src/primitives.rs` — Codec impls for bool, numerics, String (no Vec<T> yet)
  - `experiments/camera-view/src/main.rs` — reference for camera + minifb pattern
- **Gotchas:**
  - `SenderServer::bind` and `ReceiverClient::connect` are async — need tokio runtime
  - Camera `recv()` returns `Tensor<u8>` with shape `[H, W, 3]` — extract `.data` (Vec<u8>) for the Frame
  - `Vec<T>` does NOT implement Codec yet — must add this to deli-codec primitives first
  - Codec derive macro generates code that calls `.encode()`/`.decode()` on each field, so Vec<u8> Codec impl is required
  - minifb expects ARGB u32 packed pixels — need RGB→ARGB conversion (same as camera-view)
  - Default address should be `127.0.0.1:9920` (or similar) for both camera and viewer to agree on

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add Vec<T: Codec> to deli-codec primitives
- [x] Task 2: Create experiment crate with Frame type and camera binary
- [x] Task 3: Implement viewer binary

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add Vec<T: Codec> to deli-codec primitives

**Objective:** Add a `Codec` implementation for `Vec<T: Codec>` in deli-codec so that the Frame struct (which contains `Vec<u8>`) can derive Codec.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-codec/src/primitives.rs`
- Test: `crates/deli-codec/tests/codec_tests.rs` (create if doesn't exist, or add to existing)

**Key Decisions / Notes:**

- Follow the same length-prefix pattern as the String impl: encode length as u32, then encode each element
- For `Vec<u8>` specifically this is very efficient: `u32` length + raw bytes (each u8 encodes as itself)
- The generic impl handles any `Vec<T: Codec>` which makes the codec system more capable
- Use `read_bytes` helper for the u8 special case is not needed — generic approach works fine: encode count, then each element

**Definition of Done:**

- [x] `Vec<T: Codec>` implements `Codec` with length-prefix encoding
- [x] Test roundtrips `Vec<u8>`, `Vec<u32>`, and empty vec
- [x] `cargo check -p deli-codec` — no errors
- [x] Existing deli-codec tests still pass

**Verify:**

- `cargo test -p deli-codec -q` — all tests pass

### Task 2: Create experiment crate with Frame type and camera binary

**Objective:** Create the `experiments/camera-viewer/` crate with a shared `Frame` struct and the `camera` binary that captures from V4L2 and broadcasts via SenderServer.

**Dependencies:** Task 1

**Files:**

- Create: `experiments/camera-viewer/Cargo.toml`
- Create: `experiments/camera-viewer/src/lib.rs` (shared Frame type)
- Create: `experiments/camera-viewer/src/camera.rs` (camera binary)

**Key Decisions / Notes:**

- `Cargo.toml` declares two `[[bin]]` targets: `camera` (path `src/camera.rs`) and `viewer` (path `src/viewer.rs`)
- Dependencies: `deli-base`, `deli-camera` (features = ["v4l2"]), `deli-com`, `deli-codec`, `tokio` (features = ["rt", "rt-multi-thread", "macros"]), `minifb`
- `src/lib.rs` defines:
  ```rust
  #[derive(deli_codec::Codec)]
  pub struct Frame {
      pub width: u32,
      pub height: u32,
      pub data: Vec<u8>,
  }
  ```
- `src/camera.rs` main:
  1. `init_stdout_logger()`
  2. Parse optional address from args, default `0.0.0.0:9920`
  3. Open V4l2Camera with default config (640x480)
  4. Bind `SenderServer::<Frame>::bind(addr)`
  5. Log the listening address
  6. Loop: `camera.recv().await` → extract width/height/data from Tensor → create Frame → `sender.send(&frame).await`
  7. Log client count periodically or on change
- Address `0.0.0.0:9920` allows external connections; viewer uses `127.0.0.1:9920`

**Definition of Done:**

- [x] `Frame` struct derives `Codec` and is defined in `src/lib.rs`
- [x] `camera` binary compiles and runs (captures + broadcasts)
- [x] `cargo check -p camera-viewer` — no errors
- [x] Logging shows camera status and client connections

**Verify:**

- `cargo check -p camera-viewer` — no errors

### Task 3: Implement viewer binary

**Objective:** Implement the `viewer` binary that connects to the camera broadcast, receives frames, and displays them in a minifb window.

**Dependencies:** Task 2

**Files:**

- Create: `experiments/camera-viewer/src/viewer.rs`

**Key Decisions / Notes:**

- `src/viewer.rs` main:
  1. `init_stdout_logger()`
  2. Parse optional address from args, default `127.0.0.1:9920`
  3. Connect `ReceiverClient::<Frame>::connect(addr)`
  4. Create minifb Window (use frame dimensions from first received frame, or default 640x480)
  5. Loop: `receiver.recv().await` → convert RGB data to ARGB u32 buffer → `window.update_with_buffer()`
  6. Break on ESC key or window close
- RGB→ARGB conversion: same pattern as `camera-view/src/main.rs` (`(r << 16) | (g << 8) | b`)
- Window title: "Camera Viewer - ESC to exit"

**Definition of Done:**

- [x] `viewer` binary compiles and connects to camera broadcast
- [x] Received frames are displayed in a minifb window
- [x] Window closes on ESC
- [x] `cargo check -p camera-viewer` — no errors

**Verify:**

- `cargo check -p camera-viewer` — no errors

## Testing Strategy

- **Unit tests:** Vec<T> Codec roundtrip in deli-codec (Task 1). Frame Codec roundtrip in experiment tests.
- **Integration tests:** Not needed — the camera and viewer are verified by running them together.
- **Manual verification:** Run `camera` in one terminal, `viewer` in another. Verify video appears in the viewer window.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Frame data too large for TCP without compression | Medium | Medium | V4L2 at 640x480x3 = 921KB per frame. At 30fps = ~27MB/s. TCP on localhost handles this easily. For remote use, reduce resolution or add compression later. |
| Viewer connects before camera is ready | Medium | Low | ReceiverClient::connect returns error if camera isn't bound yet. Viewer should log the error and exit — user retries. |
| Camera negotiates different resolution than requested | Low | Low | Extract actual width/height from Tensor shape (frame.shape[1], frame.shape[0]) rather than hardcoding. |

## Open Questions

- None — design is straightforward.

### Deferred Ideas

- Frame compression (JPEG/H264) for remote viewing
- Multiple viewer support (already works — SenderServer broadcasts to all clients)
- Auto-reconnect on viewer when camera restarts
- FPS counter overlay
