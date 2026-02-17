# PulseAudio Abstraction Crate Implementation Plan

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
> **Worktree:** Set at plan creation (from dispatcher). `Yes` uses git worktree isolation; `No` works directly on current branch (default)

## Summary

**Goal:** Create a new `deli-audio` crate that abstracts PulseAudio for audio input, starting with an `AudioIn` struct. `AudioIn` provides async `recv()` → `Vec<i16>` for capturing mono 16-bit PCM audio from the selected input device, and `select()` for switching input devices at runtime.

**Architecture:** Follow the same background-thread + tokio mpsc channel pattern used by `deli-camera`'s `V4l2Camera`. A dedicated thread runs the PulseAudio Simple API's blocking `read()` call, sending audio chunks through a channel. The async `recv()` method awaits the next chunk from the channel. Device switching via `select()` tears down the current PulseAudio stream and creates a new one on the selected device.

**Tech Stack:** `libpulse-binding` + `libpulse-simple-binding` (Rust bindings for PulseAudio Simple API), `tokio` (async runtime, mpsc channels)

## Scope

### In Scope

- `AudioIn` struct with `new()`, async `recv()`, `select()`, and `list_devices()`
- `AudioError` error type with Display, Error, and conversions
- Mono S16NE capture at configurable sample rate
- Background capture thread with tokio mpsc channel (matching deli-camera pattern)
- Device enumeration via PulseAudio introspect API
- Runtime device switching (teardown + reconnect)
- Unit tests for error types, device info, and structural correctness (no real audio hardware)

### Out of Scope

- `AudioOut` (deferred to later task per user request)
- Real audio capture tests (require physical microphone)
- Audio format conversion (always mono i16)
- PipeWire support (PulseAudio only)
- CI/Docker configuration for libpulse-dev (no CI pipeline exists in this project)

## Prerequisites

- System libraries: `libpulse-dev` (already installed, verified via `pkg-config`)
- Rust edition 2024 (matching other crates in workspace)

## Context for Implementer

- **Patterns to follow:** The `V4l2Camera` in `crates/deli-camera/src/v4l2.rs` demonstrates the exact architecture: background thread does blocking I/O, sends results through `tokio::sync::mpsc`, struct exposes async `recv()`. The `Drop` impl drops the receiver to signal the thread, then joins it.
- **Conventions:** Builder-style config (`with_*` methods), edition 2024, `log` crate for logging, error types implement `Display` + `Error` manually (no `thiserror`). Tests live in `tests/` directory as integration tests.
- **Key files:**
  - `crates/deli-camera/src/v4l2.rs` — reference architecture for background thread + channel pattern
  - `crates/deli-camera/src/error.rs` — reference error type structure
  - `crates/deli-camera/Cargo.toml` — reference for crate layout and dependencies
- **Gotchas:**
  - PulseAudio Simple API's `read()` is blocking — must run in a dedicated thread, not on tokio runtime
  - `Simple::read()` returns `Result<()>` — on error (device disconnected, server crash), it returns `Err(PAErr)`. The capture loop must handle this by sending `AudioError::Stream` through the channel and exiting the loop.
  - Device enumeration requires the full PulseAudio mainloop + context (not the Simple API). This is a separate synchronous operation.
  - `Simple::new()` accepts `Option<&str>` for device name — `None` means default device
  - When switching devices, the old `Simple` must be dropped before creating a new one
  - `AudioIn` takes `&mut self` for both `recv()` and `select()`, which prevents concurrent calls at the type level. No additional synchronization is needed. Document that `AudioIn` is not `Sync` and should be owned by a single async task.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Crate scaffold and error type
- [x] Task 2: Device enumeration
- [x] Task 3: AudioIn struct with async recv
- [x] Task 4: Device selection (select method)

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Crate scaffold and error type

**Objective:** Create the `deli-audio` crate with its `Cargo.toml`, module structure, and `AudioError` error type.

**Dependencies:** None

**Files:**

- Create: `crates/deli-audio/Cargo.toml`
- Create: `crates/deli-audio/src/lib.rs`
- Create: `crates/deli-audio/src/error.rs`
- Create: `crates/deli-audio/tests/error_tests.rs`

**Key Decisions / Notes:**

- Use edition 2024 to match all other workspace crates
- Dependencies: `libpulse-binding`, `libpulse-simple-binding`, `tokio` (rt, sync), `log`
- `AudioError` variants: `Device(String)`, `Stream(String)`, `Channel(String)` — matching `deli-camera::CameraError` structure
- Implement `Display`, `Error`, and `From<std::io::Error>` for `AudioError`
- Follow the manual error impl pattern from `crates/deli-camera/src/error.rs` (no thiserror)

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `AudioError` variants cover Device, Stream, and Channel errors
- [ ] `From<std::io::Error>` converts to `AudioError::Device`
- [ ] `Display` formats each variant with a descriptive prefix

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

### Task 2: Device enumeration

**Objective:** Implement `list_devices()` that returns available PulseAudio input sources, and an `AudioDevice` struct to represent each device.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-audio/src/device.rs`
- Modify: `crates/deli-audio/src/lib.rs`
- Create: `crates/deli-audio/tests/device_tests.rs`

**Key Decisions / Notes:**

- `AudioDevice` struct with `name: String` (PulseAudio source name, used for `Simple::new`) and `description: String` (human-readable label)
- `list_devices()` is a standalone pub function (not a method on `AudioIn`) — creates a temporary PulseAudio mainloop + context, calls `introspect.get_source_info_list()`, collects results, returns `Vec<AudioDevice>`
- Follow the pattern from `screenpipe` and `audio-select`: create `Mainloop`, connect `Context`, wait for `State::Ready`, call `get_source_info_list`, iterate mainloop, collect `name` and `description` fields from `SourceInfo`
- Use `Rc<RefCell<Vec<AudioDevice>>>` to collect results inside the callback. Inside the callback, acquire `borrow_mut()`, push the device, and immediately drop the guard before any other logic. Never hold the borrow across function calls.
- Filter out monitor sources (they capture output audio, not input) — check if `monitor_of_sink` is not `PA_INVALID_INDEX` (i.e., `!= u32::MAX`)
- Add timeout handling for the PulseAudio connection: iterate the mainloop with a max iteration count (e.g., 100 iterations). If `State::Ready` is not reached, return `AudioError::Device("PulseAudio server unavailable or connection timed out")`
- Handle connection failure states: if context state reaches `State::Failed` or `State::Terminated`, return `AudioError::Device` immediately rather than continuing to iterate
- Tests: verify `AudioDevice` struct construction and field access, verify `list_devices()` signature returns `Result<Vec<AudioDevice>, AudioError>`

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `AudioDevice` struct has `name` and `description` String fields
- [ ] `list_devices()` returns `Result<Vec<AudioDevice>, AudioError>`
- [ ] Implementation contains `monitor_of_sink` check that skips monitor sources (verified by code inspection)
- [ ] Connection to PulseAudio server has timeout/failure handling that returns `AudioError::Device`

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

### Task 3: AudioIn struct with async recv

**Objective:** Implement `AudioIn` with `new()` constructor and async `recv()` method that captures mono S16NE audio from PulseAudio via a background thread.

**Dependencies:** Task 1, Task 2

**Files:**

- Create: `crates/deli-audio/src/audio_in.rs`
- Modify: `crates/deli-audio/src/lib.rs`
- Create: `crates/deli-audio/tests/audio_in_tests.rs`

**Key Decisions / Notes:**

- `AudioIn::new(device: Option<&str>, sample_rate: u32)` — `None` device means PulseAudio default input. Store `device` as `Option<String>` (clone the name), not move ownership. This allows retrying `ensure_started()` after a failure.
- Internal fields: `sample_rate: u32`, `device: Option<String>`, `receiver: Option<mpsc::Receiver<Result<Vec<i16>, AudioError>>>`, `thread_handle: Option<JoinHandle<()>>`
- Lazy start pattern (like V4l2Camera): capture thread starts on first `recv()` call via `ensure_started()`. Unlike V4l2Camera, device name is cloned (not moved) so `ensure_started()` can be retried if `Simple::new()` fails.
- Background thread creates `Simple::new()` with `Direction::Record`, `Format::S16NE`, 1 channel, and the given sample rate. If `Simple::new()` fails, send `AudioError::Device` through the channel and exit immediately.
- Thread reads into a byte buffer of `(sample_rate / 10) * 2` bytes (= 100ms of mono S16NE). After `Simple::read()`:
  - On error: send `AudioError::Stream(err.to_string())` through the channel via `blocking_send` and break the loop
  - On success: verify `bytes_read % 2 == 0` (should always hold for S16NE, but guard against it). Convert `&[u8]` to `Vec<i16>` using `chunks_exact(2)` and `i16::from_ne_bytes`. Send through channel.
- Channel: use `try_send()`. When channel is full (all 4 slots occupied), silently drop the chunk and continue. This provides best-effort delivery with frame dropping under backpressure, matching V4l2Camera's pattern.
- `Drop` impl: drop receiver to signal thread, then join
- `recv()` returns `Result<Vec<i16>, AudioError>`. When channel closes (thread exited), return `AudioError::Stream("capture thread terminated")`
- Tests: verify struct construction, verify `AudioIn::new()` stores device and sample_rate correctly (expose via getters or test internal state)

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `AudioIn::new()` accepts optional device name and sample rate
- [ ] `recv()` is async and returns `Result<Vec<i16>, AudioError>`
- [ ] Capture loop handles `Simple::read()` errors by sending `AudioError::Stream` through channel and exiting
- [ ] Byte-to-i16 conversion uses `chunks_exact(2)` with proper alignment
- [ ] `Drop` cleanly shuts down the capture thread

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

### Task 4: Device selection (select method)

**Objective:** Implement `AudioIn::select()` to switch the capture device at runtime by tearing down the current stream and starting a new one on the selected device.

**Dependencies:** Task 3

**Files:**

- Modify: `crates/deli-audio/src/audio_in.rs`
- Modify: `crates/deli-audio/tests/audio_in_tests.rs`

**Key Decisions / Notes:**

- `select(&mut self, device: &str)` — takes the PulseAudio source name (from `AudioDevice::name`)
- Implementation: drop the receiver (signals thread to exit), join the thread handle, update `self.device`, clear receiver/handle fields. Next `recv()` call will `ensure_started()` with the new device.
- `&mut self` prevents concurrent calls to `recv()` and `select()` at the type level — no additional synchronization needed
- If capture hasn't started yet (no active thread), just update the stored device name
- Tests: verify that calling `select()` before any `recv()` updates the device name (test via a `device()` getter), verify that `select()` clears receiver and thread_handle fields (indicating stream was torn down and will restart on next `recv()`)

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `select()` updates the stored device name
- [ ] `select()` tears down any active stream (drops receiver, joins thread, clears fields)
- [ ] `select()` works both before and after streaming has started (tested via state inspection)

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

## Testing Strategy

- **Unit tests:** Error type construction, Display formatting, From conversions, AudioDevice struct field access, AudioIn construction and state management
- **Integration tests:** Structural tests — verify `AudioIn::new()` returns correctly, `select()` updates internal state, `recv()` on a closed channel returns appropriate error. All tests in `tests/` directory.
- **Manual verification:** Real audio testing is deferred per user request. If audio hardware is available, an optional smoke test can be performed by creating a test binary that calls `AudioIn::new()`, `recv()` in a loop, and prints chunk sizes.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| PulseAudio server unavailable at connection time | Medium | Medium | `list_devices()` iterates mainloop with a max iteration count and returns `AudioError::Device` if `State::Ready` is not reached. `ensure_started()` sends `AudioError::Device` through channel if `Simple::new()` fails. Device name is cloned (not moved), so the user can retry after fixing the PulseAudio server. |
| Device disconnected during capture (USB mic unplugged, PA crash) | Medium | Medium | Capture loop wraps `Simple::read()` in a match — on error, sends `AudioError::Stream` via `blocking_send` and breaks the loop. Consumer's `recv()` receives the error, then gets `AudioError::Stream("capture thread terminated")` on subsequent calls. |
| `libpulse-binding` crate version incompatibility | Low | Medium | Pin to latest stable versions; verify both crates compile together in Task 1 |

## Open Questions

- None — requirements are clear.

### Deferred Ideas

- `AudioOut` struct for audio playback (explicitly deferred by user)
- Audio format conversion (f32, stereo, resampling)
- PipeWire backend support
- Integration test with real audio device for end-to-end verification
