# AudioOut Playback Implementation Plan

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

**Goal:** Add an `AudioOut` struct to `deli-audio` that provides audio playback via PulseAudio, mirroring the `AudioIn` pattern. `AudioOut` spawns a background blocking task that writes audio to PulseAudio. The caller sends audio data via async `send()`, cancels playback via async `cancel()`, and switches output devices via async `select()`.

**Architecture:** `AudioOut::new()` spawns a `tokio::spawn_blocking` task that connects to PulseAudio with `Direction::Playback` and waits for audio data on an mpsc channel. The caller sends `Vec<i16>` chunks via `send()`, the background task continuously plays all queued data in order, creating a continuous audio stream. The playback loop uses the same double-loop pattern as `AudioIn` for error recovery (reconnect after 100ms on write/connection errors). `cancel()` and `select()` both tear down the current task (drop sender, await handle) and start a new one — `cancel()` keeps the same device, `select()` updates the device. A `start_playback()` helper is shared by `new()`, `cancel()`, and `select()`.

**Interface deviation from AudioIn:** `AudioOut::new()` takes 2 parameters (device, sample_rate) instead of 3. Unlike `AudioIn` which needs a fixed `chunk_frames` to know how many frames to read per PA call, `AudioOut` receives arbitrary-sized buffers via `send()` — the caller controls chunk size. The user explicitly specified `new(device: Option<&str>, sample_rate: u32)` without `chunk_frames`.

**`cancel()` semantics:** `cancel()` means "stop current playback and be immediately ready for new `send()` calls." It flushes the PA buffer, tears down the old task, and starts a fresh playback task. This keeps `AudioOut` always in a ready state — no need to check task existence before calling `send()`.

**Tech Stack:** `libpulse-binding` + `libpulse-simple-binding`, `tokio` (spawn_blocking, mpsc, macros), `log`

## Scope

### In Scope

- `AudioOut` struct with `new()`, `send()`, `cancel()`, `select()`
- Background playback task via `tokio::spawn_blocking`
- Error recovery with stream reconnection (double-loop pattern)
- `sample_rate()` and `device()` getters
- `Debug` and `Drop` implementations
- Update `lib.rs` to export `AudioOut`
- Tests for all methods

### Out of Scope

- `AudioIn` changes (no modifications)
- Output device enumeration (`list_output_devices()`) — deferred
- Changes to `AudioError`, `AudioDevice`, `device.rs`, `error.rs`
- Real audio playback tests (require physical speakers/PulseAudio server)
- Stereo or multi-channel output (mono only, matching AudioIn)
- Any new Cargo.toml dependencies (existing dependencies sufficient)

## Prerequisites

- Existing `deli-audio` crate with `AudioIn`, `AudioError`, etc.
- `tokio` with "rt" and "sync" features (already in Cargo.toml)

## Context for Implementer

- **Patterns to follow:** `AudioIn` in `crates/deli-audio/src/audio_in.rs` is the reference implementation. `AudioOut` mirrors its architecture with reversed data flow: `AudioIn` reads from PulseAudio and pushes to channel; `AudioOut` receives from channel and writes to PulseAudio.
- **Conventions:** Edition 2024, `log` crate for logging, manual error impls (no thiserror), tests in `tests/` directory, `#[tokio::test]` for all tests.
- **Key files:**
  - `crates/deli-audio/src/audio_in.rs` — reference architecture (294 lines). Mirror struct layout, `start_capture()` → `start_playback()`, `capture_loop()` → `playback_loop()`, same double-loop error recovery.
  - `crates/deli-audio/src/lib.rs` — add `pub mod audio_out;` and `pub use audio_out::AudioOut;`
  - `crates/deli-audio/src/error.rs` — existing `AudioError` variants are sufficient (Channel, Stream, Device)
  - `crates/deli-audio/tests/audio_in_tests.rs` — reference test patterns
- **Gotchas:**
  - `tokio::spawn_blocking` must be called from within a tokio runtime context. `new()` panics if called outside tokio.
  - `simple.write()` takes `&[u8]`. Convert `&[i16]` by iterating with `to_ne_bytes()` and flattening.
  - `simple.flush()` wraps `pa_simple_flush()` which discards buffered audio immediately. `simple.drain()` wraps `pa_simple_drain()` which blocks until all buffered audio finishes playing. Use `flush()` when shutting down (channel closed) to stop playback promptly. Both methods return `Result<(), PAErr>`.
  - `blocking_recv()` on `mpsc::Receiver` blocks the thread — this is correct inside `spawn_blocking`.
  - `mpsc::Receiver` does not have `is_closed()`. To check if the sender is dropped before attempting PA connection, use `try_recv()`: `Disconnected` means exit, `Ok(data)` means save for writing, `Empty` means continue.
  - `send()` uses `sender.send(data).await` which provides backpressure — if the channel is full (4 slots), the caller waits until the background task consumes a message.
  - `cancel()` and `select()` are async because they `.await` the old task handle. The old task sees channel closure via `blocking_recv()` returning `None`, then flushes and exits.
  - In `Drop`, we cannot `.await` the task handle. Drop the sender (signals task), drop the handle. Task exits on its own.
  - PulseAudio `Simple::new()` with `Direction::Playback` creates an output stream. The device parameter refers to a PA sink (output), not a source (input).

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: AudioOut struct with new(), send(), basic playback loop, getters, Drop, Debug
- [x] Task 2: cancel() and select() methods
- [x] Task 3: Error recovery with stream reconnection in playback loop

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: AudioOut struct with new(), send(), basic playback loop, getters, Drop, Debug

**Objective:** Create the `AudioOut` struct that starts a background playback task immediately in `new()`, accepts audio data via async `send()`, and provides `sample_rate()` and `device()` getters. The basic playback loop exits on error (no reconnection yet).

**Dependencies:** None

**Files:**

- Create: `crates/deli-audio/src/audio_out.rs`
- Modify: `crates/deli-audio/src/lib.rs`
- Create: `crates/deli-audio/tests/audio_out_tests.rs`

**Key Decisions / Notes:**

- **Struct fields:**
  ```
  sample_rate: u32,
  device: Option<String>,
  sender: Option<mpsc::Sender<Vec<i16>>>,
  task_handle: Option<tokio::task::JoinHandle<()>>,
  ```
- **`new(device: Option<&str>, sample_rate: u32) -> Self`** — stores params, calls `start_playback()` to set sender + handle. No `chunk_frames` parameter — callers send arbitrary-sized buffers.
- **`start_playback()` helper** — creates `mpsc::channel(4)`, calls `tokio::task::spawn_blocking` with `playback_loop`, returns `(Sender, JoinHandle)`. Used by `new()`, and later by `cancel()` and `select()` in Task 2.
- **`send(&mut self, data: &[i16]) -> Result<(), AudioError>`** — async. Copies `data` to `Vec<i16>`, sends via `sender.send(data_vec).await`. Returns `Err(AudioError::Channel)` if sender is None, `Err(AudioError::Stream)` if channel is closed.
- **`playback_loop(device, sample_rate, rx)` basic version** — creates PA `Simple` with `Direction::Playback`, loops: `rx.blocking_recv()` → convert i16 to bytes via `to_ne_bytes()` → `simple.write()`. On write error: log and exit. On `blocking_recv()` returning `None`: call `simple.flush()` and exit.
- **i16 to bytes conversion:** `samples.iter().flat_map(|s| s.to_ne_bytes()).collect::<Vec<u8>>()`
- **`Drop`** — drops sender (take), drops task handle (take). No `.join()` or `.await`.
- **`Debug`** — same pattern as AudioIn: show field names with `is_some()` for Options.
- **`lib.rs`** — add `pub mod audio_out;` and `pub use audio_out::AudioOut;`. Update crate doc comment.
- **Tests** — all `#[tokio::test]`. Test construction with/without device, getters, send signature (type assertion), Drop without panic.

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `AudioOut::new()` accepts 2 parameters: `device` and `sample_rate`
- [ ] `AudioOut::new()` starts playback task immediately (spawn_blocking visible in code)
- [ ] `send()` is async and returns `Result<(), AudioError>`
- [ ] `send()` type assertion test compiles: `fn assert_send_type(_: impl Future<Output = Result<(), AudioError>>) {}`
- [ ] `sample_rate()` and `device()` getters work correctly
- [ ] `Drop` does not call `.join()` — only drops sender and handle
- [ ] `AudioOut` is exported from `lib.rs`

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

### Task 2: cancel() and select() methods

**Objective:** Add async `cancel()` and `select()` methods to `AudioOut`. Both tear down the current playback task and start a new one. `cancel()` keeps the same device; `select()` updates the device.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-audio/src/audio_out.rs`
- Modify: `crates/deli-audio/tests/audio_out_tests.rs`

**Key Decisions / Notes:**

- **`cancel(&mut self)`** — async. Drops sender (signals task to stop), awaits task handle, calls `start_playback()` with current device and sample_rate. This is identical to `select()` but without changing the device.
- **`select(&mut self, device: &str)`** — async. Drops sender, awaits task handle, updates `self.device`, calls `start_playback()` with new device. Identical to `AudioIn::select()`.
- **Shared pattern:** Both methods follow the same 4-step sequence: (1) drop sender via `self.sender.take()`, (2) await `self.task_handle.take()`, (3) optionally update device, (4) `start_playback()` to get new sender + handle.
- **`cancel()` semantics:** The old playback task receives `None` from `blocking_recv()` (sender dropped), calls `simple.flush()` (discards PA buffer), and exits. The new task is immediately ready for fresh `send()` calls.
- **Tests:** Test `select()` updates device name. Test `cancel()` preserves device name. Test that both methods complete without panicking.

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `cancel()` signature is `pub async fn cancel(&mut self)` (infallible, no return value)
- [ ] `cancel()` drops old task, starts new task with same device
- [ ] `select()` is async, drops old task, updates device, starts new task
- [ ] After `cancel()`, `device()` returns the same device as before
- [ ] After `select("new_device")`, `device()` returns `Some("new_device")`
- [ ] After `cancel()` or `select()`, `send()` can be called without error (channel is fresh)

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

### Task 3: Error recovery with stream reconnection in playback loop

**Objective:** Modify the playback loop to automatically recover from PulseAudio errors by closing the stream, waiting 100ms, and reopening, instead of exiting on error.

**Dependencies:** Task 1, Task 2

**Files:**

- Modify: `crates/deli-audio/src/audio_out.rs`
- Modify: `crates/deli-audio/tests/audio_out_tests.rs`

**Key Decisions / Notes:**

- **Double-loop pattern in `playback_loop`:**
  ```
  outer loop {
      // Check channel status via try_recv()
      let pending_data = match rx.try_recv() {
          Err(TryRecvError::Disconnected) => return,
          Ok(data) => Some(data),
          Err(TryRecvError::Empty) => None,
      };

      match Simple::new(..., Direction::Playback, ...) {
          Ok(simple) => {
              // Write pending data if any
              if let Some(samples) = pending_data {
                  convert and write; if Err: break to reconnect
              }
              inner loop {
                  match rx.blocking_recv() {
                      Some(samples) => { convert and write; if Err: break inner }
                      None => { simple.flush(); return; }
                  }
              }
              // simple dropped here (stream closed)
              std::thread::sleep(Duration::from_millis(100));
          }
          Err(e) => {
              log::warn!("Failed to connect to PulseAudio: {}", e);
              std::thread::sleep(Duration::from_millis(100));
          }
      }
  }
  ```
- **Channel status check:** Before each connection attempt, call `rx.try_recv()`. If `Disconnected`, exit (sender dropped). If `Ok(data)`, save it as `pending_data` to write after connecting. If `Empty`, proceed normally.
- **Connection retry:** If `Simple::new()` fails, log and retry after 100ms.
- **Write error recovery:** If `simple.write()` returns `Err`, log, break inner loop. `Simple` is dropped, sleep 100ms, outer loop reconnects. The failed data is lost (acceptable for real-time audio).
- **Logging:** Use `log::warn!` for connection failures and write errors. Use `log::debug!` for any diagnostic messages.
- **Tests:** Add a test verifying `send()` return type is `Result<(), AudioError>` (confirms no double-wrapping). Verify the playback loop structure compiles correctly.

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] Playback loop uses double-loop pattern: outer for connect/reconnect, inner for receiving and writing
- [ ] On `simple.write()` error: stream is dropped, 100ms sleep, then reconnection attempt
- [ ] On `Simple::new()` error: 100ms sleep, then retry
- [ ] Channel status is checked via `try_recv()` before each connection attempt (allows clean shutdown)
- [ ] `log::warn!` is used for all recoverable errors
- [ ] `simple.flush()` is called when channel closes (blocking_recv returns None) before exiting

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

## Testing Strategy

- **Unit tests:** `AudioOut` construction with 2-parameter API, getter verification (`sample_rate()`, `device()`), `send()` return type assertion, `select()` updates device name, `cancel()` preserves device name, `Drop` completes without panic. All tests require `#[tokio::test]`.
- **Integration tests:** Structural tests — verify `send()`, `cancel()`, `select()` are async, verify playback task is spawned immediately on construction. All in `tests/` directory.
- **Manual verification:** Real audio testing is deferred. If PulseAudio is available, create `AudioOut::new(None, 16000)` and call `send()` with a sine wave to verify audio output.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `new()` called outside tokio runtime panics | Medium | High | All tests use `#[tokio::test]`. `new()` doc comment has `# Panics` section stating it must be called from within a tokio runtime. |
| `blocking_recv()` blocks indefinitely when no data is sent | Low | Low | This is expected behavior — the playback task waits for data. When the sender is dropped (via Drop, cancel, or select), `blocking_recv()` returns `None` and the task exits after calling `flush()`. |
| `send()` blocks caller when channel is full (4 slots) | Medium | Low | Channel capacity of 4 provides a small buffer. If the PA write is slower than the caller's send rate, the caller's `send().await` waits (backpressure). This is intentional — prevents unbounded memory growth. Callers should send at approximately real-time rate. |
| PulseAudio write fails and data is lost during reconnection | Medium | Low | Failed write data is discarded. For real-time audio playback, retrying old data would introduce latency. The reconnected stream starts fresh with the next `send()` data. This is acceptable for live audio applications. |
| `cancel()` recreates PA connection — adds latency before next `send()` | Low | Low | PA `Simple::new()` typically completes in < 10ms. The old task is awaited (it flushes and exits quickly since `blocking_recv()` returns `None` immediately). Total cancel latency is bounded by one PA connection time. |

## Open Questions

- None — requirements are clear.

### Deferred Ideas

- `list_output_devices()` function for enumerating PulseAudio sinks
- Stereo/multi-channel output support
- Configurable channel buffer size
- `drain()` method to play remaining buffered audio before stopping
- Volume control via PulseAudio stream properties
