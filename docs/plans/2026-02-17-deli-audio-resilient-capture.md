# Deli-Audio Resilient Capture Implementation Plan

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

**Goal:** Refactor `deli-audio`'s `AudioIn` to: (1) start capture immediately in `new()` instead of lazily on first `recv()`, (2) automatically recover from PulseAudio stream errors by closing the stream, waiting 100ms, and reopening, (3) accept a configurable chunk size in frames, and (4) use `tokio::spawn_blocking` instead of `std::thread::spawn` for tokio runtime compatibility.

**Architecture:** `AudioIn::new()` immediately spawns a blocking task via `tokio::spawn_blocking` that runs the PulseAudio capture loop. The capture loop uses a double-loop pattern: an outer loop that handles connection/reconnection, and an inner loop that reads audio chunks. On read error, the inner loop breaks, the `Simple` stream is dropped, the task sleeps 100ms, and the outer loop reconnects. The channel carries `Vec<i16>` directly (not `Result`) since errors are handled internally with retry. `select()` becomes async to cleanly await the old task before restarting on the new device.

**Tech Stack:** `libpulse-binding` + `libpulse-simple-binding`, `tokio` (spawn_blocking, mpsc, macros), `log`

## Scope

### In Scope

- Refactor `AudioIn` struct fields (remove `JoinHandle` from std, use tokio's)
- Change `new()` to start capture immediately with `tokio::spawn_blocking`
- Add `chunk_frames: u32` parameter to `new()` and a `chunk_frames()` getter
- Remove `ensure_started()` entirely
- Simplify `recv()` (no lazy start logic)
- Change channel type from `Result<Vec<i16>, AudioError>` to `Vec<i16>`
- Implement error recovery in capture loop (drop stream, sleep 100ms, reconnect)
- Make `select()` async (await old task handle before restarting)
- Update `Drop` impl for tokio task handles (no sync join)
- Update all existing tests for new API

### Out of Scope

- `AudioOut` (deferred)
- Changes to `AudioError`, `AudioDevice`, `list_devices()`, `device.rs`, `error.rs`
- Real audio capture tests (require physical microphone)
- Any new Cargo.toml dependencies (existing `tokio` features sufficient)

## Prerequisites

- Existing `deli-audio` crate with all 4 original tasks completed and verified
- `tokio` with "rt" and "sync" features (already in Cargo.toml)

## Context for Implementer

- **Patterns to follow:** The current `AudioIn` in `crates/deli-audio/src/audio_in.rs` is the sole file being refactored. The V4l2Camera at `crates/deli-camera/src/v4l2.rs` uses the same background-thread pattern but is NOT being changed to tokio — only AudioIn is.
- **Conventions:** Edition 2024, `log` crate for logging, manual error impls (no thiserror), tests in `tests/` directory.
- **Key files:**
  - `crates/deli-audio/src/audio_in.rs` — the file being refactored (263 lines currently)
  - `crates/deli-audio/tests/audio_in_tests.rs` — tests to update
  - `crates/deli-audio/Cargo.toml` — may need `"time"` feature added to tokio dev-dependencies for `tokio::time::timeout` in tests
- **Gotchas:**
  - `tokio::spawn_blocking` must be called from within a tokio runtime context. `new()` remains sync but panics if called outside tokio. All tests must use `#[tokio::test]`.
  - `tokio::task::JoinHandle::abort()` has no effect on `spawn_blocking` tasks — cancellation is only via channel closure (receiver dropped → `try_send` returns `Closed` → task exits).
  - In `Drop`, we cannot `.await` the task handle. Instead, drop the receiver (signals the task to stop) and drop the handle. The task exits on its own when it next checks the channel.
  - `std::thread::sleep` (not `tokio::time::sleep`) must be used inside `spawn_blocking` since it runs on a blocking thread, not an async context.
  - Since the capture loop retries indefinitely on errors, `new()` never fails even with no PulseAudio server. Errors are only visible through `log::warn!`. `recv()` only returns `Err` if the channel closes (task panic or runtime shutdown).
  - `select()` must be async to `.await` the old task handle, ensuring the old `Simple` stream is fully dropped before the new one is created on the (potentially same) device.

## Feature Inventory

### Files Being Modified

| File | Functions/Features | Status |
| --- | --- | --- |
| `crates/deli-audio/src/audio_in.rs` | `AudioIn` struct, `new()`, `recv()`, `select()`, `ensure_started()`, `capture_loop()`, `Drop`, `Debug`, `type AudioResult` | All mapped |
| `crates/deli-audio/tests/audio_in_tests.rs` | 6 test functions | All mapped |

### Feature Mapping

| Old Feature | New Location | Task # |
| --- | --- | --- |
| `AudioIn` struct (fields) | Same file, new fields | Task 1 |
| `AudioIn::new(device, sample_rate)` | `AudioIn::new(device, sample_rate, chunk_frames)` | Task 1 |
| `AudioIn::recv()` | Same, simplified (no ensure_started) | Task 1 |
| `AudioIn::sample_rate()` | Preserved as-is | Task 1 |
| `AudioIn::device()` | Preserved as-is | Task 1 |
| `AudioIn::chunk_frames()` | NEW getter | Task 1 |
| `AudioIn::select()` | Becomes async | Task 1 |
| `AudioIn::ensure_started()` | REMOVED (replaced by immediate start in new()) | Task 1 |
| `AudioIn::capture_loop()` | New signature + error recovery | Task 1 + Task 2 |
| `type AudioResult` | REMOVED (channel carries `Vec<i16>` directly) | Task 1 |
| `impl Drop` | Simplified for tokio handles | Task 1 |
| `impl Debug` | Updated for new fields | Task 1 |
| `start_capture()` | NEW helper method | Task 1 |
| Error recovery (reconnect) | NEW behavior in capture_loop | Task 2 |
| All 6 tests | Updated for new API | Task 1 + Task 2 |

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Refactor AudioIn for immediate start, configurable chunks, and tokio spawn_blocking
- [x] Task 2: Implement error recovery with stream reconnection

**Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Refactor AudioIn for immediate start, configurable chunks, and tokio spawn_blocking

**Objective:** Restructure `AudioIn` to start capture immediately in `new()` using `tokio::spawn_blocking`, add configurable `chunk_frames` parameter, make `select()` async, simplify `recv()`, and update the `Drop` impl for tokio compatibility.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-audio/src/audio_in.rs`
- Modify: `crates/deli-audio/tests/audio_in_tests.rs`

**Key Decisions / Notes:**

- **Struct fields change:**
  ```
  sample_rate: u32,
  chunk_frames: u32,          // NEW
  device: Option<String>,
  receiver: Option<mpsc::Receiver<Vec<i16>>>,    // Was Result<Vec<i16>, AudioError>
  task_handle: Option<tokio::task::JoinHandle<()>>,  // Was std::thread::JoinHandle<()>
  ```
- **`new(device: Option<&str>, sample_rate: u32, chunk_frames: u32) -> Self`** — stores params, calls `start_capture()` to set receiver + handle. Must be called from tokio runtime context.
- **`start_capture()` helper** — creates mpsc channel(4), calls `tokio::task::spawn_blocking` with `capture_loop`, returns `(Receiver, JoinHandle)`. Both `new()` and `select()` use this.
- **`recv(&mut self) -> Result<Vec<i16>, AudioError>`** — reads from receiver directly. No `ensure_started()` call. Returns `Err(AudioError::Channel)` if receiver is None, `Err(AudioError::Stream("capture task terminated"))` if channel closes.
- **`select(&mut self, device: &str)` becomes async** — drops receiver (signals task), awaits old task handle, updates device, calls `start_capture()`.
- **`capture_loop` signature changes** to `capture_loop(device, sample_rate, chunk_frames, tx: mpsc::Sender<Vec<i16>>)`. In this task, the capture loop still exits on error (like current behavior). Error recovery is added in Task 2.
- **Buffer size** uses `chunk_frames` instead of `sample_rate / 10`: `bytes_per_chunk = chunk_frames as usize * 2`.
- **`Drop`** — drops receiver (take), drops task handle (take). No `.join()` or `.await`. Task exits on its own when channel closes.
- **`Debug`** — update to include `chunk_frames` field.
- **Remove** `ensure_started()` and `type AudioResult`.
- **All tests become `#[tokio::test]`** since `new()` requires tokio context. Update call sites for `new(device, sample_rate, chunk_frames)`. `select()` calls need `.await`.
- **Doc examples** update to reflect new `new()` signature with `chunk_frames`.

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] `AudioIn::new()` accepts 3 parameters: `device`, `sample_rate`, `chunk_frames`
- [ ] `AudioIn::new()` starts capture immediately (no `ensure_started()` exists)
- [ ] `recv()` does not call any start/initialization method
- [ ] `select()` is async and awaits the old task handle
- [ ] Channel type is `Vec<i16>` (not `Result<Vec<i16>, AudioError>`)
- [ ] `chunk_frames()` getter returns the configured chunk frame count
- [ ] `Drop` does not call `.join()` — only drops receiver and handle
- [ ] Capture task uses `tokio::task::spawn_blocking` (not `std::thread::spawn`)

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

### Task 2: Implement error recovery with stream reconnection

**Objective:** Modify the capture loop to automatically recover from PulseAudio errors by closing the stream, waiting 100ms, and reopening it, instead of exiting on error.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-audio/src/audio_in.rs`
- Modify: `crates/deli-audio/tests/audio_in_tests.rs`

**Key Decisions / Notes:**

- **Double-loop pattern in `capture_loop`:**
  ```
  outer loop {
      if tx.is_closed() { break; }
      match Simple::new(...) {
          Ok(simple) => {
              inner loop {
                  match simple.read(&mut buffer) {
                      Ok(()) => { convert and try_send; if Closed: return; }
                      Err(e) => { log::warn!("..."); break inner; }
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
- **Connection retry:** If `Simple::new()` fails (PulseAudio server down, device unavailable), log and retry after 100ms. Check `tx.is_closed()` before each retry to allow clean shutdown.
- **Read error recovery:** If `Simple::read()` returns `Err`, log the error, break the inner loop. The `Simple` is dropped (closing the PA stream). Sleep 100ms. Outer loop reconnects.
- **Spec validation:** Move spec creation and validation before the outer loop (it's static). Buffer allocation also stays before the outer loop.
- **Logging:** Use `log::warn!` for connection failures and read errors. These are recoverable events, not panics.
- **No errors sent through channel:** Since all errors are retried, the channel only carries successful audio chunks. `recv()` never sees an `AudioError` from the capture loop — only from channel closure.
- **Tests:** Add a test verifying that `recv()` return type is `Result<Vec<i16>, AudioError>` (not `Result<Result<...>>`) to confirm the channel type is correct. Verify the capture loop structure compiles and the spec/buffer are created correctly.

**Definition of Done:**

- [ ] All tests pass (unit, integration if applicable)
- [ ] No diagnostics errors (linting, type checking)
- [ ] Capture loop uses double-loop pattern: outer for connect/reconnect, inner for reading
- [ ] On `Simple::read()` error: stream is dropped, 100ms sleep, then reconnection attempt
- [ ] On `Simple::new()` error: 100ms sleep, then retry
- [ ] `tx.is_closed()` is checked before each connection attempt (allows clean shutdown)
- [ ] `log::warn!` is used for all recoverable errors
- [ ] No errors are sent through the mpsc channel (only `Vec<i16>` chunks)

**Verify:**

- `cargo test -p deli-audio` — all tests pass
- `cargo check -p deli-audio` — no compiler errors or warnings

## Testing Strategy

- **Unit tests:** `AudioIn` construction with new 3-parameter API, getter verification (`sample_rate()`, `device()`, `chunk_frames()`), `select()` updates device name, `Drop` completes without panic. All tests require `#[tokio::test]` since `new()` uses `tokio::spawn_blocking`.
- **Integration tests:** Structural tests — verify `recv()` return type, verify `select()` is async, verify capture task is spawned immediately on construction. All in `tests/` directory.
- **Manual verification:** Real audio testing is deferred. If PulseAudio is available, create `AudioIn::new(None, 16000, 1600)` and call `recv()` in a loop to verify chunks arrive.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `new()` called outside tokio runtime panics | Medium | High | `tokio::task::spawn_blocking` panics if called without a runtime. Add `# Panics` section to `new()` doc comment stating it must be called from within a tokio runtime. All tests use `#[tokio::test]` to ensure runtime is present. |
| Capture task blocked on `Simple::read()` during `select()` delays device switch | Low | Low | `select()` awaits the old `tokio::task::JoinHandle`. `Simple::read()` blocks for at most one chunk duration (`chunk_frames / sample_rate` seconds). The await completes when the read finishes and the task sees the closed channel. |
| Infinite retry loop consumes CPU when PulseAudio is unavailable | Medium | Low | Capture loop calls `std::thread::sleep(Duration::from_millis(100))` between each retry, limiting to ~10 attempts/sec. `tx.is_closed()` is checked at the top of each outer loop iteration; when the receiver is dropped (via `Drop` or `select()`), the loop exits within one iteration. |
| Tests spawn background tasks that retry PulseAudio connection | Medium | Low | Tests drop `AudioIn` promptly after checking getters. The capture task sees `tx.is_closed() == true` and exits. Tokio test runtime shuts down cleanly because `spawn_blocking` tasks complete when their work finishes. |
| `&mut self` on `recv()` and `select()` prevents concurrent calls | N/A | N/A | This is by design, not a risk. `AudioIn` is not `Sync`. The `&mut self` requirement enforces single-owner semantics at compile time. Users needing shared access should use a dedicated task that owns the `AudioIn` and communicates via channels. |

## Open Questions

- None — requirements are clear.

### Deferred Ideas

- `AudioOut` struct for audio playback
- Configurable retry delay (currently hardcoded at 100ms)
- Maximum retry count option (currently infinite)
- Exponential backoff for connection retries
