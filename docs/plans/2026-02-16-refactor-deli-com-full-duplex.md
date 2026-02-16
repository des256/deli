# Refactor deli-com to Full Duplex Implementation Plan

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

**Goal:** Rename `SenderServer` to `Server` and `ReceiverClient` to `Client` in deli-com, and make both structs full-duplex (Server gets `recv`, Client gets `send`). After refactoring, Server/Client should be the Codec-typed wrappers around the same TCP primitives as TcpServer/TcpClient.

**Architecture:** Server stores both `OwnedReadHalf` and `OwnedWriteHalf` per client (currently only stores write half). Client stores both halves (currently only stores read half). The framing module already has both `read_message` and `write_message` — these just need to be wired to the new methods. All downstream consumers (camera-viewer experiment) and tests are updated to use the new names.

**Tech Stack:** Rust, tokio (async TCP), deli-codec (Codec trait).

## Scope

### In Scope

- Rename `SenderServer` → `Server` in `crates/deli-com/src/sender.rs` (rename file to `server.rs`)
- Rename `ReceiverClient` → `Client` in `crates/deli-com/src/receiver.rs` (rename file to `client.rs`)
- Add `recv()` to `Server` — receive a message from any connected client
- Add `send()` to `Client` — send a message to the server
- Update `lib.rs` re-exports
- Update all tests in `crates/deli-com/tests/`
- Update consumers: `experiments/camera-viewer/src/camera.rs`, `experiments/camera-viewer/src/viewer.rs`

### Out of Scope

- Changing the framing protocol (length-prefix stays the same)
- Changing error types
- Adding new connection management features
- Modifying the accept loop behavior

## Prerequisites

- deli-codec crate with Codec trait (exists)
- tokio with TCP support (exists)

## Context for Implementer

- **Patterns to follow:** The existing `framing::write_message` and `framing::read_message` in `crates/deli-com/src/framing.rs` are the low-level I/O primitives. Server's `send()` already calls `write_message`; the new `recv()` should call `read_message` similarly.
- **Conventions:** Edition 2024, async methods, `ComError` for all errors, `log` crate for warnings.
- **Key files:**
  - `crates/deli-com/src/sender.rs` → becomes `server.rs` — `SenderServer<T>` with `bind()`, `send()`, `client_count()`, `local_addr()`, `Drop`
  - `crates/deli-com/src/receiver.rs` → becomes `client.rs` — `ReceiverClient<T>` with `connect()`, `recv()`
  - `crates/deli-com/src/framing.rs` — `write_message()` and `read_message()` (unchanged)
  - `crates/deli-com/src/error.rs` — `ComError` enum (unchanged)
  - `crates/deli-com/src/lib.rs` — module declarations and re-exports
- **Gotchas:**
  - Server currently only stores `OwnedWriteHalf` per client (discards read half at line 35: `let (_, write_half) = stream.into_split()`). For `recv()`, it must also store `OwnedReadHalf`.
  - Client currently only stores `OwnedReadHalf` (discards write half at line 18: `let (read_half, _) = stream.into_split()`). For `send()`, it must also store `OwnedWriteHalf`.
  - Server's `recv()` should receive from any client (first available message). This requires reading from multiple read halves concurrently — use `tokio::select!` or iterate clients.
  - Server's `send()` is a broadcast to all clients. This semantics should be preserved.

## Feature Inventory

### Files Being Modified

| Old File | Functions/Methods | Mapped to Task |
| --- | --- | --- |
| `crates/deli-com/src/sender.rs` | `SenderServer::bind()`, `send()`, `client_count()`, `local_addr()`, `Drop` | Task 1 |
| `crates/deli-com/src/receiver.rs` | `ReceiverClient::connect()`, `recv()` | Task 2 |
| `crates/deli-com/src/lib.rs` | module decls, re-exports | Task 1, Task 2 |
| `crates/deli-com/tests/sender_tests.rs` | sender tests | Task 3 |
| `crates/deli-com/tests/receiver_tests.rs` | receiver tests | Task 3 |
| `crates/deli-com/tests/integration_tests.rs` | integration tests | Task 3 |
| `experiments/camera-viewer/src/camera.rs` | uses `SenderServer` | Task 4 |
| `experiments/camera-viewer/src/viewer.rs` | uses `ReceiverClient` | Task 4 |

### Feature Mapping Verification

- [x] All old files listed above
- [x] All functions/methods identified
- [x] Every feature has a task number
- [x] No features accidentally omitted

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Refactor Server (rename + add recv)
- [x] Task 2: Refactor Client (rename + add send)
- [x] Task 3: Update deli-com tests
- [x] Task 4: Update camera-viewer consumers

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Refactor Server (rename + add recv)

**Objective:** Rename `SenderServer` to `Server`, rename `sender.rs` to `server.rs`, store both read and write halves per client, and add `recv()` method.

**Dependencies:** None

**Files:**

- Rename: `crates/deli-com/src/sender.rs` → `crates/deli-com/src/server.rs`
- Modify: `crates/deli-com/src/lib.rs`

**Key Decisions / Notes:**

- Rename struct `SenderServer<T>` → `Server<T>`
- Change clients map type from `HashMap<SocketAddr, OwnedWriteHalf>` to `HashMap<SocketAddr, (OwnedReadHalf, OwnedWriteHalf)>` to store both halves
- Accept loop (line 34-35): change `let (_, write_half) = stream.into_split()` to `let (read_half, write_half) = stream.into_split()` and store both
- Existing `send()` method: update to access `.1` (write half) from the tuple
- New `recv()` method: iterate clients, try `read_message` from each client's read half (`.0`). Return the first successfully decoded message. Remove disconnected clients. Use a simple sequential check — for broadcast patterns, this is sufficient.
- Update `lib.rs`: change `pub mod sender` → `pub mod server`, change `pub use sender::SenderServer` → `pub use server::Server`
- Keep `client_count()`, `local_addr()`, and `Drop` unchanged (just update struct name references)

**Definition of Done:**

- [ ] `Server<T>` struct replaces `SenderServer<T>`
- [ ] `Server::bind()` stores both read and write halves per client
- [ ] `Server::send()` broadcasts to all clients (same behavior as before)
- [ ] `Server::recv()` receives a message from any connected client
- [ ] `Server::client_count()` and `local_addr()` still work
- [ ] `pub use server::Server` in lib.rs
- [ ] `cargo check -p deli-com` — no errors

**Verify:**

- `cargo check -p deli-com` — no errors

### Task 2: Refactor Client (rename + add send)

**Objective:** Rename `ReceiverClient` to `Client`, rename `receiver.rs` to `client.rs`, store both read and write halves, and add `send()` method.

**Dependencies:** None

**Files:**

- Rename: `crates/deli-com/src/receiver.rs` → `crates/deli-com/src/client.rs`
- Modify: `crates/deli-com/src/lib.rs`

**Key Decisions / Notes:**

- Rename struct `ReceiverClient<T>` → `Client<T>`
- Store both halves: add `writer: OwnedWriteHalf` field alongside existing `reader: OwnedReadHalf`
- Connect (line 18): change `let (read_half, _) = stream.into_split()` to `let (read_half, write_half) = stream.into_split()` and store both
- Existing `recv()`: unchanged, still uses `self.reader`
- New `send(&self, value: &T)`: call `framing::write_message(&mut self.writer, value).await`
- Note: `send()` needs `&mut self` since `write_message` needs `&mut W`
- Update `lib.rs`: change `pub mod receiver` → `pub mod client`, change `pub use receiver::ReceiverClient` → `pub use client::Client`

**Definition of Done:**

- [ ] `Client<T>` struct replaces `ReceiverClient<T>`
- [ ] `Client::connect()` stores both read and write halves
- [ ] `Client::recv()` receives messages (same behavior as before)
- [ ] `Client::send()` sends a message to the server
- [ ] `pub use client::Client` in lib.rs
- [ ] `cargo check -p deli-com` — no errors

**Verify:**

- `cargo check -p deli-com` — no errors

### Task 3: Update deli-com tests

**Objective:** Update all test files to use the new `Server`/`Client` names and add tests for the new `recv` (Server) and `send` (Client) methods.

**Dependencies:** Task 1, Task 2

**Files:**

- Modify: `crates/deli-com/tests/sender_tests.rs` (rename to `server_tests.rs`)
- Modify: `crates/deli-com/tests/receiver_tests.rs` (rename to `client_tests.rs`)
- Modify: `crates/deli-com/tests/integration_tests.rs`

**Key Decisions / Notes:**

- Replace all `SenderServer` → `Server` and `ReceiverClient` → `Client` in test files
- Add test for `Server::recv()`: bind server, connect client, client sends message, server receives it
- Add test for `Client::send()`: bind server, connect client, client sends message, verify server receives it (same test can cover both)
- Rename test files to match new module names
- Existing test patterns should be preserved — just update the type names

**Definition of Done:**

- [ ] All test files use `Server` and `Client` names
- [ ] Test for `Server::recv()` — server receives message sent by client
- [ ] Test for `Client::send()` — client sends message to server
- [ ] `cargo test -p deli-com -q` — all tests pass

**Verify:**

- `cargo test -p deli-com -q` — all tests pass

### Task 4: Update camera-viewer consumers

**Objective:** Update the camera-viewer experiment to use the new `Server`/`Client` names.

**Dependencies:** Task 1, Task 2

**Files:**

- Modify: `experiments/camera-viewer/src/camera.rs`
- Modify: `experiments/camera-viewer/src/viewer.rs`

**Key Decisions / Notes:**

- In `camera.rs`: replace `use deli_com::SenderServer` → `use deli_com::Server`, replace `SenderServer::<Frame>::bind` → `Server::<Frame>::bind`
- In `viewer.rs`: replace `use deli_com::ReceiverClient` → `use deli_com::Client`, replace `ReceiverClient::<Frame>::connect` → `Client::<Frame>::connect`
- No behavior changes — just name updates

**Definition of Done:**

- [ ] `camera.rs` uses `Server` instead of `SenderServer`
- [ ] `viewer.rs` uses `Client` instead of `ReceiverClient`
- [ ] `cargo check -p camera-viewer` — no errors

**Verify:**

- `cargo check -p camera-viewer` — no errors

## Testing Strategy

- **Unit tests:** Server recv and Client send roundtrip in deli-com tests
- **Integration tests:** Full server-client communication with both send and recv in both directions
- **Manual verification:** Run camera + viewer to confirm the rename didn't break anything

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Server recv blocks waiting for a specific client | Medium | Medium | Implement recv as iterating over all clients' read halves, returning first available message. Skip disconnected clients. |
| Breaking downstream consumers | Low | Low | Only 2 consumer files (camera.rs, viewer.rs) — straightforward name replacement. Task 4 covers this. |
| Test files reference old names | Low | Low | Task 3 explicitly renames and updates all test files. |

## Open Questions

- None — design is straightforward.

### Deferred Ideas

- Per-client recv (receive from a specific client by address)
- Message routing (server forwards messages between clients)
- Connection events (callbacks on client connect/disconnect)
