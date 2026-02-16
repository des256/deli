# deli-com Crate Implementation Plan

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

**Goal:** Create a new `deli-com` crate providing TCP-based communication primitives, starting with a Multiple Consumer Single Producer (MCSP) broadcast pattern.

**Architecture:** Two core structs — `SenderServer<T: Codec>` binds a TCP listener, accepts client connections in a background tokio task, and broadcasts messages to all connected clients via `send()`. `ReceiverClient<T: Codec>` connects to a sender server and receives messages via an async `recv()`. Wire encoding uses the existing `deli-codec::Codec` trait with a 4-byte little-endian length prefix framing protocol.

**Tech Stack:** Rust (edition 2024), tokio (TCP + sync + spawn), deli-codec for serialization.

## Scope

### In Scope

- New crate `crates/deli-com` with `Cargo.toml`
- Error types for communication failures (`ComError` enum)
- Length-prefixed framing: 4-byte LE length header + Codec-encoded payload
- `SenderServer<T>` — TCP listener, connection acceptance loop, broadcast `send(T)`
- `ReceiverClient<T>` — TCP client, async `recv() -> Result<T, ComError>`
- Disconnected clients silently removed with `log` warning during broadcast
- Unit and integration tests

### Out of Scope

- Other communication patterns (pub/sub, request/response, etc.)
- TLS / encryption
- Authentication / authorization
- Backpressure or flow control
- UDP transport
- Reconnection logic in `ReceiverClient`

## Prerequisites

- `deli-codec` crate (already exists at `crates/deli-codec`)
- tokio runtime (already used by `deli-camera`)

## Context for Implementer

- **Patterns to follow:** Error enum style from `crates/deli-camera/src/error.rs:1-34` (Display + Error impls, From conversions). Test file layout from `crates/deli-codec/tests/primitive_tests.rs` (separate `tests/` directory, `_tests.rs` suffix).
- **Conventions:** Edition 2024, no `#[cfg(test)]` inline modules — tests go in `tests/` directory. Crate names use `deli-` prefix. Pub re-exports from `lib.rs`.
- **Key files:**
  - `crates/deli-codec/src/lib.rs` — `Codec` trait definition (`encode`, `decode`, `to_bytes`, `from_bytes`)
  - `crates/deli-camera/src/error.rs` — error pattern to follow
  - `crates/deli-camera/Cargo.toml` — tokio dependency pattern
- **Gotchas:** `Codec::decode` takes `(buf: &[u8], pos: &mut usize)` — the `pos` parameter is an in-out cursor. For TCP framing, after reading a length-prefixed message into a buffer, decode from `pos=0`.
- **Domain context:** MCSP = Multiple Consumer, Single Producer. One server produces messages, many clients consume them. This is the reverse of Rust's standard `mpsc` channel — hence "mcsp". The server doesn't receive data from clients; clients don't send data to the server.

## Wire Protocol

Messages are framed as: `[4-byte LE u32 length][payload bytes]`

- Sender encodes `T` via `Codec::to_bytes()`, writes length as `u32` LE, then writes payload
- Receiver reads 4 bytes for length, reads that many payload bytes, decodes via `Codec::from_bytes()`
- Maximum message size: 64 MB (`MAX_MESSAGE_SIZE` constant). `read_message` rejects lengths exceeding this to prevent memory exhaustion from malformed/malicious frames.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Crate scaffold and error types
- [x] Task 2: Length-prefixed framing helpers
- [x] Task 3: SenderServer implementation
- [x] Task 4: ReceiverClient implementation
- [x] Task 5: Integration tests (sender + receiver together)

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Crate scaffold and error types

**Objective:** Create the `deli-com` crate with its `Cargo.toml`, `lib.rs`, and `ComError` error type.

**Dependencies:** None

**Files:**

- Create: `crates/deli-com/Cargo.toml`
- Create: `crates/deli-com/src/lib.rs`
- Create: `crates/deli-com/src/error.rs`
- Test: `crates/deli-com/tests/error_tests.rs`

**Key Decisions / Notes:**

- `Cargo.toml` dependencies: `deli-codec` (path), `tokio` (version "1", features: `rt`, `net`, `sync`, `io-util`, `macros`), `log` (version "0.4")
- Dev-dependencies: `tokio` with `rt-multi-thread` and `macros` features
- `ComError` variants: `Io(std::io::Error)`, `Decode(deli_codec::DecodeError)`, `ConnectionClosed`, `MessageTooLarge(u32)` (carries the offending length)
- Implement `Display`, `Error`, `From<std::io::Error>`, `From<deli_codec::DecodeError>` for `ComError`
- `lib.rs` declares `mod error;` and pub-uses `ComError`

**Definition of Done:**

- [ ] `cargo check -p deli-com` succeeds with no errors
- [ ] `ComError` has `Io`, `Decode`, `ConnectionClosed`, and `MessageTooLarge` variants
- [ ] `From<std::io::Error>` and `From<DecodeError>` conversions work
- [ ] Error Display messages are descriptive

**Verify:**

- `cargo test -p deli-com -q` — tests pass
- `cargo check -p deli-com` — no errors

### Task 2: Length-prefixed framing helpers

**Objective:** Implement async helper functions for writing and reading length-prefixed Codec messages over `tokio::io::AsyncWrite` / `AsyncRead` streams.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-com/src/framing.rs`
- Modify: `crates/deli-com/src/lib.rs` (add `mod framing;`)
- Test: `crates/deli-com/tests/framing_tests.rs`

**Key Decisions / Notes:**

- `pub async fn write_message<T: Codec, W: AsyncWriteExt + Unpin>(writer: &mut W, value: &T) -> Result<(), ComError>` — encode `T` to bytes, write `u32` LE length, write payload
- `pub async fn read_message<T: Codec, R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<T, ComError>` — read 4-byte length, validate against `MAX_MESSAGE_SIZE`, read payload, decode `T`
- `pub const MAX_MESSAGE_SIZE: u32 = 64 * 1024 * 1024;` — 64 MB max message size. `read_message` returns `ComError::MessageTooLarge` if length exceeds this.
- When `read_exact` returns 0 bytes (EOF), return `ComError::ConnectionClosed`
- Partial writes leave the stream in a corrupted state — the receiver must close the connection on any framing error. This is inherent to length-prefixed protocols over TCP.
- Use `tokio::io::{AsyncReadExt, AsyncWriteExt}` for `read_exact` and `write_all`
- Test with `tokio::io::duplex` to create an in-memory bidirectional stream

**Definition of Done:**

- [ ] `write_message` encodes T and writes length-prefixed frame
- [ ] `read_message` reads length-prefixed frame and decodes T
- [ ] EOF during read returns `ComError::ConnectionClosed`
- [ ] Test exists that simulates EOF (drop writer) and asserts `ComError::ConnectionClosed`
- [ ] Test exists that sends a length exceeding `MAX_MESSAGE_SIZE` and asserts `ComError::MessageTooLarge`
- [ ] Round-trip test: write then read recovers the original value

**Verify:**

- `cargo test -p deli-com -q` — all tests pass

### Task 3: SenderServer implementation

**Objective:** Implement `SenderServer<T>` that listens for TCP connections and broadcasts messages to all connected clients.

**Dependencies:** Task 2

**Files:**

- Create: `crates/deli-com/src/sender.rs`
- Modify: `crates/deli-com/src/lib.rs` (add `mod sender;`, pub-use `SenderServer`)
- Test: `crates/deli-com/tests/sender_tests.rs`

**Key Decisions / Notes:**

- `SenderServer<T>` fields:
  - `clients: Arc<tokio::sync::RwLock<HashMap<SocketAddr, OwnedWriteHalf>>>` — map of connected client write halves (tokio Mutex, not std, to hold across await points)
  - `_accept_task: JoinHandle<()>` — background task handle for accepting connections
- Constructor: `pub async fn bind(addr: impl ToSocketAddrs) -> Result<Self, ComError>` — binds a `TcpListener`, spawns an accept loop that adds new client write halves to the shared map
- `pub async fn send(&self, value: &T) -> Result<(), ComError>` where `T: Codec` — locks the mutex, iterates over all clients writing the message via `write_message`. Collects `SocketAddr`s of failed clients. After iteration, removes failed clients from the map and logs a warning per removal via `log::warn!`. Returns `Ok(())` (broadcast errors don't fail the method). Note: `tokio::io::AsyncWriteExt::write_all` handles `WouldBlock`/`Interrupted` internally via tokio's async reactor — no special handling needed for those error kinds.
- `pub fn client_count(&self) -> usize` — returns current number of connected clients (useful for testing)
- Accept loop error handling: on `accept()` error, log `warn!` and continue the loop (transient errors like EMFILE). The loop only exits if the `TcpListener` is dropped.
- Use `tokio::net::{TcpListener, TcpStream}` and `TcpStream::into_split()` for write half
- The read half from accepted connections is dropped (clients don't send data to the server in MCSP)

**Definition of Done:**

- [ ] `SenderServer::bind` creates a listening server
- [ ] Accept loop adds new clients to the internal map
- [ ] `send()` broadcasts to all connected clients
- [ ] Disconnected clients are removed during `send()` with a log warning
- [ ] `client_count()` reflects current connections

**Verify:**

- `cargo test -p deli-com -q` — all tests pass
- `cargo check -p deli-com` — no errors

### Task 4: ReceiverClient implementation

**Objective:** Implement `ReceiverClient<T>` that connects to a `SenderServer` and receives messages via async `recv()`.

**Dependencies:** Task 2

**Files:**

- Create: `crates/deli-com/src/receiver.rs`
- Modify: `crates/deli-com/src/lib.rs` (add `mod receiver;`, pub-use `ReceiverClient`)
- Test: `crates/deli-com/tests/receiver_tests.rs`

**Key Decisions / Notes:**

- `ReceiverClient<T>` fields:
  - `reader: OwnedReadHalf` — the read half of the TCP connection
  - `_marker: PhantomData<T>` — for the generic type parameter
- Constructor: `pub async fn connect(addr: impl ToSocketAddrs) -> Result<Self, ComError>` — connects via `TcpStream::connect`, splits, stores read half, drops write half
- `pub async fn recv(&mut self) -> Result<T, ComError>` where `T: Codec` — calls `read_message` on the internal reader
- When the server closes the connection, `recv()` returns `Err(ComError::ConnectionClosed)`

**Definition of Done:**

- [ ] `ReceiverClient::connect` establishes TCP connection
- [ ] `recv()` returns decoded `T` values from the server
- [ ] Server disconnect returns `ComError::ConnectionClosed`

**Verify:**

- `cargo test -p deli-com -q` — all tests pass

### Task 5: Integration tests (sender + receiver together)

**Objective:** End-to-end integration tests proving the full MCSP broadcast pattern works with real TCP connections.

**Dependencies:** Task 3, Task 4

**Files:**

- Create: `crates/deli-com/tests/integration_tests.rs`

**Key Decisions / Notes:**

- Test scenarios:
  1. Single sender, single receiver: send a message, receiver gets it
  2. Single sender, multiple receivers: all receivers get the same message
  3. Receiver disconnects, sender continues broadcasting to remaining receivers
  4. Multiple sequential messages arrive in order
- Use `127.0.0.1:0` for port allocation (OS picks a free port)
- Use `tokio::time::sleep` for small delays to allow accept loop to process connections
- Use `#[derive(Codec)]` from `deli-codec-derive` for a test struct, or use primitive types like `u32` or `String` which already implement `Codec`
- All tests are `#[tokio::test]`

**Definition of Done:**

- [ ] Single receiver receives a broadcast message correctly
- [ ] Multiple receivers all receive the same broadcast
- [ ] Sender survives receiver disconnect and continues serving remaining clients
- [ ] Messages arrive in order across multiple sends

**Verify:**

- `cargo test -p deli-com -q` — all tests pass (unit + integration)
- `cargo check -p deli-com` — no warnings or errors

## Testing Strategy

- **Unit tests:** Error type conversions (Task 1), framing round-trips with in-memory streams (Task 2), SenderServer accept/client-count (Task 3), ReceiverClient connect/recv (Task 4)
- **Integration tests:** Full sender-receiver flows over real TCP (Task 5)
- **Manual verification:** None needed — this is a library crate, integration tests cover real TCP

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Port conflicts in parallel tests | Medium | Medium | Use `127.0.0.1:0` for OS-assigned ports in every test |
| Race between bind and connect | Medium | Low | After `SenderServer::bind`, the listener is ready; use small sleep before `recv` assertions |
| Lock contention on clients RwLock during send | Low | High | Use `tokio::sync::RwLock` (not std) so the lock can be held across await points. In `send()`: acquire write lock, iterate over all clients writing messages sequentially, collect failed `SocketAddr`s, remove them from the map, then drop the lock. Accept loop acquires write lock to insert new clients. `client_count()` acquires read lock. |
| Test flakiness from timing | Medium | Medium | Use `tokio::time::timeout` in tests to avoid hangs; use explicit synchronization (client_count checks) rather than sleeps where possible |

## Open Questions

- None — all design decisions resolved.

### Deferred Ideas

- Reconnection logic for `ReceiverClient` (auto-reconnect on disconnect)
- Backpressure mechanism (slow consumer handling, write timeouts for stalled clients)
- TLS support for encrypted transport
- UDP-based variant for lossy but fast communication
- `ReceiverServer` + `SenderClient` for the reverse (MPSC) pattern
- Max client limit on `SenderServer` to prevent connection storms
- Graceful shutdown with `shutdown()` method and `Drop` impl
- Health monitoring: `connected_clients() -> Vec<SocketAddr>`, per-client stats
