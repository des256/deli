# WebSocket Server and Client for deli-com Implementation Plan

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

**Goal:** Add async WebSocket server and client to deli-com with Codec-based binary serialization, using tokio-websockets for protocol handling.

**Architecture:** Mirror the existing TCP `Server<T>`/`Client<T>` API surface but transport messages over WebSocket binary frames instead of raw TCP with length-prefix framing. The WebSocket protocol handles framing natively, so `Codec::to_bytes()` payloads are sent directly as binary WebSocket messages. New modules `ws/server.rs` and `ws/client.rs` are added under a `ws` module, re-exported from `lib.rs` as `WsServer` and `WsClient`.

**Tech Stack:** `tokio-websockets` (with `server`, `client`, `sha1_smol`, `fastrand` features), `futures-util` (for `StreamExt`/`SinkExt` on `WebSocketStream`), existing `deli-codec::Codec` trait.

## Scope

### In Scope

- `WsServer<T: Codec>`: bind, accept WebSocket connections, broadcast to all clients, receive messages from any client
- `WsClient<T: Codec>`: connect via WebSocket, send and receive messages
- New `ws` module with `server.rs` and `client.rs` submodules
- `ComError` extended for WebSocket-specific errors
- Integration tests mirroring existing TCP test patterns
- Re-exports in `lib.rs`

### Out of Scope

- TLS/WSS support (can be added later via `tokio-websockets` TLS features)
- WebSocket compression
- Custom HTTP upgrade headers or path routing
- Modifying existing TCP `Server`/`Client` code
- Custom close frame handling (basic close is sufficient)

## Prerequisites

- `tokio-websockets` crate added to `deli-com/Cargo.toml`
- `futures-util` crate added to `deli-com/Cargo.toml` (for `StreamExt`/`SinkExt`)

## Context for Implementer

> This section is critical for cross-session continuity.

- **Patterns to follow:** The existing TCP server/client in `crates/deli-com/src/server.rs` and `crates/deli-com/src/client.rs` define the API pattern. The WebSocket versions should match the same `bind`/`connect`/`send`/`recv`/`client_count`/`local_addr` method signatures.
- **Conventions:**
  - `T: Codec + Send + 'static` bounds on server (since it spawns tasks)
  - `T: Codec` bounds on client (single connection, no spawning)
  - `ComError` is the unified error type in `crates/deli-com/src/error.rs`
  - Tests use `tokio::test`, `127.0.0.1:0` for ephemeral ports, `tokio::time::timeout` for assertions
- **Key files:**
  - `crates/deli-com/src/server.rs` — TCP server implementation (the pattern to mirror)
  - `crates/deli-com/src/client.rs` — TCP client implementation (the pattern to mirror)
  - `crates/deli-com/src/framing.rs` — Length-prefix framing (NOT used for WebSocket; WS has native framing)
  - `crates/deli-com/src/error.rs` — `ComError` enum (needs WebSocket error variant)
  - `crates/deli-com/src/lib.rs` — Module declarations and re-exports
  - `crates/deli-codec/src/lib.rs` — `Codec` trait definition (`encode`/`decode`/`to_bytes`/`from_bytes`)
- **Gotchas:**
  - `tokio-websockets` `WebSocketStream` implements `futures_util::Stream` (for receiving) and `futures_util::Sink` (for sending). You need `StreamExt::next()` and `SinkExt::send()`.
  - `ServerBuilder::new().accept(tcp_stream).await` performs the WebSocket handshake and returns `(http::Request, WebSocketStream)`.
  - `ClientBuilder::from_uri(uri).connect().await` returns `(WebSocketStream, http::Response)`.
  - `Message::binary(payload)` creates a binary frame. `msg.into_payload()` gets the `Payload` (which derefs to `[u8]`).
  - The server needs to split the `WebSocketStream` for concurrent read/write. Use `stream.split()` from `StreamExt` to get `(SplitSink, SplitStream)`.
  - `sha1_smol` feature provides SHA1 for the WebSocket handshake without pulling in heavy TLS deps.
  - `fastrand` feature provides the random number generator needed for client-side frame masking.
- **Domain context:** deli-com is a communication layer for the deli project. `Codec` types serialize to/from bytes. The server broadcasts to all connected clients (pub-sub pattern), and also receives messages from clients (bidirectional). Messages are typed — both sides agree on `T: Codec`.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Add dependencies and error variant
- [x] Task 2: Implement WsServer
- [x] Task 3: Implement WsClient
- [x] Task 4: Integration tests

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add dependencies and error variant

**Objective:** Add `tokio-websockets` and `futures-util` to `Cargo.toml` and extend `ComError` with a WebSocket error variant.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-com/Cargo.toml`
- Modify: `crates/deli-com/src/error.rs`
- Test: `crates/deli-com/tests/error_tests.rs`

**Key Decisions / Notes:**

- Add `tokio-websockets` with features: `client`, `server`, `sha1_smol`, `fastrand`
- Add `futures-util` with default features (needed for `StreamExt`/`SinkExt`)
- Add `ComError::WebSocket(tokio_websockets::Error)` variant with `Display` and `From` impl
- Keep `http` as a dependency (pulled in by `tokio-websockets`, needed for URI in client)

**Definition of Done:**

- [ ] `tokio-websockets` and `futures-util` are in `[dependencies]` with correct features
- [ ] `ComError::WebSocket` variant exists with `Display` and `From<tokio_websockets::Error>` impl
- [ ] `cargo check -p deli-com` succeeds with no errors
- [ ] Error variant test passes

**Verify:**

- `cargo check -p deli-com` — compiles cleanly
- `cargo test -p deli-com --test error_tests -q` — error tests pass

### Task 2: Implement WsServer

**Objective:** Create `WsServer<T: Codec>` that binds a TCP listener, accepts WebSocket connections via upgrade handshake, broadcasts binary messages to all clients, and receives messages from any client via an mpsc channel.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-com/src/ws/mod.rs`
- Create: `crates/deli-com/src/ws/server.rs`
- Modify: `crates/deli-com/src/lib.rs` (add `pub mod ws;` and `pub use ws::server::WsServer;`)
- Test: `crates/deli-com/tests/ws_server_tests.rs`

**Key Decisions / Notes:**

- Follow the same architecture as TCP `Server<T>`:
  - Background accept loop spawned in `bind()`
  - Per-client reader task spawned on accept, sending to mpsc channel
  - Write halves stored in `Arc<RwLock<HashMap<SocketAddr, SplitSink>>>` for broadcast
  - `send(&self, value: &T)` broadcasts `Message::binary(value.to_bytes())` to all clients
  - `recv(&mut self)` reads from mpsc channel
  - `client_count()` returns number of connected clients
  - `local_addr()` returns bound address
  - `Drop` aborts accept task
- The accept loop calls `ServerBuilder::new().accept(tcp_stream).await` for WebSocket handshake, then `ws_stream.split()` to separate read/write halves
- Reader tasks decode binary messages: `msg.into_payload()` → `T::from_bytes(&payload)` → send to mpsc
- Text messages and control frames (ping/pong/close) are ignored in the reader loop (only binary messages are forwarded)
- The `SplitSink` type is `futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>`

**Definition of Done:**

- [ ] `WsServer::bind("127.0.0.1:0")` returns a server with `local_addr()` and `client_count() == 0`
- [ ] WebSocket clients can connect and are tracked in `client_count()`
- [ ] `send()` broadcasts binary messages to all connected WebSocket clients
- [ ] `recv()` returns messages sent by any WebSocket client
- [ ] Disconnected clients are removed from the client map
- [ ] All tests in `ws_server_tests.rs` pass

**Verify:**

- `cargo test -p deli-com --test ws_server_tests -q` — all server tests pass

### Task 3: Implement WsClient

**Objective:** Create `WsClient<T: Codec>` that connects to a `WsServer` via WebSocket, and provides `send()` and `recv()` methods for typed binary messages.

**Dependencies:** Task 2

**Files:**

- Create: `crates/deli-com/src/ws/client.rs`
- Modify: `crates/deli-com/src/ws/mod.rs` (add `pub mod client;`)
- Modify: `crates/deli-com/src/lib.rs` (add `pub use ws::client::WsClient;`)
- Test: `crates/deli-com/tests/ws_client_tests.rs`

**Key Decisions / Notes:**

- Follow the same pattern as TCP `Client<T>`:
  - `connect(addr)` creates a `TcpStream`, then uses `ClientBuilder` to perform WebSocket handshake
  - Store the `WebSocketStream` directly (no split needed for client since send/recv are `&mut self`)
  - `send(&mut self, value: &T)` sends `Message::binary(value.to_bytes())`
  - `recv(&mut self)` reads next binary message via `StreamExt::next()`, ignoring non-binary frames, returns `T::from_bytes(&payload)`
  - Returns `ComError::ConnectionClosed` on `None` from the stream
- The connect method needs to build a `ws://` URI from the socket address: `format!("ws://{addr}")`
- Use `ClientBuilder::from_uri(uri).connect_on(tcp_stream).await` to reuse an existing TcpStream (avoiding DNS resolution for `127.0.0.1` addresses). Alternatively, `ClientBuilder::from_uri(uri).connect().await` handles TCP internally.

**Definition of Done:**

- [ ] `WsClient::connect(addr)` successfully connects to a running `WsServer`
- [ ] `send()` transmits Codec-encoded binary messages over WebSocket
- [ ] `recv()` decodes binary WebSocket messages back to `T`
- [ ] Returns `ComError::ConnectionClosed` when server disconnects
- [ ] All tests in `ws_client_tests.rs` pass

**Verify:**

- `cargo test -p deli-com --test ws_client_tests -q` — all client tests pass

### Task 4: Integration tests

**Objective:** End-to-end tests verifying WsServer and WsClient work together, mirroring the existing TCP integration test patterns.

**Dependencies:** Task 3

**Files:**

- Create: `crates/deli-com/tests/ws_integration_tests.rs`

**Key Decisions / Notes:**

- Mirror the existing `tests/integration_tests.rs` patterns:
  - Single sender, single receiver
  - Single sender, multiple receivers (broadcast)
  - Receiver disconnect, sender continues
  - Multiple messages arrive in order
  - Stress test with many receivers
- Also test client-to-server communication (client sends, server receives)
- Use `tokio::time::timeout` for all async assertions
- Use `sleep(Duration::from_millis(50))` after connect to allow accept loop to process

**Definition of Done:**

- [ ] `test_ws_single_sender_single_receiver` passes
- [ ] `test_ws_single_sender_multiple_receivers` passes (broadcast)
- [ ] `test_ws_receiver_disconnect_sender_continues` passes
- [ ] `test_ws_multiple_messages_arrive_in_order` passes
- [ ] `test_ws_client_to_server` passes (client sends, server recvs)
- [ ] `test_ws_stress_many_receivers` passes
- [ ] All existing TCP tests still pass (no regressions)

**Verify:**

- `cargo test -p deli-com --test ws_integration_tests -q` — all WS integration tests pass
- `cargo test -p deli-com -q` — all tests pass (including existing TCP tests)

## Testing Strategy

- **Unit tests:** Error variant formatting and conversion (Task 1)
- **Component tests:** WsServer bind/accept/broadcast (Task 2), WsClient connect/send/recv (Task 3)
- **Integration tests:** Full server+client workflows (Task 4)
- **Regression:** All existing TCP tests must continue to pass

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `tokio-websockets` API incompatibility with split pattern | Low | High | Verified via GitHub code examples that `WebSocketStream` supports `StreamExt::split()` into `SplitSink`/`SplitStream` |
| WebSocket handshake adds latency to tests | Low | Low | Tests use `127.0.0.1:0` (loopback), handshake overhead is negligible |
| `futures-util` version conflicts with workspace | Low | Med | Check workspace lockfile; `tokio-websockets` already depends on `futures-core`/`futures-sink` so `futures-util` should be compatible |
| Binary message payload extraction API changes | Low | Med | Pin `tokio-websockets` to a specific version in Cargo.toml |

## Open Questions

- None — the design mirrors the existing TCP architecture closely.

### Deferred Ideas

- WSS (TLS) support via `tokio-websockets` rustls features
- Custom HTTP upgrade paths for routing multiple services on one port
- Per-client send (not just broadcast) on the server side
