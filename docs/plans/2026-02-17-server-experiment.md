# Server Experiment Implementation Plan

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

**Goal:** Add a `server` experiment that binds a `WsServer` on `127.0.0.1:5090`, defines a `Data` struct (with an `i32` and a `bool`) deriving `Codec` and `Dart`, outputs the generated Dart class to `rstypes/lib/src/`, exports it from the `rstypes` barrel file, and periodically increments the int value and broadcasts `Data` over WebSocket.

**Architecture:** The experiment follows the existing pattern (e.g., `camera-viewer`): a `lib.rs` defining the shared `Data` type with `#[derive(Codec, Dart)]`, and a `src/main.rs` binary that binds a `WsServer<Data>` and runs a broadcast loop. The `Dart` derive macro's default output path must be changed from `rstypes/lib/src/tests` to `rstypes/lib/src` so that production types land in the correct location (test types move along with them, which is harmless).

**Tech Stack:** Rust, tokio, deli-com (WsServer), deli-codec (Codec + Dart derives), Flutter/Dart (rstypes package)

## Scope

### In Scope

- New `experiments/server` crate with `lib.rs` (Data struct) and `src/main.rs` (WsServer + broadcast loop)
- Change `Dart` derive macro default output path from `rstypes/lib/src/tests` to `rstypes/lib/src`
- Add `export` for `Data` in `rstypes/lib/rstypes.dart`
- Generated `data.dart` file in `rstypes/lib/src/`

### Out of Scope

- Changes to `WsServer` or `Codec` implementations
- Changes to the Flutter monitor app (the monitor already connects to 5090)
- Changing the `DELI_RSTYPES_PATH` env var mechanism

## Prerequisites

- Workspace Cargo.toml already includes `experiments/*` in members
- `deli-com` crate provides `WsServer`
- `deli-codec` crate provides `Codec` and `Dart` derive macros

## Context for Implementer

- **Patterns to follow:** `experiments/camera-viewer/` — has `lib.rs` for shared types (a struct with `#[derive(deli_codec::Codec)]`), and binary entries that import from the lib. See `experiments/camera-viewer/Cargo.toml` and `experiments/camera-viewer/src/lib.rs`.
- **Conventions:** Experiments use `edition = "2024"`, depend on workspace-local crates via relative paths (`{ path = "../../crates/..." }`). Binaries use `#[tokio::main]` with `deli_base::init_stdout_logger()` for logging.
- **Key files:**
  - `crates/deli-com/src/ws/server.rs` — `WsServer<T>` implementation (bind, send, recv)
  - `crates/deli-codec-derive/src/lib.rs:396-419` — `Dart` derive output path logic (needs default change)
  - `rstypes/lib/rstypes.dart` — barrel export file (currently empty)
  - `experiments/camera-viewer/src/camera.rs` — example of a WsServer-like broadcast loop
- **Gotchas:** The `Dart` derive runs at compile time as a proc macro side-effect (writes files to disk). Changing the default path will move all generated test Dart files from `rstypes/lib/src/tests/` to `rstypes/lib/src/`. The old files in `tests/` should be cleaned up.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Change Dart derive default output path
- [x] Task 2: Create the server experiment with Data struct and broadcast loop
- [x] Task 3: Export Data from rstypes

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Change Dart derive default output path

**Objective:** Change the `Dart` derive macro's default output directory from `rstypes/lib/src/tests` to `rstypes/lib/src`, so that production types (including the new `Data` struct) are generated in the correct location.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-codec-derive/src/lib.rs` (lines 407, 414 — change `rstypes/lib/src/tests` to `rstypes/lib/src`)
- Delete: `rstypes/lib/src/tests/` directory (old generated files will be regenerated in new location after build)

**Key Decisions / Notes:**

- Both fallback paths (lines 407 and 414) must be updated from `rstypes/lib/src/tests` to `rstypes/lib/src`
- After changing the derive, rebuild `deli-codec` tests to regenerate dart files in the new location: `cargo build -p deli-codec`
- Delete the stale `rstypes/lib/src/tests/` directory
- The comment on line 396 already says `rstypes/lib/src` — it just wasn't matching the actual code

**Definition of Done:**

- [x] Both path strings in `derive_dart` changed from `rstypes/lib/src/tests` to `rstypes/lib/src`
- [x] `cargo build -p deli-codec` succeeds and generates dart files in `rstypes/lib/src/`
- [x] Old `rstypes/lib/src/tests/` directory removed
- [x] Generated files (point.dart, message.dart, etc.) exist in `rstypes/lib/src/`

**Verify:**

- `cargo build -p deli-codec` — compiles successfully
- `ls rstypes/lib/src/*.dart` — generated dart files exist in new location
- `test ! -d rstypes/lib/src/tests` — old directory removed

### Task 2: Create the server experiment with Data struct and broadcast loop

**Objective:** Create a new `experiments/server` crate with a `Data` struct (containing an `i32` and a `bool`) that derives `Codec` and `Dart`, and a main binary that binds `WsServer<Data>` on `127.0.0.1:5090` and periodically broadcasts `Data` with an incrementing int value.

**Dependencies:** Task 1 (the Dart derive must output to the correct path)

**Files:**

- Create: `experiments/server/Cargo.toml`
- Create: `experiments/server/src/lib.rs` (Data struct with derives)
- Create: `experiments/server/src/main.rs` (WsServer bind + broadcast loop)

**Key Decisions / Notes:**

- `Data` struct: `{ value: i32, flag: bool }` — deriving `Codec` and `Dart`
- Follow `camera-viewer` pattern: lib.rs for shared types, binary for the server logic
- Use `WsServer` (not `Server`) since the Flutter monitor app uses WebSocket
- Broadcast interval: use `tokio::time::interval` (e.g., 1 second)
- Increment the `value` field each iteration; `flag` can toggle or remain constant
- Single `[[bin]]` entry named `server` at `src/main.rs`
- Dependencies: `deli-base`, `deli-com`, `deli-codec`, `tokio` (with rt, rt-multi-thread, macros, time)

**Definition of Done:**

- [x] `experiments/server/Cargo.toml` exists with correct dependencies
- [x] `experiments/server/src/lib.rs` defines `Data` with `#[derive(Codec, Dart)]` and fields `value: i32, flag: bool`
- [x] `experiments/server/src/main.rs` binds `WsServer<Data>` on `127.0.0.1:5090` and broadcasts with incrementing value
- [x] `cargo build -p server` compiles successfully
- [x] `rstypes/lib/src/data.dart` is generated with `class Data` containing `int value` and `bool flag`

**Verify:**

- `cargo build -p server` — compiles successfully
- `test -f rstypes/lib/src/data.dart` — Dart file generated
- `grep 'class Data' rstypes/lib/src/data.dart` — contains Data class

### Task 3: Export Data from rstypes

**Objective:** Update the `rstypes` barrel file to export the generated `Data` class so downstream Dart code (like the monitor app) can import it.

**Dependencies:** Task 2 (data.dart must exist)

**Files:**

- Modify: `rstypes/lib/rstypes.dart` (add export for `src/data.dart`)

**Key Decisions / Notes:**

- Use `export 'src/data.dart';` in the barrel file
- This is the standard Dart pattern for re-exporting library types

**Definition of Done:**

- [x] `rstypes/lib/rstypes.dart` contains `export 'src/data.dart';`
- [x] The export references the correct relative path

**Verify:**

- `grep "export.*data.dart" rstypes/lib/rstypes.dart` — export line present

## Testing Strategy

- Unit tests: The `Codec` derive is already tested via existing tests in `crates/deli-codec/tests/dart_codegen_tests.rs`. The `Data` struct derives the same macros, so correctness is covered.
- Integration tests: Build the `server` experiment and verify the `data.dart` file is generated with correct fields.
- Manual verification: Run the server binary briefly to verify it starts and binds to port 5090.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Moving Dart output path breaks existing test assertions | Low | Low | Test assertions use `DELI_RSTYPES_PATH` env var, not the default path. If tests fail, update the env var to point to `rstypes/lib/src`. |
| Port 5090 already in use during testing | Low | Low | The binary is an experiment — just verify it compiles and starts. |

## Open Questions

- None — requirements are clear.
