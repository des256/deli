# deli-base Logging Implementation Plan

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

**Goal:** Add logging to `deli-base` using the `log` crate facade, with five log levels (Fatal, Error, Warn, Info, Debug), two swappable backends (StdoutLogger, FileLogger), and debug-only filtering.

**Architecture:** Implement the `log` crate's `Log` trait for two backends: `StdoutLogger` (println) and `FileLogger` (date-named files with day rollover). Users call `init_stdout_logger()` or `init_file_logger(dir)` at startup, which calls `log::set_logger()` + `log::set_max_level()`. Standard `log::error!`, `log::warn!`, `log::info!`, `log::debug!` macros work out of the box. Log output includes timestamp, level, thread ID, and source location. A custom `log_fatal!` macro logs at Error level then calls `std::process::exit(1)` since the `log` crate has no Fatal level. In release builds, `log::set_max_level(LevelFilter::Info)` suppresses Debug output.

**Tech Stack:** Rust (edition 2024), `log` crate (version "0.4").

## Scope

### In Scope

- Add `log = "0.4"` dependency to `deli-base`
- `StdoutLogger` struct implementing `log::Log` via `println!`
- `FileLogger` struct implementing `log::Log`, appending to date-named files (e.g., `2026-02-16.log`)
- `FileLogger` day rollover: detect date change, close old file, open new file
- Global logger initialization functions: `init_stdout_logger()`, `init_file_logger(dir: impl Into<PathBuf>)`
- `log_fatal!` macro that logs at Error level then exits via `std::process::exit(1)`
- In debug builds: `log::set_max_level(LevelFilter::Debug)` — all levels active
- In release builds: `log::set_max_level(LevelFilter::Info)` — Debug suppressed
- Log output format: `YYYY-MM-DDTHH:MM:SS [LEVEL] [thread:ID] file:line - message` (e.g., `2026-02-16T14:30:00 [ERROR] [thread:1] src/main.rs:42 - something went wrong`)
- Timestamp from `SystemTime::now()` (UTC, no timezone)
- Thread ID from `std::thread::current().id()`
- Source file and line from `log::Record::file()` and `log::Record::line()`
- Unit tests for all components

### Out of Scope

- Async logging
- Log rotation by file size
- Structured/JSON logging
- Timezone-aware timestamps
- Configuration file support

## Prerequisites

- `deli-base` crate (already exists at `crates/deli-base`)
- `log` crate (version "0.4") — to be added as dependency

## Context for Implementer

- **Patterns to follow:** Module structure from `crates/deli-base/src/lib.rs` (mod declaration + pub use re-exports). The `log` crate's `Log` trait has three methods: `enabled(&self, metadata: &Metadata) -> bool`, `log(&self, record: &Record)`, `flush(&self)`.
- **Conventions:** Edition 2024, tests in `tests/` directory with `_tests.rs` suffix. No `#[cfg(test)]` inline modules.
- **Key files:**
  - `crates/deli-base/src/lib.rs` — module declarations, pub re-exports
  - `crates/deli-base/Cargo.toml` — currently zero dependencies, will add `log`
  - `crates/deli-com/src/sender.rs` — example of existing `log` crate usage (`log::warn!`)
- **Gotchas:** The `log` crate's `set_logger` takes a `&'static dyn Log`. Use `Box::leak(Box::new(logger))` to create the `&'static` reference. `set_logger` can only be called once per process — subsequent calls return `Err`. The `log` crate has no Fatal level; we define `log_fatal!` as a custom macro.
- **Domain context:** Using the `log` crate facade standardizes logging across the deli ecosystem. `deli-com` already uses `log::warn!` — once `deli-base` provides a logger, all crates can use the same backend.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Crate setup and StdoutLogger
- [x] Task 2: FileLogger with day rollover
- [x] Task 3: Initialization functions, fatal macro, and debug gating

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Crate setup and StdoutLogger

**Objective:** Add the `log` crate dependency and implement `StdoutLogger` as a `log::Log` backend.

**Dependencies:** None

**Files:**

- Modify: `crates/deli-base/Cargo.toml` (add `log = "0.4"`)
- Create: `crates/deli-base/src/logging.rs`
- Modify: `crates/deli-base/src/lib.rs` (add `pub mod logging;` and re-exports)
- Test: `crates/deli-base/tests/logging_tests.rs`

**Key Decisions / Notes:**

- Module named `logging` (not `log`) to avoid collision with the `log` crate name
- Add `log = "0.4"` to `[dependencies]` in Cargo.toml
- `pub struct StdoutLogger;`
- Implement `log::Log` for `StdoutLogger`:
  - `enabled()`: return `true` (accept all levels that pass the max level filter)
  - `log()`: format with timestamp, level, thread ID, source location: `println!("{} [{}] [thread:{:?}] {}:{} - {}", format_timestamp(), record.level(), std::thread::current().id(), record.file().unwrap_or("unknown"), record.line().unwrap_or(0), record.args())`
  - `flush()`: `std::io::stdout().flush().ok();`
- Provide a shared `fn format_timestamp() -> String` helper that returns `YYYY-MM-DDTHH:MM:SS` (UTC) from `SystemTime::now()`. Uses the same `civil_from_days` algorithm as `format_today()` for the date portion, plus `secs % 86400` to extract hours/minutes/seconds.
- `record.level()` displays as `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`
- `std::thread::current().id()` returns a `ThreadId` — use `{:?}` to format it (displays as e.g., `ThreadId(1)`)
- `record.file()` and `record.line()` come from the `log::Record` and are populated by the `log!` macros automatically
- `lib.rs` declares `pub mod logging;` and re-exports `StdoutLogger`
- Also `pub use log;` from lib.rs so downstream crates can use `deli_base::log::info!()` etc. without adding `log` as a direct dependency

**Definition of Done:**

- [ ] `log = "0.4"` added to `Cargo.toml`
- [ ] `StdoutLogger` implements `log::Log` trait
- [ ] Output format is `YYYY-MM-DDTHH:MM:SS [LEVEL] [thread:ID] file:line - message` via `println!`
- [ ] `lib.rs` declares `pub mod logging;` and re-exports `StdoutLogger`
- [ ] `log` crate is re-exported from `lib.rs`
- [ ] Test verifies `StdoutLogger` can be created and implements `log::Log`

**Verify:**

- `cargo test -p deli-base --test logging_tests -q` — tests pass
- `cargo check -p deli-base` — no errors

### Task 2: FileLogger with day rollover

**Objective:** Implement `FileLogger` that appends log messages to a date-named file, rolling over to a new file when the day changes.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-base/src/logging.rs` (add `FileLogger` struct and impl)
- Modify: `crates/deli-base/src/lib.rs` (add `FileLogger` to re-exports)
- Test: `crates/deli-base/tests/logging_tests.rs` (add tests)

**Key Decisions / Notes:**

- `FileLogger` struct fields: `state: Mutex<FileLoggerState>` where `FileLoggerState { dir: PathBuf, current_date: String, file: std::fs::File }`
- Constructor: `pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self>` — creates the directory if it doesn't exist (`std::fs::create_dir_all`), opens initial file
- Filename format: `YYYY-MM-DD.log` (e.g., `2026-02-16.log`)
- **Date from SystemTime without extra dependencies:** Provide a `pub fn format_today() -> String` helper. Implementation:
  1. Get seconds since epoch: `SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()`
  2. Convert to days since epoch: `secs / 86400`
  3. Convert days to civil date using Howard Hinnant's algorithm (public domain, from http://howardhinnant.github.io/date_algorithms.html):
     ```
     fn civil_from_days(z: i64) -> (i64, u32, u32) {
         let z = z + 719468;
         let era = if z >= 0 { z } else { z - 146096 } / 146097;
         let doe = (z - era * 146097) as u32;
         let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
         let y = yoe as i64 + era * 400;
         let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
         let mp = (5 * doy + 2) / 153;
         let d = doy - (153 * mp + 2) / 5 + 1;
         let m = if mp < 10 { mp + 3 } else { mp - 9 };
         let y = if m <= 2 { y + 1 } else { y };
         (y, m, d)
     }
     ```
  4. Format as `format!("{:04}-{:02}-{:02}", year, month, day)`
- Implement `log::Log` for `FileLogger`:
  - `enabled()`: return `true`
  - `log()`: acquire mutex, check date rollover, write formatted line with timestamp/level/thread/source via `writeln!`. Same format as StdoutLogger. If `writeln!` fails, fall back to `eprintln!`. Never panic.
  - `flush()`: acquire mutex, call `file.flush().ok();`
- **Day rollover is lazy:** new file is created on the first `log()` call after midnight. If no logs occur on day N, no file is created.
- **Atomic rollover guarantee:** only update `current_date` and replace the `file` handle if the new file opens successfully. If open fails, keep using the old file and report via `eprintln!`.
- File open mode: `OpenOptions::new().create(true).append(true).open(path)`
- Mutex poisoning recovery: use `state.lock().unwrap_or_else(|e| e.into_inner())`
- **Test isolation:** Tests use unique directories via `std::env::temp_dir().join(format!("deli-log-test-{}", std::process::id()))` plus test-specific subdirectories

**Definition of Done:**

- [ ] `FileLogger::new(dir)` creates the log directory and opens an initial date-named file
- [ ] `log::Log::log` writes formatted line with timestamp/level/thread/source to the file
- [ ] Day rollover: when the date changes, a new file is created with the new date
- [ ] `format_today()` correctly handles leap years and year/month boundaries (verified via unit tests with known epoch values)
- [ ] Poisoned mutex is recovered using `unwrap_or_else(|e| e.into_inner())`
- [ ] Test verifies log messages are written to the correct file
- [ ] Test verifies day rollover creates a new file (using injected date or mock)

**Verify:**

- `cargo test -p deli-base --test logging_tests -q` — tests pass

### Task 3: Initialization functions, fatal macro, and debug gating

**Objective:** Implement global logger init functions, the `log_fatal!` macro, and debug-level gating for release builds.

**Dependencies:** Task 1, Task 2

**Files:**

- Modify: `crates/deli-base/src/logging.rs` (add init functions and fatal macro)
- Modify: `crates/deli-base/src/lib.rs` (add re-exports for init functions)
- Test: `crates/deli-base/tests/logging_tests.rs` (add tests)

**Key Decisions / Notes:**

- `pub fn init_stdout_logger()` — creates `StdoutLogger`, calls `log::set_logger(Box::leak(Box::new(logger)))`, sets max level based on build mode. Silently ignores if already initialized (`set_logger` returns `Err` on second call — drop the error).
- `pub fn init_file_logger(dir: impl Into<PathBuf>) -> std::io::Result<()>` — creates `FileLogger::new(dir)?`, calls `log::set_logger(Box::leak(Box::new(logger)))`, sets max level. Returns the `io::Result` from FileLogger::new; silently ignores double-init.
- **Debug gating:** Both init functions call `log::set_max_level(if cfg!(debug_assertions) { LevelFilter::Debug } else { LevelFilter::Info })`. This means Debug messages are suppressed in release builds automatically by the `log` crate filtering.
- `log_fatal!` macro:
  ```rust
  #[macro_export]
  macro_rules! log_fatal {
      ($($arg:tt)*) => {{
          log::error!($($arg)*);
          // Flush stdout to ensure message is visible
          {
              use std::io::Write;
              let _ = std::io::stdout().flush();
          }
          std::process::exit(1);
      }};
  }
  ```
  This logs at Error level (since `log` crate has no Fatal), then flushes and exits.
- For testing: tests should call loggers directly via `log::Log::log()` and `log::Log::flush()` rather than using the global `log::set_logger()`, since it can only be called once per process. The init functions are verified by a single test that calls init and verifies `log::logger()` returns a valid logger.

**Definition of Done:**

- [ ] `init_stdout_logger()` initializes the global logger with StdoutLogger
- [ ] `init_file_logger(dir)` initializes the global logger with FileLogger
- [ ] Debug messages are suppressed in release builds via `log::set_max_level`
- [ ] `log_fatal!` macro logs at Error level then calls `std::process::exit(1)`
- [ ] Both StdoutLogger and FileLogger produce identical log format: `YYYY-MM-DDTHH:MM:SS [LEVEL] [thread:ID] file:line - message`
- [ ] Test verifies init sets the global logger
- [ ] Test verifies standard `log::info!`, `log::error!` etc. work after init

**Verify:**

- `cargo test -p deli-base --test logging_tests -q` — all tests pass
- `cargo check -p deli-base` — no errors

## Testing Strategy

- **Unit tests:** StdoutLogger log::Log impl (Task 1), FileLogger writes to files and rolls over (Task 2), init functions set global logger, log_fatal! structure (Task 3)
- **Integration tests:** Not needed — loggers are self-contained
- **Manual verification:** Fatal exit is verified via code inspection (presence of `std::process::exit(1)` in `log_fatal!` macro). The `log::Log::log` method itself does NOT exit, so tests can verify that fatal-level messages are logged by calling `logger.log()` directly. Runtime exit behavior cannot be tested without terminating the test runner.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Date computation without external crate may have edge cases | Medium | Medium | Use Howard Hinnant's well-tested `civil_from_days` algorithm (pseudocode in Task 2 notes). Test with known epoch timestamps: 2000-02-29 (leap year), 2024-12-31 (year boundary), 1970-01-01 (epoch). |
| FileLogger tests may conflict if run in parallel (same temp dir) | Medium | Low | Use unique directories via `std::env::temp_dir().join(format!("deli-log-test-{}", std::process::id()))` plus test-specific subdirectories |
| Global `log::set_logger` can only be called once per process | Medium | Medium | Tests for individual loggers use `log::Log::log()` directly without the global. Only one test calls init to verify global setup. |
| Mutex poisoning on FileLogger if a panic occurs during log | Low | Low | Use `lock().unwrap_or_else(|e| e.into_inner())` to recover from poisoned mutex. |

## Open Questions

- None — all design decisions resolved.

### Deferred Ideas

- Log level filtering beyond debug gating (e.g., user-configurable minimum level)
- Log rotation by file size
- Structured JSON logging for machine parsing
- Timezone-aware timestamps (current implementation uses UTC)
- Migrate `deli-com` from its own `log` dependency to use `deli-base::log` re-export
