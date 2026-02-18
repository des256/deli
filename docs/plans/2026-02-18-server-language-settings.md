# Server Language Settings Implementation Plan

Created: 2026-02-18
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No

> **Status Lifecycle:** PENDING -> COMPLETE -> VERIFIED
> **Iterations:** Tracks implement->verify cycles (incremented by verify phase)
>
> - PENDING: Initial state, awaiting implementation
> - COMPLETE: All tasks implemented
> - VERIFIED: All checks passed
>
> **Approval Gate:** Implementation CANNOT proceed until `Approved: Yes`

## Summary

**Goal:** Add bidirectional language settings between server and monitor. Server sends `ToMonitor::Settings { language }` on new client connection. Monitor displays a language dropdown on the Settings page and sends `FromMonitor::Settings { language }` back when the user changes the selection.

**Architecture:** Define `Language` enum (5 variants matching `base::Language`), add `Settings { language: Language }` variant to `ToMonitor`, create `FromMonitor` enum with `Settings { language: Language }`. Write corresponding Dart codegen files manually (can't cross-compile server on this machine). Monitor `Server` class gets a `language` field updated on `ToMonitorSettings` and a `setLanguage` method that sends `FromMonitor` back. Settings tab gets a `DropdownButton<Language>`.

**Tech Stack:** Rust (server experiment), Dart/Flutter (monitor, rstypes)

## Scope

### In Scope

- Define `Language` enum in `experiments/server/src/lib.rs` with `Codec`+`Dart` derives
- Add `Settings { language: Language }` variant to `ToMonitor`
- Create `FromMonitor` enum with `Settings { language: Language }`
- Server broadcasts `ToMonitor::Settings` when new clients connect
- Write Dart codegen files: `language.dart`, updated `to_monitor.dart`, `from_monitor.dart`
- Monitor `Server` class: `language` field, `ToMonitorSettings` handling, `setLanguage` method
- Settings tab: language dropdown

### Out of Scope

- Server-side handling of received `FromMonitor` messages (WsServer reader will log decode warnings — acceptable for experiment)
- Modifying the `com` crate's `WsServer<T>` to support separate send/receive types
- Modifying `base::Language` (no codec dependency on base crate)

## Prerequisites

- None — all dependencies in place

## Context for Implementer

- **Patterns to follow:**
  - Dart enum codegen: `rstypes/lib/src/color.dart` (simple enum with no-data variants)
  - Dart enum with fields: `rstypes/lib/src/shape.dart` (named fields in variants)
  - Dart sealed class with custom type field: `rstypes/lib/src/line.dart` (imports custom type)
  - Current `ToMonitor` codegen: `rstypes/lib/src/to_monitor.dart`
- **Conventions:** Dart codegen uses `sealed class` for enums, subclass per variant named `{Enum}{Variant}`. Variant index is a `Uint32` little-endian tag. Custom types decoded via `Type.decode(bd, buf, offset)`.
- **Key files:**
  - `experiments/server/src/lib.rs` — Rust type definitions (ToMonitor)
  - `experiments/server/src/main.rs` — Server main loop
  - `crates/base/src/language.rs` — Reference for Language variants and display names
  - `crates/com/src/ws/server.rs` — WsServer API (send broadcasts to all, no per-client send)
  - `rstypes/lib/src/to_monitor.dart` — Current Dart ToMonitor codegen
  - `monitor/lib/server.dart` — Monitor Server class
  - `monitor/lib/home.dart` — Monitor UI with tabs
- **Gotchas:**
  - `WsServer<ToMonitor>` uses same type T for send AND recv decode. Monitor sending `FromMonitor` will cause decode warnings in server reader task — this is acceptable, server doesn't call `recv()`.
  - `WsServer::send()` broadcasts to ALL clients. No per-client send. So when detecting new clients, broadcast Settings to all (harmless refresh for existing clients).
  - Video frames arrive ~30fps overwriting `_data`. The `language` field must be stored separately, not extracted from `_data`.
  - Language enum in Dart is a sealed class (not a Dart `enum`). The dropdown needs a manually-defined list of all variants with display names.
  - **Manual codegen sync:** Dart files (`language.dart`, `to_monitor.dart`, `from_monitor.dart`) must be kept in sync with Rust types manually. Add a comment in `lib.rs` above `Language`, `ToMonitor`, and `FromMonitor`: `// MANUAL DART CODEGEN: Update rstypes/lib/src/{language,to_monitor,from_monitor}.dart when modifying this type`.

## Runtime Environment

- **Build command:** `cd monitor && flutter build web`
- **Analyze command:** `cd monitor && flutter analyze --no-fatal-infos`

## Progress Tracking

- [x] Task 1: Define Rust types and server logic
- [x] Task 2: Write Dart codegen files for Language, ToMonitor, FromMonitor
- [x] Task 3: Update monitor Server class
- [x] Task 4: Add Settings page language dropdown

**Total Tasks:** 4 | **Completed:** 4 | **Remaining:** 0

## Implementation Tasks

### Task 1: Define Rust types and server logic

**Objective:** Define `Language` enum, add `Settings` variant to `ToMonitor`, create `FromMonitor` enum, and send initial settings on new client connection.

**Dependencies:** None

**Files:**

- Modify: `experiments/server/src/lib.rs`
- Modify: `experiments/server/src/main.rs`

**Key Decisions / Notes:**

- `Language` enum: `EnglishUs`, `ChineseChina`, `KoreanKorea`, `DutchNetherlands`, `FrenchFrance` (matches `crates/base/src/language.rs`)
- All three types (`Language`, `ToMonitor`, `FromMonitor`) derive `codec::Codec` and `codec::Dart`
- `ToMonitor` variant order: `VideoJpeg` (0), `Settings` (1) — VideoJpeg stays at index 0
- `FromMonitor` has one variant: `Settings { language: Language }` at index 0
- In `main.rs`: detect `client_count > prev_client_count`, broadcast `ToMonitor::Settings { language: Language::EnglishUs }` **after** the next video frame send (not before). This avoids a race condition where the Settings message arrives before the client's WebSocket stream listener is established — the video frame acts as a synchronization point since video frames already work reliably.
- Additionally, broadcast `ToMonitor::Settings` on every client count change (increase OR decrease back to nonzero), not just increases above a previous maximum. This ensures reconnecting clients receive Settings after a disconnect/reconnect cycle.
- Remove unused `use server::Data` if still present; add `use server::{Language, ToMonitor}`

**Definition of Done:**

- [ ] `Language` enum defined with 5 variants and `Codec`+`Dart` derives
- [ ] `ToMonitor` has `VideoJpeg(Vec<u8>)` at index 0 and `Settings { language: Language }` at index 1
- [ ] `FromMonitor` enum defined with `Settings { language: Language }` at index 0
- [ ] Server broadcasts `ToMonitor::Settings` when client count changes (new client connects or reconnects after disconnect)
- [ ] Settings broadcast happens AFTER the video frame send (not before), to avoid race with client listener setup
- [ ] Code compiles: `cargo check -p server` (if target available)

**Verify:**

- Visual inspection of lib.rs and main.rs for correctness (can't cross-compile without RPi target)
- Verify Settings send is placed after `server.send(ToMonitor::VideoJpeg(...))` in the main loop

### Task 2: Write Dart codegen files for Language, ToMonitor, FromMonitor

**Objective:** Create/update Dart codegen files matching the new Rust type definitions, following existing codegen patterns.

**Dependencies:** Task 1

**Files:**

- Create: `rstypes/lib/src/language.dart`
- Modify: `rstypes/lib/src/to_monitor.dart`
- Create: `rstypes/lib/src/from_monitor.dart`
- Modify: `rstypes/lib/rstypes.dart`

**Key Decisions / Notes:**

- `language.dart`: Follow `color.dart` pattern — sealed class with 5 subclasses (`LanguageEnglishUs`, `LanguageChineseChina`, etc.). Variant indices: EnglishUs=0, ChineseChina=1, KoreanKorea=2, DutchNetherlands=3, FrenchFrance=4.
- **Equality:** Each Language subclass MUST override `operator==` (compare by `runtimeType`) and `hashCode` (use `runtimeType.hashCode`). This is required for `DropdownButton<Language>` value matching. Without it, `DropdownButton.value` won't highlight the selected item because Dart sealed classes don't auto-generate `==`.
- `to_monitor.dart`: Add `case 1:` in decode switch for `Settings` variant. Decode language via `Language.decode(bd, buf, offset)`. Add `ToMonitorSettings` subclass with `Language language` field.
- `from_monitor.dart`: Follow `to_monitor.dart` pattern — sealed class with `FromMonitorSettings` subclass containing `Language language` field.
- `rstypes.dart`: Add exports for `language.dart` and `from_monitor.dart`.

**Definition of Done:**

- [ ] `language.dart` has sealed class `Language` with 5 variant subclasses and correct encode/decode
- [ ] Each Language subclass overrides `operator==` (compare by `runtimeType`) and `hashCode` (`runtimeType.hashCode`)
- [ ] `to_monitor.dart` has `ToMonitorSettings` subclass with `Language language` field, decode case 1
- [ ] `from_monitor.dart` has sealed class `FromMonitor` with `FromMonitorSettings` subclass
- [ ] `rstypes.dart` exports all three new/updated files
- [ ] `flutter analyze --no-fatal-infos` reports no errors

**Verify:**

- `cd /home/desmond/deli/monitor && flutter analyze --no-fatal-infos` — no errors

### Task 3: Update monitor Server class

**Objective:** Add `language` field to Server, handle `ToMonitorSettings` messages to update it, and add `setLanguage` method that sends `FromMonitor::Settings` back to the server.

**Dependencies:** Task 2

**Files:**

- Modify: `monitor/lib/server.dart`

**Key Decisions / Notes:**

- Add `Language? _language` field with getter `Language? get language => _language`
- In the stream listener, after decoding `ToMonitor`, check if it's `ToMonitorSettings` and update `_language`
- **`_language` is NOT derived from `_data`** — it persists across `ToMonitorVideoJpeg` messages. `_data` is overwritten ~30fps by video frames, but `_language` only changes when a `ToMonitorSettings` message arrives.
- The existing `_data` and callback pattern stays unchanged — callbacks fire on ALL `ToMonitor` messages (both `VideoJpeg` and `Settings`), so the UI rebuilds ~30fps. The `_language` field persists across video frames, ensuring the dropdown always reflects the latest Settings message.
- `setLanguage(Language language)` method: encode `FromMonitorSettings(language: language)` to bytes via `.toBin()`, send via `_channel!.sink.add(bytes)`
- Import `dart:typed_data` for `Uint8List` (already imported via rstypes)

**Definition of Done:**

- [ ] `Server` has `Language? _language` field with public getter
- [ ] `ToMonitorSettings` messages update `_language` field; `_language` persists across `ToMonitorVideoJpeg` messages
- [ ] `setLanguage(Language language)` encodes `FromMonitorSettings(language: language)` and sends bytes via `_channel!.sink.add()`
- [ ] `_language` is stored as a separate field, NOT extracted from `_data`
- [ ] `flutter analyze --no-fatal-infos` reports no errors

**Verify:**

- `cd /home/desmond/deli/monitor && flutter analyze --no-fatal-infos` — no errors

### Task 4: Add Settings page language dropdown

**Objective:** Replace the Settings tab placeholder with a `DropdownButton` showing all Language variants, initialized from `server.language`, calling `server.setLanguage` on change.

**Dependencies:** Task 3

**Files:**

- Modify: `monitor/lib/home.dart`

**Key Decisions / Notes:**

- Define a constant list of `(Language, String)` pairs for dropdown items: `[(LanguageEnglishUs(), 'English (US)'), (LanguageChineseChina(), 'Chinese (China)'), ...]` matching `base::Language`'s Display impl
- Use `DropdownButton<Language>` with white text styling on black background
- Current value: `widget.server.language` — if null, show no selection (dropdown hint)
- On change: call `widget.server.setLanguage(selectedLanguage)`
- Wrap in `Center` + `Column` for layout consistency
- Use `DropdownButton.value` comparison: Language subclasses need to work with `==`. Since they're `const` and same type, `runtimeType` comparison works. Override `==` and `hashCode` on the Language sealed class or each subclass to compare by `runtimeType`.

**Definition of Done:**

- [ ] Settings tab shows a `DropdownButton` with 5 language options
- [ ] Dropdown displays language display names (e.g., "English (US)")
- [ ] Dropdown value reflects `server.language`; when `server.language` changes (from incoming `ToMonitorSettings`), dropdown updates on next `setState`
- [ ] When `server.language` is null, dropdown shows hint text "Waiting for server..."
- [ ] Selecting a language calls `server.setLanguage` with the selected `Language` instance
- [ ] Dropdown text is visible (white on black background)
- [ ] `flutter analyze --no-fatal-infos` reports no errors
- [ ] `flutter build web` succeeds

**Verify:**

- `cd /home/desmond/deli/monitor && flutter analyze --no-fatal-infos` — no errors
- `cd /home/desmond/deli/monitor && flutter build web` — builds successfully

## Testing Strategy

- Static analysis via `flutter analyze`
- Build verification via `flutter build web`
- Rust code verified by visual inspection (cross-compilation not available on this machine)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `FromMonitor` messages cause decode warnings in server | High | Low | Server reader decodes as `ToMonitor`, `FromMonitor` bytes will fail decode. Logged as warnings only — server doesn't call `recv()`. Verify: server continues running after warning, subsequent messages sent normally, no panics or connection drops. Acceptable for experiment. |
| Language sealed class instances don't compare correctly in DropdownButton | Med | Med | Override `operator==` and `hashCode` on each Language subclass to compare by `runtimeType`. Verify: `LanguageEnglishUs() == LanguageEnglishUs()` returns true. |
| Settings broadcast arrives before client listener is ready | Med | Med | Send Settings AFTER the video frame (not before), so the client's stream listener is already established by the time Settings arrives. Additionally, broadcast Settings on every client count change (not just increases) to handle reconnections. |
| Settings broadcast goes to all clients on new connect | Low | Low | All clients receive a redundant settings refresh. Harmless — just resets to same value. |
| Manual Dart codegen diverges from Rust types | Med | Med | Add `// MANUAL DART CODEGEN` comment above each Rust type definition pointing to corresponding Dart files. |

## Open Questions

- None — task is well-defined.
