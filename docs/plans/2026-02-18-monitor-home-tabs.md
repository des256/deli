# Monitor Home Tabs Implementation Plan

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
> **Worktree:** Set at plan creation (from dispatcher). `Yes` uses git worktree isolation; `No` works directly on current branch (default)

## Summary

**Goal:** Extract `MonitorHome` from `main.dart` into a new `home.dart` file, refactor it to use a `TabBar` with three tabs (Overview, Audio/Video, Settings), and move the existing image feed to the Audio/Video tab.

**Architecture:** `MonitorHome` becomes a `StatefulWidget` with `SingleTickerProviderStateMixin` (required for `TabController`). The `Scaffold` gets a top `TabBar` and a `TabBarView` body. Each tab is a separate widget method. The image feed (currently the entire body) moves into the Audio/Video tab. Overview and Settings tabs get placeholder content.

**Tech Stack:** Flutter, Material Design TabBar/TabBarView

## Scope

### In Scope

- Extract `MonitorHome` and `_MonitorHomeState` from `main.dart` into `lib/home.dart`
- Update `main.dart` to import from `home.dart`
- Add `TabBar` with three tabs: Overview, Audio/Video, Settings
- Move existing `Image.memory` feed to the Audio/Video tab
- Placeholder content for Overview and Settings tabs

### Out of Scope

- Actual content for Overview and Settings tabs (beyond placeholder)
- Audio playback functionality
- Any changes to `Server`, `Config`, or `Data` classes
- Tests (no test infrastructure exists in this Flutter web project)

## Prerequisites

- None — all dependencies already in place (Flutter Material, existing server/config)

## Context for Implementer

- **Patterns to follow:** The existing `MonitorHome` in `main.dart:30-76` is the only widget pattern in the project. It uses `StatefulWidget` with server callback subscription in `initState`/`dispose`.
- **Conventions:** The project uses `super.key` in constructors, `required` named parameters, and passes `config`/`server` through widget constructors.
- **Key files:**
  - `monitor/lib/main.dart` — Entry point, contains `MonitorApp` and `MonitorHome`
  - `monitor/lib/server.dart` — WebSocket server connection, provides `Data` with `frame` bytes
  - `monitor/lib/config.dart` — YAML-based config loader
- **Gotchas:** `Image.memory` uses `gaplessPlayback: true` to avoid flickering during frame updates — must preserve this. The `SingleTickerProviderStateMixin` is needed for `TabController`.

## Runtime Environment

- **Start command:** `cd monitor && flutter run -d chrome` (Flutter web app)
- **Build command:** `cd monitor && flutter build web`

## Feature Inventory

### Files Being Replaced/Modified

| Old File | Functions/Classes | Mapped to Task |
| --- | --- | --- |
| `lib/main.dart` | `MonitorHome`, `_MonitorHomeState` (move out); `MonitorApp` (update import) | Task 1, Task 2 |

### Feature Mapping Verification

- [x] All old files listed above
- [x] All functions/classes identified
- [x] Every feature has a task number
- [x] No features accidentally omitted

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Extract MonitorHome to home.dart
- [x] Task 2: Add TabBar with three tabs and move image feed to Audio/Video tab

**Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Extract MonitorHome to home.dart

**Objective:** Move `MonitorHome` and `_MonitorHomeState` from `main.dart` into a new `lib/home.dart` file, and update `main.dart` to import it.

**Dependencies:** None

**Files:**

- Create: `monitor/lib/home.dart`
- Modify: `monitor/lib/main.dart`

**Key Decisions / Notes:**

- Copy `MonitorHome` and `_MonitorHomeState` classes verbatim into `home.dart`
- Add necessary imports to `home.dart`: `dart:typed_data`, `package:flutter/material.dart`, `server.dart`, `config.dart`
- Remove `MonitorHome` and `_MonitorHomeState` from `main.dart`
- Add `import 'home.dart';` to `main.dart`
- Remove `dart:typed_data` import from `main.dart` (only used by MonitorHome)

**Definition of Done:**

- [ ] `home.dart` contains `MonitorHome` and `_MonitorHomeState` with all necessary imports
- [ ] `main.dart` imports `home.dart` and no longer contains `MonitorHome`
- [ ] `flutter analyze` reports no errors in the monitor project
- [ ] App builds without errors: `flutter build web`

**Verify:**

- `cd /home/desmond/deli/monitor && flutter analyze --no-fatal-infos` — no errors
- `cd /home/desmond/deli/monitor && flutter build web` — builds successfully

### Task 2: Add TabBar with three tabs and move image feed to Audio/Video tab

**Objective:** Refactor `MonitorHome` to use a `TabBar`/`TabBarView` with three tabs: Overview, Audio/Video, Settings. Move the existing image feed display into the Audio/Video tab. Overview and Settings get placeholder content.

**Dependencies:** Task 1

**Files:**

- Modify: `monitor/lib/home.dart`

**Key Decisions / Notes:**

- Change `_MonitorHomeState` to use `SingleTickerProviderStateMixin` (provides a single Ticker for `TabController`; if future animations are needed on individual tabs, refactor to `TickerProviderStateMixin`)
- Initialize `TabController` with `length: 3` and `vsync: this` in `initState` BEFORE calling `widget.server.onUpdate`
- Dispose `TabController` in `dispose` BEFORE calling `widget.server.removeOnUpdate`
- Use `Scaffold` with `appBar` containing a `TabBar` (horizontal tabs at the top)
- Use `TabBarView` as the `Scaffold` body
- Tab order: Overview (index 0), Audio/Video (index 1), Settings (index 2)
- Audio/Video tab gets the existing `Image.memory` + "waiting..." fallback logic
- Overview tab: `Center(child: Text('Overview'))` placeholder
- Settings tab: `Center(child: Text('Settings'))` placeholder
- Keep `backgroundColor: Colors.black` on the Scaffold
- TabBar styling: `TabBar(labelColor: Colors.white, unselectedLabelColor: Colors.grey, indicatorColor: Colors.white, tabs: [...])`
- `AppBar` should use `backgroundColor: Colors.black` and `elevation: 0` to blend with Scaffold
- `setState` continues firing on all server updates regardless of active tab; Flutter only paints visible tabs so performance impact is minimal

**Definition of Done:**

- [ ] `MonitorHome` uses `SingleTickerProviderStateMixin` and has a `TabController`
- [ ] `TabBar` in `AppBar` shows three tabs: Overview, Audio/Video, Settings
- [ ] Audio/Video tab displays `Image.memory` with `gaplessPlayback: true`, `width`/`height: double.infinity`, `fit: BoxFit.contain` (preserving original behavior)
- [ ] Audio/Video tab shows "waiting..." text with red color (`Colors.red`) and font size 48 when data is null
- [ ] Overview and Settings tabs show placeholder text
- [ ] TabBar uses white text and indicator (`labelColor: Colors.white`, `indicatorColor: Colors.white`) on black background
- [ ] `flutter analyze` reports no errors
- [ ] App builds without errors

**Verify:**

- `cd /home/desmond/deli/monitor && flutter analyze --no-fatal-infos` — no errors
- `cd /home/desmond/deli/monitor && flutter build web` — builds successfully

## Testing Strategy

- No unit/integration test infrastructure exists in this project
- Manual verification: build the web app and confirm tabs render, image feed appears on Audio/Video tab
- Static analysis via `flutter analyze`

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Image feed shows brief blank when returning to Audio/Video tab | Low | Low | `gaplessPlayback: true` prevents flickering between frames; tab reconstruction may show brief blank until next server frame (sub-33ms at 30fps). Acceptable for initial implementation. |
| Unnecessary rebuilds when not on Audio/Video tab | Low | Low | Flutter only paints visible tabs; `setState` calls while on other tabs trigger rebuild but no paint. If performance issues arise, add `TabController` listener to gate `setState`. |

## Open Questions

- None — task is well-defined.
