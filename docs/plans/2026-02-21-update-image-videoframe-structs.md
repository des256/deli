# Update Image and VideoFrame Structs Implementation Plan

Created: 2026-02-21
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No

> **Status Lifecycle:** PENDING → COMPLETE → VERIFIED
> **Iterations:** Tracks implement→verify cycles (incremented by verify phase)

## Summary

**Goal:** Propagate the new `Image` struct (`size: Vec2<usize>`, `data: Vec<u8>`, `format: PixelFormat`) and simplified `VideoFrame` (`image: Image`) throughout all dependent code. Remove the now-deleted `VideoFormat`, `VideoData` enums and old `Image` enum (with Tensor). Update all video capture backends, convert functions, error types, tests, and experiments.

**Architecture:** The new `Image` struct in the image crate is the canonical pixel buffer. `PixelFormat` (Rgb8, Argb8, Yuyv, Yu12, Srggb10p, Jpeg) replaces the old `VideoFormat` enum. `VideoFrame` now wraps a single `Image` (no more separate `VideoData` enum). Convert functions change signature to accept/return `Image` instead of `(Vec2, &[u8])` pairs. The video crate's `VideoFormat` and `VideoData` enums are deleted — `image::PixelFormat` is used everywhere instead.

**Tech Stack:** Rust, base crate (Vec2), image crate (crates_image for JPEG encoding)

## Scope

### In Scope

- Delete `VideoFormat` and `VideoData` enums from `videoframe.rs` (already done by user)
- Update `VideoIn` struct: replace `format: VideoFormat` with `format: PixelFormat` (from image crate)
- Update `VideoInConfig`, `V4l2Config`, `RpiCamConfig`, `Realsense` config: `VideoFormat` → `PixelFormat`
- Update v4l2 and rpicam backends: construct `VideoFrame { image: Image::new(...) }` instead of `VideoFrame { data: VideoData::X(..), size }`
- Update videoin `mod.rs`: fourcc→PixelFormat mappings, `decode_config` return type, `VideoIn` fields
- Update convert functions: change signatures from `(Vec2, &[u8])` → accept/return `Image`
- Update `image::error.rs`: remove stale `Tensor` variant (old `Image` used Tensor, new one doesn't)
- Update `view.rs` binary: use `frame.image` instead of `frame.data`/`frame.size`
- Update server experiment: use `frame.image` pattern
- Update realsense stub
- Rewrite all tests (`image_tests.rs`, `encode_tests.rs`, `decode_tests.rs`)
- Add `Debug` and `Clone` derives to `Image` and `PixelFormat` (required by `VideoFrame` derives)

### Out of Scope

- Changing the Bayer/YUV conversion algorithms themselves
- Adding new pixel formats beyond what already exists
- Changing the async architecture of VideoIn

## Prerequisites

- The user has already updated `image::Image` to the new struct and `VideoFrame` to wrap it (note: VideoFrame won't compile yet — Image/PixelFormat need derives, addressed in Task 1)
- The `image` crate depends on `base` (for `Vec2`) and `crates_image` (for JPEG)
- The `video` crate depends on `image`

## Context for Implementer

- **Patterns to follow:** The new `Image` at `crates/image/src/image.rs` is the target: `Image { size: Vec2<usize>, data: Vec<u8>, format: PixelFormat }`. All frame construction must use `Image::new(size, data, format)`.
- **Conventions:** `PixelFormat` replaces both the old `image::Image` enum variants (U8/U16/F32) and `video::VideoFormat`. The video crate should `use image::PixelFormat` rather than defining its own format enum.
- **Key files:**
  - `crates/image/src/image.rs` — New Image struct and PixelFormat (already done by user)
  - `crates/image/src/convert.rs` — All pixel conversion functions (needs signature updates)
  - `crates/image/src/error.rs` — ImageError enum (needs Tensor variant removed)
  - `crates/video/src/videoframe.rs` — New VideoFrame (already done by user)
  - `crates/video/src/videoin/mod.rs` — VideoIn, VideoInConfig, fourcc constants
  - `crates/video/src/videoin/v4l2.rs` — V4L2 backend, V4l2Config
  - `crates/video/src/videoin/rpicam.rs` — RPi camera backend, RpiCamConfig
  - `crates/video/src/videoin/realsense.rs` — Stub (uses old types)
  - `crates/video/src/bin/view.rs` — minifb viewer binary
  - `experiments/server/src/main.rs` — WebSocket video server
- **Gotchas:**
  - `VideoFrame` derives `Debug, Clone`, so `Image` and `PixelFormat` must also derive them
  - The old `Image` was an enum with `Tensor<T>` — the new one is a plain struct. All test code referencing `Image::U8(tensor)` etc. must be rewritten
  - `image_tests.rs` references old `Image::U8/U16/F32` variants and `Tensor` — must be completely rewritten
  - `decode_tests.rs` references a `decode_image` function that no longer exists in lib.rs — DELETE entirely (all 8 tests use deleted API)
  - `realsense.rs` is a stale stub using old types (`Camera` trait, `VideoData`, `width`/`height` fields) — needs update to match current `VideoInDevice` trait
  - The `image::error::ImageError::Tensor` variant references `base::TensorError` — Image no longer uses Tensor, so this variant should be removed along with the `From<TensorError>` impl
  - Convert functions currently take `(Vec2<usize>, &[u8])` pairs — they should take `&Image` and return `Image` to be consistent with the new model

## Feature Inventory — Files Being Replaced/Modified

| Old Code | Functions/Items | Mapped to Task |
|----------|----------------|----------------|
| `videoframe.rs` old `VideoFormat` enum | `Yuyv, Jpeg, Srggb10p, Yu12` | Already replaced by `image::PixelFormat` |
| `videoframe.rs` old `VideoData` enum | `Yuyv(Vec<u8>), Jpeg(Vec<u8>), ...` | Deleted — `Image.data` + `Image.format` replaces this |
| `videoframe.rs` old `VideoFrame` struct | `data: VideoData, size: Vec2` | Already replaced: `image: Image` |
| `videoin/mod.rs` `VideoFormat` usage | `VideoIn.format`, `decode_config`, configs | Task 3 |
| `videoin/v4l2.rs` `VideoFormat`/`VideoData` usage | `V4l2Config.format`, `blocking_capture` frame construction | Task 4 |
| `videoin/rpicam.rs` `VideoFormat`/`VideoData` usage | `RpiCamConfig.format`, callback frame construction | Task 5 |
| `videoin/realsense.rs` stub | entire file (stale) | Task 6 |
| `image/error.rs` `Tensor` variant | `ImageError::Tensor`, `From<TensorError>` | Task 1 |
| `image/convert.rs` all functions | 14 convert functions with `(Vec2, &[u8])` signatures + new `jpeg_to_rgb`, `argb_to_u32` | Task 2 |
| `image/image.rs` missing derives | `Image`, `PixelFormat` need `Debug, Clone` | Task 1 |
| `video/bin/view.rs` | `frame_to_argb`, main | Task 7 |
| `experiments/server/src/main.rs` | main loop frame handling | Task 7 |
| `image/tests/image_tests.rs` | 6 tests using old Image enum | Task 8 |
| `image/tests/encode_tests.rs` | 13 tests | Task 8 |
| `image/tests/decode_tests.rs` | 8 tests using deleted `decode_image` | Task 8 |

### Feature Mapping Verification

- [x] All old files listed above
- [x] All functions/classes identified
- [x] Every feature has a task number
- [x] No features accidentally omitted

## Progress Tracking

- [x] Task 1: Update image crate (Image derives, error cleanup)
- [x] Task 2: Update convert.rs function signatures
- [x] Task 3: Update videoin/mod.rs (VideoIn, configs, fourcc mappings)
- [x] Task 4: Update v4l2.rs backend
- [x] Task 5: Update rpicam.rs backend
- [x] Task 6: Update realsense.rs stub
- [x] Task 7: Update view.rs binary and server experiment
- [x] Task 8: Rewrite all tests

**Total Tasks:** 8 | **Completed:** 8 | **Remaining:** 0

## Implementation Tasks

### Task 1: Update image crate foundations (Image derives, error cleanup)

**Objective:** Add required derives to `Image` and `PixelFormat`, clean up `ImageError` to remove the stale `Tensor` variant, remove unused dependencies.

**Dependencies:** None

**Files:**

- Modify: `crates/image/src/image.rs` — add `#[derive(Debug, Clone)]` to `Image` and `PixelFormat`; also add `Copy` to `PixelFormat`
- Modify: `crates/image/src/error.rs` — remove `Tensor(base::TensorError)` variant, remove `From<TensorError>` impl, remove `crates_image::ImageError` From impl if no longer needed (check), add `Convert(String)` variant for conversion errors
- Modify: `crates/image/Cargo.toml` — remove `tokio` dependency if no async functions remain in the crate

**Key Decisions / Notes:**

- `PixelFormat` should derive `Copy` since it's a simple fieldless enum — this makes it ergonomic to use
- Current state verified: `Image` and `PixelFormat` in `image.rs` have NO derives — `VideoFrame` (which derives `Debug, Clone`) cannot compile until these are added
- The `Tensor` error variant is dead code since `Image` no longer wraps `Tensor`
- Keep `crates_image` dependency since JPEG encoding still uses it in convert.rs
- Replace `ImageError::Decode`/`Encode` with a single `Convert(String)` or keep both — keeping both is fine since decode errors (input validation) and encode errors (JPEG encoding) are semantically distinct
- Image crate has no async functions — remove `tokio` from both `[dependencies]` and `[dev-dependencies]` in Cargo.toml

**Definition of Done:**

- [ ] `Image` and `PixelFormat` derive `Debug, Clone`; `PixelFormat` also derives `Copy`
- [ ] `ImageError::Tensor` variant removed
- [ ] `From<base::TensorError>` impl removed
- [ ] `cargo build -p image@0.1.0` succeeds with no errors

**Verify:**

- `cargo build -p image@0.1.0 2>&1`

### Task 2: Update convert.rs function signatures

**Objective:** Refactor all convert functions to accept `&Image` and return `Image` instead of raw `(Vec2<usize>, &[u8])` pairs. All convert functions return `Image` for a uniform API. Add `jpeg_to_rgb` for JPEG decoding. Add `argb_to_u32` helper for minifb consumption.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/image/src/convert.rs` — update all function signatures, add `jpeg_to_rgb`, add `argb_to_u32`

**Key Decisions / Notes:**

- **Uniform API:** ALL convert functions take `&Image` and return `Result<Image, ImageError>`. No exceptions — the user asked for uniformity across the repo.
- `to_rgb` functions: `fn yuyv_to_rgb(image: &Image) -> Result<Image, ImageError>` — returns `Image` with `PixelFormat::Rgb8`
- `to_argb` functions: `fn yuyv_to_argb(image: &Image) -> Result<Image, ImageError>` — returns `Image` with `PixelFormat::Argb8`. ARGB stored as 4 bytes per pixel in `Vec<u8>` with byte order `[A, R, G, B, A, R, G, B, ...]`.
- `to_jpeg` functions: `fn yuyv_to_jpeg(image: &Image, quality: u8) -> Result<Image, ImageError>` — returns `Image` with `PixelFormat::Jpeg`
- `rgb_to_jpeg` takes `&Image` (must be Rgb8 format), returns `Image` with Jpeg format
- **New: `jpeg_to_rgb`:** Decodes JPEG bytes using `crates_image::load_from_memory`, extracts RGB8 pixels via `DynamicImage::to_rgb8()`, returns `Image` with `PixelFormat::Rgb8`. Returns `ImageError::Decode` for invalid JPEG data.
- **New: `argb_to_u32`:** Helper `fn argb_to_u32(image: &Image) -> Result<Vec<u32>, ImageError>` packs Argb8 byte data into `Vec<u32>` (0xAARRGGBB format) for minifb display. This is the ONLY function returning non-`Image`, and it's a utility for display consumers, not a conversion function.
- Internal helpers (`yuv_to_rgb`, `pack_argb`) remain unchanged
- Remove `argb_to_jpeg` — ARGB→JPEG path is not needed since ARGB is only used for display (minifb)

**Definition of Done:**

- [ ] All `_to_rgb` functions take `&Image` and return `Result<Image, ImageError>`
- [ ] All `_to_argb` functions take `&Image` and return `Result<Image, ImageError>` with `PixelFormat::Argb8`
- [ ] All `_to_jpeg` functions take `&Image` (+ quality) and return `Result<Image, ImageError>`
- [ ] `jpeg_to_rgb` implemented: decodes JPEG via `crates_image::load_from_memory`, returns `Image` with `PixelFormat::Rgb8`
- [ ] `argb_to_u32` helper implemented: packs Argb8 byte data `[A,R,G,B,...]` into `Vec<u32>` (0xAARRGGBB)
- [ ] `argb_to_jpeg` removed (ARGB is only for display, not encoding)
- [ ] `cargo build -p image@0.1.0` succeeds

**Verify:**

- `cargo build -p image@0.1.0 2>&1`

### Task 3: Update videoin/mod.rs

**Objective:** Replace all `VideoFormat` references with `image::PixelFormat` in the VideoIn infrastructure. Remove `VideoFormat`-related code.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/video/src/videoin/mod.rs` — `VideoIn.format` type changes to `PixelFormat`, `decode_config` returns `PixelFormat`, `VideoInConfig::Realsense.format` changes, `format()` accessor return type changes. Add `use image::PixelFormat;`
- Modify: `crates/video/src/lib.rs` — ensure `image::PixelFormat` is re-exported if needed (or callers import from `image` directly)

**Key Decisions / Notes:**

- The fourcc constants and `fourcc_to_string` stay — they're used by v4l2/rpicam backends
- `VideoInConfig` variants' inner configs (V4l2Config, RpiCamConfig) will be updated in Tasks 4-5, so this task focuses on `mod.rs` itself and the `Realsense` variant inline fields
- `VideoIn.format` field type: `VideoFormat` → `PixelFormat`
- `VideoIn::format()` return type: `VideoFormat` → `PixelFormat`

**Definition of Done:**

- [ ] No references to `VideoFormat` or `VideoData` in `videoin/mod.rs`
- [ ] `VideoIn` stores `format: PixelFormat`
- [ ] `format()` returns `PixelFormat`

**Verify:**

- `cargo check -p video 2>&1` (may still have errors from v4l2/rpicam — that's expected until Tasks 4-5)

### Task 4: Update v4l2.rs backend

**Objective:** Replace all `VideoFormat`/`VideoData` usage with `PixelFormat` and `Image` construction.

**Dependencies:** Task 1, Task 2, Task 3

**Files:**

- Modify: `crates/video/src/videoin/v4l2.rs`

**Key Decisions / Notes:**

- `V4l2Config.format: Option<VideoFormat>` → `Option<PixelFormat>`
- `V4l2.format: VideoFormat` → `PixelFormat`
- The `desired_fourcc` match: `VideoFormat::Yuyv` → `PixelFormat::Yuyv`, etc.
- The fourcc→format match: `FOURCC_YUYV => VideoFormat::Yuyv` → `FOURCC_YUYV => PixelFormat::Yuyv`, etc.
- `blocking_capture`: instead of `match &self.format { VideoFormat::Yuyv => VideoData::Yuyv(...) }`, just construct `Image::new(self.size, frame_data.to_vec(), self.format)` directly — the format already records what kind of data it is. Then wrap in `VideoFrame { image }`.
- The `self.format.clone()` becomes `self.format` (Copy)

**Definition of Done:**

- [ ] No references to `VideoFormat` or `VideoData` in `v4l2.rs`
- [ ] Frame construction uses `Image::new(size, data, format)` → `VideoFrame { image }`
- [ ] `V4l2Config` uses `PixelFormat`

**Verify:**

- `cargo check -p video --features v4l2 2>&1`

### Task 5: Update rpicam.rs backend

**Objective:** Replace all `VideoFormat`/`VideoData` usage with `PixelFormat` and `Image` in the RPi camera backend.

**Dependencies:** Task 1, Task 2, Task 3

**Files:**

- Modify: `crates/video/src/videoin/rpicam.rs`

**Key Decisions / Notes:**

- `RpiCamConfig.format: Option<VideoFormat>` → `Option<PixelFormat>`
- All `VideoFormat::X` → `PixelFormat::X` in fourcc matches
- The callback frame construction: replace the big `match cb_format` block with direct `Image::new(cb_size, raw_data[..expected].to_vec(), cb_format)`. For YUYV and YU12 that need truncation, compute expected length then slice. For JPEG and SRGGB10P, use full raw_data.
- The old duplicate `FOURCC_*` constants were already removed (using `super::` now)

**Definition of Done:**

- [ ] No references to `VideoFormat` or `VideoData` in `rpicam.rs`
- [ ] `RpiCamConfig` uses `PixelFormat`
- [ ] Callback constructs `VideoFrame { image: Image::new(...) }`

**Verify:**

- `cargo check -p video --features rpicam 2>&1` (will need rpicam deps — may only verify on Pi)

### Task 6: Update realsense.rs stub

**Objective:** Update the stale realsense stub to use current types and trait.

**Dependencies:** Task 3

**Files:**

- Modify: `crates/video/src/videoin/realsense.rs`

**Key Decisions / Notes:**

- Currently references `Camera` trait (confirmed nonexistent — `grep "trait Camera"` in video crate returns zero matches), `VideoData`, `width`/`height` fields — all stale
- Update to implement `VideoInDevice` trait (the actual trait from mod.rs)
- Use `VideoFrame { image: Image::new(...) }` in the stub
- Keep it as a stub with `todo!()` or minimal placeholder impl

**Definition of Done:**

- [ ] Uses `VideoInDevice` trait, not `Camera`
- [ ] Uses `Image`/`VideoFrame`/`PixelFormat`, no `VideoData`/`VideoFormat`
- [ ] Compiles when realsense feature is enabled (or at minimum, no type errors against current API)

**Verify:**

- `cargo check -p video --features realsense 2>&1` (may fail on missing realsense dep — struct check only)

### Task 7: Update view.rs binary and server experiment

**Objective:** Update both consumer applications to use the new `VideoFrame { image: Image }` structure.

**Dependencies:** Task 2, Task 4, Task 5

**Files:**

- Modify: `crates/video/src/bin/view.rs`
- Modify: `experiments/server/src/main.rs`

**Key Decisions / Notes:**

- **view.rs:** Currently matches on `frame.data` (VideoData variants). Now match on `frame.image.format` (PixelFormat). The `frame_to_argb` helper becomes a match calling the appropriate `_to_argb` function, then `argb_to_u32` to get `Vec<u32>` for minifb. For JPEG: chain `jpeg_to_rgb(&frame.image)` → `rgb_to_argb(&rgb_image)` → `argb_to_u32(&argb_image)`. For other formats: call `yuyv_to_argb`/`yu12_to_argb`/`srggb10p_to_argb` → `argb_to_u32`.
- **server:** Currently matches on `frame.data` (VideoData). Now match on `frame.image.format` and call `yuyv_to_jpeg(&frame.image, 80)` etc. For Jpeg format, just use `frame.image.data.clone()` directly.
- Remove `VideoData` from imports, add `PixelFormat` and relevant convert functions

**Definition of Done:**

- [ ] view.rs compiles and uses `frame.image.format` match with `argb_to_u32` for minifb display
- [ ] server compiles and uses `frame.image.format` match
- [ ] No references to `VideoData` or `VideoFormat` in either file

**Verify:**

- `cargo build -p server 2>&1`
- `cargo build -p video --bin view 2>&1`

### Task 8: Rewrite all tests

**Objective:** Update all test files to work with the new `Image` struct and API. Delete obsolete tests.

**Dependencies:** Task 1, Task 2

**Files:**

- Modify: `crates/image/tests/image_tests.rs` — rewrite for new `Image` struct (no more `Image::U8(tensor)`)
- Modify: `crates/image/tests/encode_tests.rs` — update to use `Image` input for convert functions
- Delete: `crates/image/tests/decode_tests.rs` — `decode_image` was removed from the public API; all 8 tests reference this deleted function and old `Image` enum variants

**Key Decisions / Notes:**

- `image_tests.rs`: Old tests used `Image::U8(Tensor::new(...))` with `.shape()`, `.height()`, `.width()`, `.channels()`. New tests should construct `Image::new(Vec2::new(w,h), data, PixelFormat::Rgb8)` and test `.size`, `.data`, `.format` fields directly. Test construction with all 6 PixelFormat variants.
- `encode_tests.rs`: Tests currently call `rgb_to_jpeg(size, &data, quality)`. Update to `rgb_to_jpeg(&image, quality)` where image is constructed with `Image::new(...)`. Add tests for `jpeg_to_rgb` roundtrip and `argb_to_u32` helper.
- `decode_tests.rs`: DELETE entirely — `decode_image` no longer exists in lib.rs.
- Error tests: `ImageError::Tensor` variant is gone — remove those tests.
- Check `tokio` is no longer needed in dev-dependencies after removing async tests.

**Definition of Done:**

- [ ] `decode_tests.rs` deleted
- [ ] `image_tests.rs` rewritten with tests covering `Image::new()` with all 6 PixelFormat variants
- [ ] `encode_tests.rs` updated to use `&Image` input for all convert functions
- [ ] Tests cover `jpeg_to_rgb` roundtrip and `argb_to_u32` helper
- [ ] All tests compile and pass
- [ ] No references to old `Image` enum variants, `Tensor`, `decode_image`, `VideoFormat`, `VideoData`
- [ ] `cargo test -p image@0.1.0` passes

**Verify:**

- `cargo test -p image@0.1.0 2>&1`

## Testing Strategy

- Unit tests: Test `Image::new()` with all 6 PixelFormat variants. Test each convert function with known pixel data. Test `argb_to_u32` packing. Test `jpeg_to_rgb` decoding.
- Integration tests: JPEG encode/decode roundtrip (`rgb_to_jpeg` → `jpeg_to_rgb` → verify pixel data matches). YUV conversion roundtrips with known neutral gray values.
- Manual verification: `cargo build -p server && cargo build -p video --bin view` both succeed. With camera hardware, verify live video capture/display works end-to-end.

## Runtime Environment

**Binaries:**

- `view.rs`: Video viewer using minifb. Requires camera device (V4L2 on Linux, RPi Camera on Raspberry Pi). Run: `cargo run -p video --bin view`. Displays live video feed in window.
- `server`: WebSocket video streaming server. Requires camera device. Run: `cargo run -p server`. Streams JPEG frames to connected WebSocket clients.

**Hardware Requirements:**

- V4L2 camera (Linux): /dev/video0 device
- RPi Camera (Raspberry Pi): libcamera-dev installed, camera enabled

**Verification:**

- With camera: Run binaries, verify video display/streaming
- Without camera: Run unit tests only (integration verification requires hardware)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ARGB byte storage differs from old `Vec<u32>` | High | Medium | Store ARGB as 4 bytes/pixel `[A,R,G,B,...]` in `Image.data` with `PixelFormat::Argb8`. Provide `argb_to_u32(&Image) -> Vec<u32>` helper that packs bytes into minifb-compatible u32 (0xAARRGGBB). view.rs calls this helper before passing to minifb. |
| JPEG decode needed for view.rs display | High | Medium | Implement `jpeg_to_rgb` in convert.rs using `crates_image::load_from_memory` to decode JPEG bytes, then `DynamicImage::to_rgb8()` to extract pixels. Returns `Image` with `PixelFormat::Rgb8`. view.rs chains `jpeg_to_rgb` → `rgb_to_argb` → `argb_to_u32` for display. |
| rpicam/realsense won't compile without hardware deps | Medium | Low | Use `cargo check` with appropriate feature flags. Realsense is a stub with `todo!()` — no runtime verification possible without hardware. |
| Tests reference deleted `decode_image` function | High | Low | Delete `decode_tests.rs` entirely — `decode_image` was removed from the public API and does not exist in lib.rs. |

## Open Questions

- None — the new struct shape is clear from the user's changes.
