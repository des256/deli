# deli-image Crate Implementation Plan

Created: 2026-02-15
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: Yes

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

**Goal:** Create a `deli-image` crate that wraps the `image` crate to decode images from memory into `Tensor<T>` from `deli-base`. A `decode_image` function takes raw bytes, auto-detects the format via `image::load_from_memory`, and returns a `DecodedImage` enum preserving the original pixel precision (u8, u16, or f32).

**Architecture:** Thin wrapper around `image::load_from_memory`. The `image` crate handles format detection and decoding. `deli-image` converts the resulting `DynamicImage` into a `Tensor<T>` with HWC layout `[height, width, channels]`. A `DecodedImage` enum wraps the typed tensors so the caller can pattern-match to get the right `Tensor<T>`.

**Tech Stack:** Rust, `image` crate (decoding), `deli-base` (Tensor).

## Scope

### In Scope

- New `crates/deli-image` crate in the workspace
- `ImageError` error type wrapping `image::ImageError` and `TensorError`
- `DecodedImage` enum with variants `U8(Tensor<u8>)`, `U16(Tensor<u16>)`, `F32(Tensor<f32>)`
- `decode_image(&[u8]) -> Result<DecodedImage, ImageError>` — auto-detects format, decodes, returns HWC tensor
- All image formats supported by the `image` crate (JPEG, PNG, GIF, BMP, TIFF, WebP, etc.)
- Grayscale, RGB, and RGBA channel layouts (1, 3, or 4 channels)
- Unit and integration tests

### Out of Scope

- Image encoding (only decoding)
- Image resizing, cropping, or other transformations
- Streaming/async decoding
- CHW layout (only HWC)

## Prerequisites

- `deli-base` crate exists with `Tensor<T>` (confirmed at `crates/deli-base/src/tensor.rs`)

## Context for Implementer

- **Patterns to follow:** Follow `crates/deli-base/Cargo.toml` for minimal crate setup with `edition = "2024"`. Follow `crates/deli-base/src/lib.rs` for `pub mod` + `pub use` re-exports. Follow `crates/deli-infer/src/error.rs` for error enum pattern.
- **Conventions:** Workspace uses `crates/*` layout. Edition 2024. Tests go in `crates/<name>/tests/` as integration test files. Types use `pub` fields (see `Tensor` in `deli-base`).
- **Key files:**
  - `crates/deli-base/src/tensor.rs:22-56` — `Tensor<T>` with `shape: Vec<usize>`, `data: Vec<T>`, `Tensor::new(shape, data)` validates data length matches shape product
  - `crates/deli-infer/src/error.rs:1-43` — error enum pattern with `Display` + `Error` impls
  - `crates/deli-infer/tests/pose_integration_tests.rs:7-29` — existing `load_image_as_tensor` showing how `image` crate converts to HWC tensor (this is essentially what `deli-image` replaces)
- **Gotchas:** `Tensor::new` validates that `data.len()` equals the product of `shape` dimensions. Always compute shape as `[height, width, channels]`.
- **Domain context:** `image::load_from_memory` returns `DynamicImage` enum. Each variant (e.g., `ImageRgb8`, `ImageRgba16`, `ImageRgb32F`) contains a typed image buffer. The pixel data is accessible via `.into_raw()` which returns `Vec<Subpixel>` in row-major HWC order — exactly what we need for `Tensor::new`.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create deli-image crate skeleton with error type and DecodedImage enum
- [ ] Task 2: Implement decode_image function

**Total Tasks:** 2 | **Completed:** 1 | **Remaining:** 1

## Implementation Tasks

### Task 1: Create deli-image crate skeleton with error type and DecodedImage enum

**Objective:** Set up the `deli-image` crate with `ImageError`, `DecodedImage` enum, and module structure. No decode logic yet.

**Dependencies:** None

**Files:**

- Create: `crates/deli-image/Cargo.toml`
- Create: `crates/deli-image/src/lib.rs`
- Create: `crates/deli-image/src/error.rs`
- Create: `crates/deli-image/src/types.rs`
- Create: `crates/deli-image/tests/types_tests.rs`

**Key Decisions / Notes:**

- `Cargo.toml`: `edition = "2024"`, dependencies: `deli-base = { path = "../deli-base" }`, `image = { version = "0.25", default-features = false, features = ["jpeg", "png", "gif", "bmp", "tiff", "webp"] }`
- `ImageError` enum:
  - `Decode(String)` — wraps `image::ImageError` (format detection failure, corrupt data, unsupported format)
  - `Tensor(deli_base::TensorError)` — wraps tensor construction errors
  - Implement `From<image::ImageError>` and `From<deli_base::TensorError>` for ergonomic `?` usage
- `DecodedImage` enum:
  - `U8(Tensor<u8>)` — for 8-bit images (Luma8, LumaA8, Rgb8, Rgba8)
  - `U16(Tensor<u16>)` — for 16-bit images (Luma16, LumaA16, Rgb16, Rgba16)
  - `F32(Tensor<f32>)` — for float images (Rgb32F, Rgba32F)
  - Helper methods: `shape() -> &[usize]`, `channels() -> usize`, `width() -> usize`, `height() -> usize`
- `lib.rs`: `pub mod error; pub mod types;` and re-exports: `pub use error::ImageError; pub use types::DecodedImage;`
- Add module-level doc comment in `lib.rs` documenting that all tensors use HWC layout `[height, width, channels]`

**Definition of Done:**

- [ ] `cargo check -p deli-image` compiles with no errors
- [ ] `ImageError` has `Display`, `Error`, `From<image::ImageError>`, and `From<TensorError>` impls
- [ ] `DecodedImage` has `U8`, `U16`, `F32` variants with `shape()`, `channels()`, `width()`, `height()` helpers
- [ ] Tests verify error conversions and `DecodedImage` helper methods

**Verify:**

- `cargo check -p deli-image` — compiles cleanly
- `cargo test -p deli-image -q` — types tests pass

### Task 2: Implement decode_image function

**Objective:** Implement `decode_image(&[u8]) -> Result<DecodedImage, ImageError>` that decodes image bytes into a typed tensor using the `image` crate, preserving original precision and channel count.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-image/src/lib.rs` (add `decode_image` function and re-export)
- Create: `crates/deli-image/tests/decode_tests.rs`

**Key Decisions / Notes:**

- Implementation:
  ```rust
  pub fn decode_image(data: &[u8]) -> Result<DecodedImage, ImageError> {
      let img = image::load_from_memory(data)?;
      match img {
          // 8-bit variants
          DynamicImage::ImageLuma8(buf) => { /* Tensor::new([h, w, 1], buf.into_raw()) */ }
          DynamicImage::ImageLumaA8(buf) => { /* [h, w, 2] */ }
          DynamicImage::ImageRgb8(buf) => { /* [h, w, 3] */ }
          DynamicImage::ImageRgba8(buf) => { /* [h, w, 4] */ }
          // 16-bit variants
          DynamicImage::ImageLuma16(buf) => { /* [h, w, 1] */ }
          DynamicImage::ImageLumaA16(buf) => { /* [h, w, 2] */ }
          DynamicImage::ImageRgb16(buf) => { /* [h, w, 3] */ }
          DynamicImage::ImageRgba16(buf) => { /* [h, w, 4] */ }
          // Float variants
          DynamicImage::ImageRgb32F(buf) => { /* [h, w, 3] */ }
          DynamicImage::ImageRgba32F(buf) => { /* [h, w, 4] */ }
          _ => { /* Convert to Rgba8 as fallback */ }
      }
  }
  ```
- Extract a helper to reduce repetition across variants: `fn buf_to_tensor<T>(buf: ImageBuffer<P, Vec<T>>) -> Result<Tensor<T>, ImageError>` using `GenericImageView::dimensions()` and `into_raw()`
- For the `_` arm (future `DynamicImage` variants), convert to `Rgba8` to ensure forward compatibility
- Tests:
  - Decode a JPEG from bytes → expect `DecodedImage::U8` with shape `[h, w, 3]`
  - Decode a PNG (RGBA) from bytes → expect `DecodedImage::U8` with shape `[h, w, 4]`
  - Decode a 16-bit PNG → expect `DecodedImage::U16`
  - Verify pixel values match `image` crate decode
  - Error on invalid/empty data
  - Use `image` crate in tests to generate small test images (encode to bytes, then decode with `decode_image`)

**Definition of Done:**

- [ ] `decode_image` decodes JPEG bytes into `DecodedImage::U8` with correct HWC shape
- [ ] `decode_image` decodes PNG (RGBA) bytes into `DecodedImage::U8` with 4 channels
- [ ] `decode_image` decodes 16-bit PNG into `DecodedImage::U16`
- [ ] `decode_image` returns `ImageError::Decode` for corrupt or unrecognized data
- [ ] Pixel data in tensor matches direct `image` crate decode (exact match for lossless formats)
- [ ] All tests pass

**Verify:**

- `cargo test -p deli-image -q` — all tests pass
- `cargo check -p deli-image` — no warnings

## Testing Strategy

- **Unit tests:** `ImageError` conversions, `DecodedImage` helper methods
- **Integration tests:** Generate test images using `image` crate (dev-dependency for encoding), encode to bytes, decode with `decode_image`, verify tensor shape and pixel values
- **Formats to test:** JPEG (RGB u8), PNG (RGBA u8), 16-bit PNG (u16), grayscale PNG (1 channel)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| New `DynamicImage` variants in future `image` versions | Low | Low | `_` arm converts unknown variants to `Rgba8` as fallback, ensuring forward compatibility |
| `image` crate feature flags miss a needed format | Low | Med | Include common format features explicitly in Cargo.toml; users can enable more via feature passthrough if needed |

## Open Questions

- None — task is well-specified.
