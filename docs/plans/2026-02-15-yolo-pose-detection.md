# YOLO Pose Detection Implementation Plan

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

**Goal:** Add a YOLO26n-pose detection pipeline to `deli-infer` that takes an image as `Tensor<f32>` (HWC, 0-255 range), runs preprocessing (resize/letterbox/normalize), inference via the ONNX backend, and post-processing (transpose, NMS, keypoint extraction) to produce structured pose detections.

**Architecture:** A new `pose` module inside `deli-infer` containing:
- `Keypoint` struct (x, y, confidence) using `deli_math::Vec2`
- `PoseDetection` struct (bounding box as `Rect<f32>`, confidence, 17 COCO keypoints)
- `YoloPoseEstimator` struct that owns an ONNX `Session`, handles preprocessing (resize + letterbox to 640×640, rescale by 1/255, HWC→CHW transpose), runs inference, and applies post-processing (confidence threshold, NMS, coordinate rescaling back to original image dimensions)

**Tech Stack:** Rust, `ort` (ONNX Runtime), `deli-math` (Tensor, Vec2, Rect), `ndarray`

## Scope

### In Scope

- `Keypoint` and `PoseDetection` output types
- `YoloPoseEstimator` struct with `new()` (loads model) and `estimate()` (full pipeline)
- Preprocessing: letterbox resize to 640×640, HWC→CHW, rescale 0-255 → 0.0-1.0
- Post-processing: transpose output, confidence filtering, greedy NMS, keypoint extraction
- Coordinate rescaling from model space (640×640) back to original image dimensions
- Support for the uint8 quantized model variant (`model_uint8.onnx`) and standard fp32
- Unit tests for preprocessing, NMS, and output decoding with synthetic data
- Integration test with the actual `yolo26n-pose` ONNX model (downloaded as test fixture)

### Out of Scope

- Image loading/decoding (user provides `Tensor<f32>` in HWC format)
- Video/streaming inference
- Multi-class pose (only single "person" class)
- Tracking across frames
- Visualization/drawing utilities
- Other YOLO task heads (detection, segmentation, classification)

## Prerequisites

- ONNX feature enabled (`cargo build --features onnx`)
- For integration tests: `yolo26n-pose` ONNX model file (downloaded to `tests/fixtures/`)
- Python + `huggingface_hub` for downloading the model: `pip install huggingface_hub`

## Context for Implementer

> This section is critical for cross-session continuity. Write it for an implementer who has never seen the codebase.

- **Patterns to follow:** The existing `OnnxBackend`/`OnnxSession` in `crates/deli-infer/src/onnx.rs` demonstrates how to load ONNX models and convert between `Tensor<f32>` and `ndarray`. The `YoloPoseEstimator` will use these conversion helpers (`tensor_to_ndarray`, `ndarray_to_tensor`) and the `Session` trait for inference.
- **Conventions:** All math types live in `deli-math` (`Vec2`, `Rect`, `Tensor`). The `deli-infer` crate uses feature-gated modules (`#[cfg(feature = "onnx")]`). New pose module should follow the same pattern.
- **Key files:**
  - `crates/deli-infer/src/onnx.rs` — ONNX backend, tensor conversion helpers
  - `crates/deli-infer/src/backend.rs` — `Backend`/`Session` traits
  - `crates/deli-infer/src/error.rs` — `InferError` enum
  - `crates/deli-math/src/rect.rs` — `Rect<T>` (origin + size bounding box)
  - `crates/deli-math/src/vec2.rs` — `Vec2<T>` (2D point)
  - `crates/deli-math/src/tensor.rs` — `Tensor<T>` (shape + flat data)
- **Gotchas:**
  - `OnnxSession::run` currently supports max 2 inputs (YOLO pose only needs 1, so this is fine)
  - `Tensor<T>` has public `shape` and `data` fields — direct access is idiomatic
  - `Rect<T>` uses `origin` (top-left) + `size`, not center-based — YOLO outputs center-based boxes that must be converted
  - The ONNX model output is `[1, 56, N]` and must be transposed to `[1, N, 56]` for per-detection iteration
- **Domain context:**
  - YOLO26n-pose outputs 17 COCO keypoints per detected person
  - Output tensor shape is `[1, 56, N]` where 56 = 4 (bbox xywh) + 1 (confidence) + 51 (17 × 3 keypoint values: x, y, visibility)
  - N = number of candidate anchors (typically 8400 for 640×640 input)
  - Letterboxing maintains aspect ratio by padding with gray (value 114/255) and tracking the pad offsets for coordinate rescaling
  - NMS (Non-Maximum Suppression) filters overlapping detections using IoU (Intersection over Union)
  - Model from HuggingFace: `onnx-community/yolo26n-pose-ONNX` — `onnx/model_uint8.onnx` or `onnx/model.onnx`
  - **uint8 model note:** ONNX Runtime automatically handles dequantization internally — the model I/O interface remains float32. No special preprocessing or postprocessing changes are needed for uint8 variants. The existing `OnnxSession::run()` with `try_extract_array::<f32>()` works because ONNX Runtime dequantizes outputs before returning them.
  - Preprocessor config: resize to 640×640, rescale by 1/255.0, no normalization

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Keypoint and PoseDetection types
- [x] Task 2: Preprocessing pipeline (letterbox + normalize + HWC→CHW)
- [x] Task 3: Post-processing pipeline (transpose, threshold, NMS, coordinate rescaling)
- [x] Task 4: YoloPoseEstimator (full pipeline integrating pre/post-processing with ONNX inference)
- [x] Task 5: Integration test with real YOLO26n-pose model

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Keypoint and PoseDetection Types

**Objective:** Define the output data types for pose estimation results — `Keypoint` (2D point + confidence) and `PoseDetection` (bounding box + confidence + keypoints).

**Dependencies:** None

**Files:**

- Create: `crates/deli-infer/src/pose.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add `pub mod pose` under onnx feature gate, add re-exports)
- Test: `crates/deli-infer/tests/pose_types_tests.rs`

**Key Decisions / Notes:**

- `Keypoint` holds `position: Vec2<f32>` and `confidence: f32` — note: YOLO pose models output a continuous confidence/visibility value in `[0.0, 1.0]` range (not COCO categorical 0/1/2), so `confidence` is the correct field name
- `PoseDetection` holds `bbox: Rect<f32>`, `confidence: f32`, and `keypoints: [Keypoint; 17]`
- Both types derive `Debug, Clone, PartialEq`
- Define `const COCO_KEYPOINT_COUNT: usize = 17` in the module
- Define a `KeypointIndex` enum with named variants (Nose, LeftEye, RightEye, ...) that converts to/from `usize` for indexing into the keypoints array
- Feature-gate the module with `#[cfg(feature = "onnx")]` since it depends on the ONNX backend

**Definition of Done:**

- [ ] `Keypoint` struct with `position: Vec2<f32>` and `confidence: f32` fields
- [ ] `PoseDetection` struct with `bbox: Rect<f32>`, `confidence: f32`, `keypoints: [Keypoint; 17]`
- [ ] `KeypointIndex` enum with all 17 COCO body parts, implements `Into<usize>` and `TryFrom<usize>`
- [ ] `PoseDetection::keypoint(&self, index: KeypointIndex) -> &Keypoint` accessor method
- [ ] All tests pass

**Verify:**

- `cargo test -p deli-infer --features onnx --test pose_types_tests -q`

---

### Task 2: Preprocessing Pipeline

**Objective:** Implement image preprocessing: letterbox resize to 640×640 maintaining aspect ratio, HWC→CHW transpose, and rescale pixel values from 0-255 to 0.0-1.0. Return both the preprocessed tensor and the letterbox parameters needed for coordinate rescaling.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-infer/src/pose.rs` (add preprocessing functions)
- Test: `crates/deli-infer/tests/pose_preprocess_tests.rs`

**Key Decisions / Notes:**

- Input: `Tensor<f32>` with shape `[H, W, 3]` (HWC format, values 0-255)
- Output: `Tensor<f32>` with shape `[1, 3, 640, 640]` (NCHW format, values 0.0-1.0) plus `LetterboxInfo { scale: f32, pad_x: f32, pad_y: f32 }` for coordinate rescaling
- Letterboxing: compute scale = min(640/H, 640/W), resize image, pad remaining area with 114.0 (gray)
- Use nearest-neighbor interpolation for resize (no external dep needed). Implementation approach: for each output pixel `(out_y, out_x)` in the resized image, compute `src_y = (out_y as f32 / scale).floor() as usize` and `src_x = (out_x as f32 / scale).floor() as usize`, clamped to source dimensions. Copy the pixel value from `src[src_y][src_x]`. This is simple index mapping — no blending.
- The preprocessing operates purely on `Tensor<f32>` data — no image library dependency
- `LetterboxInfo` is needed by post-processing to map coordinates back to original image space

**Definition of Done:**

- [ ] `LetterboxInfo` struct with `scale`, `pad_x`, `pad_y` fields
- [ ] `preprocess(image: &Tensor<f32>) -> Result<(Tensor<f32>, LetterboxInfo), InferError>` function
- [ ] Output tensor has shape `[1, 3, 640, 640]` with values in `[0.0, 1.0]`
- [ ] Letterbox correctly maintains aspect ratio (a 320×640 image gets padded horizontally, not stretched)
- [ ] Square image (640×640) produces zero padding
- [ ] Non-square image produces correct padding offsets
- [ ] Round-trip coordinate test: `LetterboxInfo` can recover original pixel coordinates from model-space coordinates within ±1.0 pixel tolerance
- [ ] All tests pass

**Verify:**

- `cargo test -p deli-infer --features onnx --test pose_preprocess_tests -q`

---

### Task 3: Post-processing Pipeline

**Objective:** Implement NMS and output decoding: transpose the raw `[1, 56, N]` output to iterate per-detection, filter by confidence threshold, apply greedy NMS, extract bounding boxes and keypoints, and rescale coordinates from model space to original image space.

**Dependencies:** Task 1, Task 2 (needs `LetterboxInfo`)

**Files:**

- Modify: `crates/deli-infer/src/pose.rs` (add post-processing functions)
- Test: `crates/deli-infer/tests/pose_postprocess_tests.rs`

**Key Decisions / Notes:**

- Raw output shape: `[1, 56, N]` — transpose to `[N, 56]` for per-detection processing
- Per detection: `[x_center, y_center, width, height, confidence, kp0_x, kp0_y, kp0_vis, ..., kp16_x, kp16_y, kp16_vis]`
- Confidence filtering: discard detections below configurable threshold (default 0.25)
- NMS: sort by confidence descending, greedy suppress overlapping boxes with IoU > threshold (default 0.45)
- IoU computation: use `Rect::intersection()` from `deli-math` for intersection area
- Coordinate rescaling: subtract pad, divide by scale (from `LetterboxInfo`) for both bbox and keypoints
- Convert center-based YOLO bbox (cx, cy, w, h) to `Rect<f32>` (origin + size)
- Return `Vec<PoseDetection>` sorted by confidence descending
- Helper function `iou(a: &Rect<f32>, b: &Rect<f32>) -> f32` for NMS

**Definition of Done:**

- [ ] `iou(a: &Rect<f32>, b: &Rect<f32>) -> f32` computes intersection-over-union correctly; returns 0.0 for non-overlapping or zero-area boxes (no division by zero)
- [ ] `postprocess(output: &Tensor<f32>, letterbox: &LetterboxInfo, conf_threshold: f32, iou_threshold: f32) -> Vec<PoseDetection>` function
- [ ] Confidence filtering correctly discards low-confidence detections
- [ ] NMS correctly suppresses overlapping detections (higher confidence kept)
- [ ] NMS edge cases tested: empty input returns empty vec, single detection passes through, all detections suppressed returns only highest-confidence, identical confidence scores handled deterministically (stable sort by index)
- [ ] Coordinates correctly rescaled from 640×640 model space to original image dimensions
- [ ] Keypoint coordinates correctly rescaled with same letterbox parameters
- [ ] Bounding boxes converted from center-based (cx, cy, w, h) to origin-based `Rect<f32>`
- [ ] Round-trip coordinate test: synthetic bbox in original image space → convert to model space (apply letterbox) → postprocess back → recovered coords match within ±1.0 pixel
- [ ] All tests pass

**Verify:**

- `cargo test -p deli-infer --features onnx --test pose_postprocess_tests -q`

---

### Task 4: YoloPoseEstimator

**Objective:** Implement the main `YoloPoseEstimator` struct that ties together model loading, preprocessing, ONNX inference, and post-processing into a single `estimate()` call.

**Dependencies:** Task 2, Task 3

**Files:**

- Modify: `crates/deli-infer/src/pose.rs` (add `YoloPoseEstimator`)
- Modify: `crates/deli-infer/src/lib.rs` (add re-exports for `YoloPoseEstimator`)
- Test: `crates/deli-infer/tests/pose_estimator_tests.rs`

**Key Decisions / Notes:**

- `YoloPoseEstimator` holds a `Box<dyn Session>` and configuration (`conf_threshold`, `iou_threshold`)
- Constructor: `YoloPoseEstimator::new(model: ModelSource, device: Device) -> Result<Self, InferError>` — uses `OnnxBackend` internally to load the model
- Main method: `estimate(&mut self, image: &Tensor<f32>) -> Result<Vec<PoseDetection>, InferError>` — runs full pipeline
- Builder pattern for optional config: `with_conf_threshold(f32)`, `with_iou_threshold(f32)`
- Default thresholds: `conf_threshold = 0.25`, `iou_threshold = 0.45`
- The estimator validates that the input tensor is 3D (HWC format) with 3 channels
- Unit tests use a mock/synthetic approach: test that the struct constructs correctly, validates inputs, and that the pipeline wiring is correct. Real inference is tested in Task 5.

**Definition of Done:**

- [ ] `YoloPoseEstimator::new(model, device)` loads model and returns estimator with default thresholds
- [ ] `with_conf_threshold()` and `with_iou_threshold()` builder methods work
- [ ] `estimate()` validates input tensor shape (must be 3D with last dim = 3)
- [ ] `estimate()` runs preprocess → inference → postprocess pipeline
- [ ] Invalid input shape returns descriptive `InferError`
- [ ] All tests pass

**Verify:**

- `cargo test -p deli-infer --features onnx --test pose_estimator_tests -q`

---

### Task 5: Integration Test with Real Model

**Objective:** Download the yolo26n-pose ONNX model and write an integration test that runs the full pipeline on a synthetic test image, verifying the end-to-end flow produces valid `PoseDetection` results.

**Dependencies:** Task 4

**Files:**

- Create: `crates/deli-infer/tests/fixtures/generate_pose_model.py` (script to download model from HuggingFace)
- Create: `crates/deli-infer/tests/pose_integration_tests.rs`
- Add fixture: `crates/deli-infer/tests/fixtures/yolo26n-pose.onnx` (downloaded, gitignored)

**Key Decisions / Notes:**

- Python script uses `huggingface_hub` to download `onnx-community/yolo26n-pose-ONNX` model — downloads BOTH `onnx/model.onnx` (fp32) and `onnx/model_uint8.onnx` (uint8 quantized) variants
- Add `crates/deli-infer/tests/fixtures/yolo*.onnx` to `.gitignore` (large binaries; this pattern keeps `test_add.onnx` tracked since it doesn't match `yolo*`)
- Integration test uses a real photograph as test image: download `https://cdn.create.vista.com/api/media/small/44316099/stock-photo-happy-family-have-fun-walking-on-beach-at-sunset` and save as `tests/fixtures/test_pose_image.jpg`. The download script should fetch this image alongside the models. Load the JPEG into a `Tensor<f32>` in HWC format using the `image` crate (add as dev-dependency for tests only).
- Test verifies: model loads, inference runs without error, output is `Vec<PoseDetection>`, detections have valid structure (17 keypoints, bbox within image bounds, confidence in 0-1 range). With a real image of people, we expect actual person detections.
- Test BOTH fp32 and uint8 variants to ensure compatibility — both model variants accept float32 input (the quantization is internal to the model weights, not the I/O interface)
- Test is gated with `#[cfg(feature = "onnx")]` and can be skipped if model file is not present (use `#[ignore]` attribute with note to download fixture first)
- Download script should: check if `huggingface_hub` is installed (print `pip install huggingface_hub` if missing), verify downloaded file is non-empty, handle network errors gracefully

**Definition of Done:**

- [ ] Python download script works: `python generate_pose_model.py` downloads both fp32 and uint8 models to `tests/fixtures/`
- [ ] `.gitignore` updated to exclude large ONNX fixture files
- [ ] Integration test loads fp32 model via `YoloPoseEstimator::new()` and runs `estimate()` without error
- [ ] Integration test loads uint8 model via `YoloPoseEstimator::new()` and runs `estimate()` without error
- [ ] If detections are returned, each has 17 keypoints and confidence in `[0.0, 1.0]`
- [ ] All returned bounding boxes have `origin.x >= 0`, `origin.y >= 0`, and `max()` coordinates <= original image dimensions
- [ ] All keypoint positions fall within original image bounds or have confidence near zero (for occluded/invisible keypoints)
- [ ] All tests pass

**Verify:**

- `python crates/deli-infer/tests/fixtures/generate_pose_model.py`
- `cargo test -p deli-infer --features onnx --test pose_integration_tests -q`

## Testing Strategy

- **Unit tests (Tasks 1-3):** Test types, preprocessing math, NMS logic, and coordinate transforms with synthetic data. No model file needed.
- **Unit tests (Task 4):** Test input validation and struct construction. Model loading tested only if fixture available.
- **Integration tests (Task 5):** Full pipeline with real ONNX model. Requires downloading model fixture.
- **Manual verification:** Run integration test and inspect output detections for plausibility.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| YOLO26 output tensor shape differs from expected `[1, 56, N]` | Low | High | Integration test validates actual output shape; post-processing checks shape before proceeding and returns `InferError::ShapeMismatch` if unexpected |
| Letterbox coordinate rescaling produces off-by-one errors | Medium | Medium | Unit tests verify round-trip: known input coords → preprocess → postprocess should recover original coords within tolerance |
| NMS implementation edge cases (all same confidence, zero-area boxes) | Low | Low | Unit tests cover edge cases: empty input, single detection, all detections suppressed, identical confidence scores |
| Large ONNX model file in test fixtures | Medium | Low | Gitignore the model file; provide download script; integration test uses `#[ignore]` if fixture missing |
| `ort` crate API changes between versions | Low | Medium | Pin `ort` version in Cargo.toml (already pinned at `2.0.0-rc.11`) |

## Open Questions

- None — all design decisions resolved.

### Deferred Ideas

- Support for other YOLO task heads (detection, segmentation) in a similar pipeline pattern
- Extract NMS/IoU into a shared module (e.g., `deli-infer::nms` or `Rect::iou()` method) when a second consumer appears
- Batch inference (multiple images per call)
- GPU-accelerated preprocessing
- Skeleton drawing utility
