# Inference & PoseDetector Implementation Plan

Created: 2026-02-16
Status: VERIFIED
Approved: Yes
Iterations: 1
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

**Goal:** Add an `Inference` struct (CPU/CUDA device selection) and `PoseDetector` API to `deli-infer` using Candle, with `Inference::use_pose_detector()` to create detector sessions and `PoseDetector::detect()` (async) to run YOLOv8-Pose estimation on `deli-base::Tensor<f32>` frames.

**Architecture:** The crate is structured in layers: core types (`InferError`, `Keypoint`, `PoseDetection`) → device abstraction (`Inference`) → YOLOv8-Pose model (ported from candle-examples, Apache-2.0/MIT) → post-processing (NMS, coordinate scaling) → public `PoseDetector` API that ties everything together. The model code is split across multiple files under `model/` to stay under the 300-line limit. The `detect()` method is async via `tokio::task::spawn_blocking` to avoid blocking the async runtime during inference.

**Tech Stack:** candle-core 0.9, candle-nn 0.9, safetensors 0.5 (model metadata), tokio (rt), deli-base

## Scope

### In Scope

- `InferError` enum for all inference errors
- `Inference` struct with CPU/CUDA device selection
- `PoseDetector` struct created via `Inference::use_pose_detector()`
- YOLOv8-Pose model port (DarkNet backbone, FPN neck, PoseHead) from candle-examples
- All 5 model sizes: N, S, M, L, X — auto-detected from safetensors file (no user-specified size parameter)
- Input preprocessing: `deli-base::Tensor<f32>` HWC 0-255 → candle NCHW 0-1, with resize
- Post-processing: NMS, bbox extraction, keypoint extraction → `Vec<PoseDetection>`
- `detect()` as async method using `spawn_blocking`
- CUDA support behind `cuda` feature flag
- `CocoKeypoint` enum for named keypoint access (17 COCO body keypoints)

### Out of Scope

- Model downloading from HuggingFace Hub (user provides `.safetensors` path)
- Object detection (YoloV8 without pose) — only pose estimation
- Training / fine-tuning
- Video-level tracking or temporal smoothing
- Model quantization or optimization
- Batch inference (single-frame only)

## Prerequisites

- candle-core 0.9.2 and candle-nn 0.9.2 crates (added to Cargo.toml)
- YOLOv8-Pose `.safetensors` model file is already present at `models/yolov8n-pose.safetensors` (nano variant, 6.3MB). Other variants can be downloaded from HuggingFace `lmz/candle-yolo-v8`. The model must have separate conv and batch_norm weights (not pre-fused) — the HuggingFace models satisfy this.
- Model size (N/S/M/L/X) is **auto-detected** from the safetensors file by inspecting the first conv layer's output channels. No user-specified size parameter needed.
- For CUDA: CUDA toolkit and cuDNN installed, `cuda` feature enabled

## Context for Implementer

- **Patterns to follow:** Error enum pattern from `crates/deli-camera/src/error.rs:1-34` (enum with Display, Error, From impls). Async trait pattern from `crates/deli-camera/src/traits.rs:8-15` (edition 2024 `allow(async_fn_in_trait)`). Module organization from `crates/deli-camera/src/lib.rs:1-26` (feature-gated modules). Image tensor layout is always HWC per `crates/deli-image/src/lib.rs:7`.
- **Conventions:** Edition 2024. All crate names prefixed with `deli-`. Tensors use HWC layout `[height, width, channels]` in deli-base. Error types are crate-level enums (not `anyhow`/`thiserror`). Feature flags for optional backends (see deli-camera `v4l2`/`rpicam`).
- **Key files the implementer must read:**
  - `crates/deli-base/src/tensor.rs` — `Tensor<T>` struct (shape: `Vec<usize>`, data: `Vec<T>`)
  - `crates/deli-base/src/rect.rs` — `Rect<T>` struct (origin + size)
  - `crates/deli-base/src/vec2.rs` — `Vec2<T>` struct (x, y)
  - `crates/deli-camera/src/error.rs` — Error enum pattern to follow
- **Gotchas:**
  - `deli-base::Tensor` is a simple `Vec`-backed container — no GPU support, no strides. Data must be copied into candle tensors for inference.
  - `Rect<T>` uses origin+size representation (not min/max). Use `Rect::from_min_max()` when converting from YOLO bbox format.
  - candle's `VarBuilder::from_mmaped_safetensors` is `unsafe` (memory-mapped file). This is standard candle usage.
  - candle `Conv2dConfig` may differ across 0.9.x point releases — use `..Default::default()` for forward compatibility.
  - The YOLOv8-Pose model expects NCHW input at 640-pixel max dimension (divisible by 32). Preprocessing must handle arbitrary input sizes.
- **Domain context:**
  - COCO pose estimation uses 17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
  - YOLOv8-Pose output tensor shape is `[56, N]` where 56 = 4 (bbox xywh) + 1 (confidence) + 51 (17 keypoints × 3 values: x, y, visibility).
  - NMS (Non-Maximum Suppression) removes duplicate detections by comparing IoU between bounding boxes.
  - Model sizes (N→X) trade speed for accuracy. N is fastest (~3ms/frame on GPU), X is most accurate.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Core types and error handling
- [x] Task 2: Inference struct (device selection)
- [x] Task 3: YOLOv8 model building blocks
- [x] Task 4: YOLOv8 model assembly (backbone, neck, heads)
- [x] Task 5: Post-processing (NMS, output extraction)
- [x] Task 6: PoseDetector and public API

**Total Tasks:** 6 | **Completed:** 6 | **Remaining:** 0

## Implementation Tasks

### Task 1: Core types and error handling

**Objective:** Define the public output types (`Keypoint`, `PoseDetection`, `CocoKeypoint`) and error type (`InferError`) that the rest of the crate depends on.

**Dependencies:** None

**Files:**

- Create: `crates/deli-infer/src/error.rs`
- Create: `crates/deli-infer/src/types.rs`
- Modify: `crates/deli-infer/src/lib.rs`
- Modify: `crates/deli-infer/Cargo.toml` (add candle-core, candle-nn, tokio deps)
- Test: `crates/deli-infer/tests/types_test.rs`

**Key Decisions / Notes:**

- `InferError` variants: `Candle(String)` (wraps `candle_core::Error`), `Shape(String)` (invalid input dimensions), `Io(String)` (file not found for model). Implement `From<candle_core::Error>`.
- `Keypoint`: `{ position: Vec2<f32>, confidence: f32 }` — uses deli-base `Vec2<f32>`.
- `PoseDetection`: `{ bbox: Rect<f32>, confidence: f32, keypoints: [Keypoint; 17] }` — uses deli-base `Rect<f32>`. Fixed array of 17 COCO keypoints.
- `CocoKeypoint` enum: `Nose = 0, LeftEye = 1, ..., RightAnkle = 16`. Used as index into the keypoints array. Implements `Into<usize>`.
- Cargo.toml additions: `candle-core = "0.9"`, `candle-nn = "0.9"`, `safetensors = "0.5"` (for reading model metadata to auto-detect size), `tokio = { version = "1", features = ["rt"] }`. Feature `cuda` enables `candle-core/cuda` and `candle-nn/cuda`. Dev-deps: `tokio = { version = "1", features = ["rt", "macros"] }`.

**Definition of Done:**

- [x] All tests pass (unit tests for type construction, Display, From impls, CocoKeypoint indexing)
- [x] No diagnostics errors
- [x] `InferError` has Display, Error, and `From<candle_core::Error>` impls
- [x] `CocoKeypoint` maps all 17 COCO body keypoints to indices 0-16
- [x] `cargo build -p deli-infer` succeeds

**Verify:**

- `cargo test -p deli-infer --test types_test -q` — type tests pass
- `cargo build -p deli-infer -q` — crate compiles

---

### Task 2: Inference struct (device selection)

**Objective:** Create the `Inference` struct that wraps a candle `Device` and provides a factory method `use_pose_detector()` to create `PoseDetector` instances.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-infer/src/inference.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add module, re-export)
- Test: `crates/deli-infer/tests/inference_test.rs`

**Key Decisions / Notes:**

- `Inference::cpu()` → creates with `Device::Cpu`.
- `Inference::cuda(ordinal: usize)` → creates with `Device::new_cuda(ordinal)?`. Only available when `cuda` feature is enabled (`#[cfg(feature = "cuda")]`).
- `Inference` stores the `candle_core::Device` internally.
- `use_pose_detector()` signature: `pub fn use_pose_detector(&self, model_path: impl AsRef<Path>) -> Result<PoseDetector, InferError>`. Model size is auto-detected from the safetensors file. This will be wired up in Task 6 — for now, create the struct with a placeholder method that returns `todo!()`.
- The struct should derive `Debug`.

**Definition of Done:**

- [x] All tests pass (CPU construction, device() accessor)
- [x] No diagnostics errors
- [x] `Inference::cpu()` creates successfully and reports CPU device
- [x] `Inference::cuda()` is gated behind `#[cfg(feature = "cuda")]`
- [x] `use_pose_detector()` method exists with correct signature (placeholder body for now)

**Verify:**

- `cargo test -p deli-infer --test inference_test -q` — inference tests pass
- `cargo build -p deli-infer -q` — compiles

---

### Task 3: YOLOv8 model building blocks

**Objective:** Port the fundamental neural network building blocks from the candle-examples YOLOv8 implementation: `ConvBlock`, `Bottleneck`, `C2f`, `Sppf`, `Upsample`, `Dfl`.

**Dependencies:** Task 1 (for Cargo.toml candle deps)

**Files:**

- Create: `crates/deli-infer/src/model/mod.rs`
- Create: `crates/deli-infer/src/model/blocks.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add `mod model;`)
- Test: `crates/deli-infer/tests/model_blocks_test.rs`

**Key Decisions / Notes:**

- Source: `candle-wasm-examples/yolo/src/model.rs` from `huggingface/candle` (Apache-2.0/MIT dual-licensed, verified: https://github.com/huggingface/candle/blob/main/LICENSE-APACHE). The wasm version is identical to the examples version but without tracing spans — use it as reference. Add attribution comment at top of each ported file.
- All structs are `pub(crate)` — not exposed in public API.
- `ConvBlock`: conv2d_no_bias + batch_norm + SiLU activation. `load(vb, c1, c2, k, stride, padding)`.
- `Bottleneck`: Two ConvBlocks with optional residual connection.
- `C2f`: Cross Stage Partial with bottleneck blocks. Chunks input, processes through bottlenecks, concatenates.
- `Sppf`: Spatial Pyramid Pooling - Fast. Three max_pool2d cascaded, concatenated.
- `Upsample`: Simple nearest-neighbor upsampling via `upsample_nearest2d`.
- `Dfl`: Distribution Focal Loss head. Reshapes, softmax, 1x1 conv.
- Use `Conv2dConfig { padding, stride, groups: 1, dilation: 1, ..Default::default() }` for forward compatibility.
- All blocks implement `candle_nn::Module` (fn forward(&self, &Tensor) -> Result<Tensor>).
- `model/mod.rs` re-exports blocks and will later re-export the full model types.

**Definition of Done:**

- [x] All tests pass (each block produces correct output shape with random weights via VarMap)
- [x] No diagnostics errors
- [x] `ConvBlock::load` + forward: input `[1, 3, 64, 64]` → output channels and spatial dims correct
- [x] `C2f::load` + forward: correct output channels for given parameters
- [x] `Sppf::load` + forward: spatial dimensions preserved, channels correct
- [x] All blocks use `pub(crate)` visibility

**Verify:**

- `cargo test -p deli-infer --test model_blocks_test -q` — block shape tests pass

---

### Task 4: YOLOv8 model assembly (backbone, neck, heads)

**Objective:** Port the DarkNet backbone, YoloV8Neck (FPN), DetectionHead, PoseHead, and the top-level `YoloV8Pose` struct. Also port the `Multiples` config and anchor generation.

**Dependencies:** Task 3

**Files:**

- Create: `crates/deli-infer/src/model/backbone.rs`
- Create: `crates/deli-infer/src/model/neck.rs`
- Create: `crates/deli-infer/src/model/head.rs`
- Modify: `crates/deli-infer/src/model/mod.rs` (add submodules, re-export `YoloV8Pose`, `Multiples`)
- Test: `crates/deli-infer/tests/model_assembly_test.rs`

**Key Decisions / Notes:**

- Source: Same candle-wasm-examples reference.
- `backbone.rs` (~100 lines): `DarkNet` struct with `load(vb, Multiples)` and `forward(&Tensor) -> (Tensor, Tensor, Tensor)` returning 3 feature map scales.
- `neck.rs` (~100 lines): `YoloV8Neck` with upsample + C2f blocks for Feature Pyramid Network.
- `head.rs` (~200 lines): `DetectionHead` (DFL + detection convolutions), `PoseHead` (detection + keypoint convolutions), `make_anchors()`, `dist2bbox()` helper functions. `PoseHead::forward` returns raw `[56, N]` tensor.
- `model/mod.rs` (~70 lines): `Multiples` struct with `n()/s()/m()/l()/x()` constructors and `filters()` method. `YoloV8Pose` struct with `load(vb, Multiples)` and `Module` impl.
- All internal structs are `pub(crate)`. Only `YoloV8Pose` and `Multiples` are `pub(crate)` — they're used by `PoseDetector` but not exposed to users.
- `YoloV8Pose::load` hardcodes `num_classes=1` and `kpt=(17, 3)` for COCO pose.

**Definition of Done:**

- [x] All tests pass (full model forward pass with VarMap random weights)
- [x] No diagnostics errors
- [x] `YoloV8Pose::load` + `forward`: input `[1, 3, 640, 640]` → output shape `[1, 56, N]` where N = 8400 (sum of anchor grid sizes: 80×80 + 40×40 + 20×20)
- [x] Each file stays under 300 lines
- [x] All 5 model size variants construct without error

**Verify:**

- `cargo test -p deli-infer --test model_assembly_test -q` — assembly tests pass

---

### Task 5: Post-processing (NMS, output extraction)

**Objective:** Implement non-maximum suppression and the conversion from raw model output tensor to `Vec<PoseDetection>` using deli-base types.

**Dependencies:** Task 1

**Files:**

- Create: `crates/deli-infer/src/postprocess.rs`
- Modify: `crates/deli-infer/src/lib.rs` (add module)
- Test: `crates/deli-infer/tests/postprocess_test.rs`

**Key Decisions / Notes:**

- `pub(crate) fn postprocess(pred: &candle_core::Tensor, original_hw: (usize, usize), model_hw: (usize, usize), conf_threshold: f32, nms_threshold: f32) -> Result<Vec<PoseDetection>, InferError>`.
- Steps: (1) squeeze batch dim, (2) iterate predictions, (3) filter by confidence threshold, (4) extract bbox (cx,cy,w,h → Rect via `Rect::from_min_max`), (5) extract 17 keypoints as `[Keypoint; 17]`, (6) run NMS on collected bboxes, (7) scale coordinates from model space to original image space.
- NMS: Sort by confidence descending, greedily keep boxes with IoU < threshold against all previously kept boxes. Confidence filtering (step 3) reduces 8400 raw anchors to typically <100 candidates before NMS runs, so O(N²) IoU comparisons are fast (~50-100 boxes).
- IoU function: intersection area / union area, same formula as candle-examples.
- Coordinate scaling: `x * (original_w / model_w)`, same for y, clamped to image bounds.
- `conf_threshold` default: 0.25, `nms_threshold` default: 0.45.

**Definition of Done:**

- [x] All tests pass
- [x] No diagnostics errors
- [x] NMS correctly removes overlapping boxes (test with hand-crafted bbox list)
- [x] IoU computation matches expected values for known bbox pairs
- [x] Coordinate scaling maps model-space coords to original image space
- [x] Output `PoseDetection` uses `Rect<f32>` with origin+size (not min/max)
- [x] Keypoints use `Vec2<f32>` for position

**Verify:**

- `cargo test -p deli-infer --test postprocess_test -q` — postprocess tests pass

---

### Task 6: PoseDetector and public API

**Objective:** Implement `PoseDetector` which loads the model from a `.safetensors` file, preprocesses input frames, runs inference, and returns detections. Wire up `Inference::use_pose_detector()`. Finalize `lib.rs` re-exports.

**Dependencies:** Task 2, Task 4, Task 5

**Files:**

- Create: `crates/deli-infer/src/detector.rs`
- Modify: `crates/deli-infer/src/inference.rs` (replace `todo!()` in `use_pose_detector()`)
- Modify: `crates/deli-infer/src/lib.rs` (add module, finalize public re-exports)
- Test: `crates/deli-infer/tests/detector_test.rs`

**Key Decisions / Notes:**

- `PoseDetector` struct fields:
  - `model: Arc<YoloV8Pose>` — shared for async spawn_blocking
  - `device: candle_core::Device`
  - `model_hw: (usize, usize)` — stored from construction (max dim 640, divisible by 32)
  - `conf_threshold: f32` (default 0.25)
  - `nms_threshold: f32` (default 0.45)
- Constructor (called from `Inference::use_pose_detector`):
  1. **Auto-detect model size** from safetensors: read the file with `safetensors::SafeTensors::deserialize()`, inspect `net.b1.0.conv.weight` shape (first conv layer), compute `width = out_channels / 64.0`, look up matching `Multiples` variant (0.25→N, 0.50→S, 0.75→M, 1.00→L, 1.25→X). Return `InferError::Shape` if width doesn't match any known variant.
  2. Load safetensors via `unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }`
  3. Construct `YoloV8Pose::load(vb, multiples)` with the auto-detected multiples
  4. Wrap model in `Arc`
- `pub fn with_thresholds(mut self, conf: f32, nms: f32) -> Self` — builder-style threshold override.
- Preprocessing (`preprocess` method):
  1. Input: `&Tensor<f32>` in HWC layout, RGB, 0-255 range
  2. Compute target size: maintain aspect ratio, max dim 640, both dims divisible by 32
  3. Create candle tensor from deli-base data: `Tensor::from_vec(data, (h, w, 3), &device)`
  4. Permute HWC → CHW: `.permute((2, 0, 1))`
  5. Resize to target via `upsample_nearest2d`
  6. Normalize: `(tensor / 255.0)`, add batch dim `.unsqueeze(0)`
  7. Return `(candle_tensor, original_hw)`
- `pub async fn detect(&self, frame: &Tensor<f32>) -> Result<Vec<PoseDetection>, InferError>`:
  1. Preprocess on current thread (cheap)
  2. Clone Arc, spawn_blocking for `model.forward(&input)` (expensive)
  3. Postprocess on current thread (cheap)
- Public re-exports from `lib.rs`: `Inference`, `PoseDetector`, `InferError`, `PoseDetection`, `Keypoint`, `CocoKeypoint`.

**Definition of Done:**

- [x] All tests pass
- [x] No diagnostics errors
- [x] `Inference::cpu().use_pose_detector(path)` constructs successfully with valid model file and auto-detects model size
- [x] Preprocessing converts HWC `[480, 640, 3]` f32 tensor to NCHW `[1, 3, 480, 640]` candle tensor normalized to 0-1
- [x] `detect()` is async and returns `Vec<PoseDetection>`
- [x] Public API surface is clean: only types from this task + Task 1 are re-exported
- [x] `with_thresholds()` builder method works
- [x] `cargo build -p deli-infer -q` succeeds with clean public API

**Verify:**

- `cargo test -p deli-infer --test detector_test -q` — detector tests pass
- `cargo test -p deli-infer -q` — all crate tests pass
- `cargo build -p deli-infer -q` — final build clean

## Testing Strategy

- **Unit tests:** Each task has a dedicated test file. Model blocks tested with random weights via `VarMap` (verify output shapes). Postprocessing tested with synthetic tensors (hand-crafted bbox/keypoint data). Types tested for construction and trait impls.
- **Integration tests:** Task 6 test exercises the full pipeline: construct Inference → create PoseDetector with random model (VarMap) → preprocess a synthetic tensor → run detect → verify output type structure. A separate manual test with a real `.safetensors` model verifies end-to-end accuracy (not automated — requires model download).
- **Manual verification:** Use `models/yolov8n-pose.safetensors` (already in repo), load a test image, run detect, verify keypoints are in reasonable positions.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| candle API differences between 0.9.x point releases | Low | Med | Use `..Default::default()` for config structs; pin `candle-core = "0.9"` (not exact patch) |
| Model weight key names don't match port | Med | High | Use same `vb.pp()` prefix strings as candle-examples verbatim; test with real model early |
| Auto-detection misidentifies model size from weight shapes | Low | Med | Use first conv layer output channels (64*width) which is unambiguous across all 5 variants; return clear error if width doesn't match any known variant |
| `spawn_blocking` JoinError on panic in forward pass | Low | Med | Map JoinError to InferError::Runtime; candle forward shouldn't panic in normal operation |
| `upsample_nearest2d` resize quality degrades detection accuracy | Med | Low | Document that nearest-neighbor is used; users can pre-resize with bilinear interpolation externally. Future improvement: add bilinear resize. |
| Large model files slow down CI/testing | Med | Low | Unit tests use VarMap random weights (no model file needed); integration test with real model is manual only |
| File exceeds 300-line limit during implementation | Low | Low | Model code is pre-split into 5 files (blocks, backbone, neck, head, mod), each estimated under 220 lines |

## Open Questions

- None — all design decisions resolved during exploration.

### Deferred Ideas

- Auto-download models from HuggingFace Hub via `hf-hub` crate
- Batch inference (multiple frames at once)
- Object detection mode (YoloV8 without pose)
- Bilinear/bicubic resize in preprocessing for better accuracy
- Model warmup/compilation on first inference
- Keypoint skeleton connection drawing utility
