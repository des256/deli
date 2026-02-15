# Camera-Pose Experiment Implementation Plan

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

**Goal:** Create a real-time pose skeleton viewer experiment in `experiments/camera-pose/` that captures webcam frames via `deli-camera`, runs YOLO pose detection via `deli-infer`, draws COCO skeleton overlays onto the frame, and displays the result in a window using `minifb`.

**Architecture:** A single-binary Rust project with an async main loop. Each iteration: `camera.recv()` → convert `Tensor<u8>` to `Tensor<f32>` → `estimator.estimate()` → draw skeletons/keypoints onto the raw u8 frame buffer → push ARGB pixels to `minifb` window. Drawing is done with simple Bresenham line and filled-circle routines directly on the pixel buffer — no external 2D graphics crate needed.

**Tech Stack:** `deli-camera` (V4L2), `deli-infer` (ONNX pose), `deli-base` (Tensor/Rect/Vec2), `minifb` (window display), `tokio` (async runtime).

## Scope

### In Scope

- New `experiments/camera-pose/` Rust binary crate
- Workspace membership for `experiments/*`
- Camera capture via `deli-camera` V4L2 backend
- Pose estimation via `deli-infer` YoloPoseEstimator
- COCO 17-keypoint skeleton drawing (lines + keypoint dots)
- `minifb` window displaying annotated frames in real-time
- Graceful exit on window close or Escape key

### Out of Scope

- GPU-accelerated rendering (we draw directly into pixel buffers)
- Recording/saving video or screenshots
- Multiple camera support
- CLI argument parsing (hardcoded defaults, adjustable via constants)
- Any changes to existing `deli-*` crates

## Prerequisites

- A YOLO pose model in ONNX format (e.g., `yolov8n-pose.onnx`) available at a known path
- A V4L2-compatible webcam at `/dev/video0`
- ONNX Runtime libraries installed (for `deli-infer` onnx feature)

## Context for Implementer

- **Patterns to follow:** The `deli-camera` V4L2 backend uses async `recv()` returning `Tensor<u8>` in HWC `[H, W, 3]` layout (see `crates/deli-camera/src/v4l2.rs:34`). The `YoloPoseEstimator::estimate()` expects `Tensor<f32>` in `[H, W, 3]` with values in `[0, 255]` range (see `crates/deli-infer/src/pose/estimator.rs:74`).
- **Conventions:** Workspace crates live under `crates/`. Experiments are a new `experiments/` top-level folder. The workspace `Cargo.toml` uses `members = ["crates/*"]` — we need to add `"experiments/*"`.
- **Key files:**
  - `crates/deli-camera/src/traits.rs` — `Camera` trait with async `recv()`
  - `crates/deli-camera/src/config.rs` — `CameraConfig` builder (default 640x480@30fps)
  - `crates/deli-infer/src/pose/estimator.rs` — `YoloPoseEstimator::new()` and `estimate()`
  - `crates/deli-infer/src/pose/types.rs` — `PoseDetection`, `Keypoint`, `KeypointIndex`, COCO layout
  - `crates/deli-base/src/tensor.rs` — `Tensor<T>` with `shape` and `data` fields
  - `crates/deli-base/src/rect.rs` — `Rect<T>` with `origin` (top-left) and `size`
- **Gotchas:**
  - Camera returns `Tensor<u8>` but estimator needs `Tensor<f32>` — must cast element-wise (values stay in 0..255 range, no normalization needed, preprocess handles /255).
  - `minifb` expects ARGB u32 pixels packed as `0x00RRGGBB`, not RGB byte triples.
  - `Rect.origin` is top-left corner, `Rect.size` is width/height — already in pixel coords after postprocess rescaling.
  - V4l2Camera captures MJPEG which always decodes to 3-channel RGB (JPEG has no alpha). The existing `v4l2.rs:154-159` already validates `DecodedImage::U8` variant. The experiment can safely assume `[H, W, 3]` shape from `camera.recv()`. However, the main loop should validate `frame.shape[2] == 3` as a safety check.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Project scaffold and workspace integration
- [x] Task 2: Drawing primitives (lines, circles, skeleton renderer)
- [x] Task 3: Main loop — camera capture, pose estimation, display

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Project scaffold and workspace integration

**Objective:** Create the `experiments/camera-pose/` binary crate with Cargo.toml, add `experiments/*` to the workspace, and set up a minimal `main.rs` that compiles.

**Dependencies:** None

**Files:**

- Create: `experiments/camera-pose/Cargo.toml`
- Create: `experiments/camera-pose/src/main.rs`
- Modify: `Cargo.toml` (workspace root — add `"experiments/*"` to members)

**Key Decisions / Notes:**

- The crate depends on `deli-camera` (feature `v4l2`), `deli-infer` (feature `onnx`), `deli-base`, `minifb`, and `tokio` (features `rt`, `macros`).
- `main.rs` starts as a minimal `fn main() {}` stub — real logic comes in Task 3.
- `minifb` version: use `0.27` (latest stable).

**Definition of Done:**

- [ ] `experiments/camera-pose/Cargo.toml` exists with correct dependencies
- [ ] `experiments/camera-pose/src/main.rs` compiles
- [ ] `cargo check -p camera-pose` succeeds with no errors
- [ ] Workspace root `Cargo.toml` includes `"experiments/*"`

**Verify:**

- `cargo check -p camera-pose` — compiles without errors

### Task 2: Drawing primitives (lines, circles, skeleton renderer)

**Objective:** Implement a `draw` module with functions to draw lines, filled circles, and a full COCO skeleton onto an RGB pixel buffer. Also implement the HWC `Tensor<u8>` → ARGB `Vec<u32>` conversion for `minifb`.

**Dependencies:** Task 1

**Files:**

- Create: `experiments/camera-pose/src/draw.rs`
- Test: `experiments/camera-pose/tests/draw_tests.rs`

**Key Decisions / Notes:**

- Drawing operates on a mutable `&mut [u8]` slice representing HWC RGB pixels with known `(width, height)`. This avoids copying — we draw directly onto the camera frame's tensor data.
- `draw_line(buf, w, h, x0, y0, x1, y1, color: [u8; 3])` — Bresenham line algorithm, clips to bounds.
- `draw_filled_circle(buf, w, h, cx, cy, radius, color: [u8; 3])` — simple filled circle, clips to bounds.
- `draw_skeleton(buf, w, h, detection: &PoseDetection, kp_threshold: f32)` — draws COCO skeleton connections as lines and keypoints as circles. Only draws keypoints/connections where confidence >= `kp_threshold`.
- `rgb_to_argb(buf: &[u8], w, h) -> Vec<u32>` — converts HWC RGB `[u8]` to packed ARGB `u32` for `minifb`.
- COCO skeleton connections (19 limb pairs):
  ```
  (Nose, LeftEye), (Nose, RightEye), (LeftEye, LeftEar), (RightEye, RightEar),
  (LeftShoulder, RightShoulder), (LeftShoulder, LeftElbow), (RightShoulder, RightElbow),
  (LeftElbow, LeftWrist), (RightElbow, RightWrist),
  (LeftShoulder, LeftHip), (RightShoulder, RightHip),
  (LeftHip, RightHip),
  (LeftHip, LeftKnee), (RightHip, RightKnee),
  (LeftKnee, LeftAnkle), (RightKnee, RightAnkle),
  (Nose, LeftShoulder), (Nose, RightShoulder)  [via midpoint or direct]
  ```
  Actually, the standard COCO skeleton is typically these 19 edges. We'll use distinct colors per limb group (face=cyan, arms=yellow, torso=green, legs=magenta).

**Definition of Done:**

- [ ] `draw_line` correctly draws a line between two points with clipping
- [ ] `draw_filled_circle` correctly draws a filled circle with clipping
- [ ] `draw_skeleton` draws all COCO connections for keypoints above threshold
- [ ] `rgb_to_argb` correctly converts RGB bytes to ARGB u32 values
- [ ] All unit tests pass covering line drawing, circle drawing, color conversion, and skeleton rendering
- [ ] `cargo test -p camera-pose` passes

**Verify:**

- `cargo test -p camera-pose` — all tests pass

### Task 3: Main loop — camera capture, pose estimation, display

**Objective:** Wire everything together in `main.rs`: open camera, load model, create minifb window, run the async capture → estimate → draw → display loop.

**Dependencies:** Task 2

**Files:**

- Modify: `experiments/camera-pose/src/main.rs`
- Modify: `experiments/camera-pose/src/draw.rs` (add `mod draw;` to main)
- Test: `experiments/camera-pose/tests/main_tests.rs`

**Key Decisions / Notes:**

- Use `#[tokio::main]` for the async runtime (needed for `camera.recv()`).
- Main loop structure:
  ```rust
  let mut camera = V4l2Camera::new(CameraConfig::default())?;
  let mut estimator = YoloPoseEstimator::new(ModelSource::File(model_path), Device::Cpu)?;
  let mut window = Window::new("Camera Pose", WIDTH, HEIGHT, ...)?;

  while window.is_open() && !window.is_key_down(Key::Escape) {
      let frame: Tensor<u8> = camera.recv().await?;
      let frame_f32 = tensor_u8_to_f32(&frame);  // cast for estimator
      let detections = estimator.estimate(&frame_f32)?;

      // Draw on the u8 frame buffer
      let mut rgb_buf = frame.data;  // take ownership
      for det in &detections {
          draw_skeleton(&mut rgb_buf, WIDTH, HEIGHT, det, 0.3);
      }

      let argb = rgb_to_argb(&rgb_buf, WIDTH, HEIGHT);
      window.update_with_buffer(&argb, WIDTH, HEIGHT)?;
  }
  ```
- Model path: read from `DELI_MODEL_PATH` env var, defaulting to `models/yolov8n-pose.onnx`.
- Camera dimensions: 640x480 (matching CameraConfig default).
- The `tensor_u8_to_f32` helper: `Tensor::new(t.shape.clone(), t.data.iter().map(|&v| v as f32).collect())`.
- Tests for this task focus on the `tensor_u8_to_f32` conversion function. The full main loop requires hardware (camera + display) and is verified by manual execution.

**Definition of Done:**

- [ ] `main.rs` compiles and runs with a webcam connected
- [ ] Frames display in a `minifb` window with skeleton overlays when people are detected
- [ ] Window closes gracefully on Escape or window close
- [ ] `cargo build -p camera-pose` succeeds with no errors or warnings
- [ ] `cargo test -p camera-pose` passes (unit tests for conversion helper)

**Verify:**

- `cargo build -p camera-pose` — builds without errors
- `cargo test -p camera-pose` — all tests pass
- Manual: `cargo run -p camera-pose` with webcam and model file — displays annotated frames

## Testing Strategy

- **Unit tests:** Drawing primitives (line endpoints, circle bounds, color conversion, skeleton rendering on a known buffer). Tensor u8→f32 conversion.
- **Integration tests:** Not applicable — the integration is the live camera+model+display pipeline which requires hardware.
- **Manual verification:** Run the binary with a webcam and ONNX model, verify real-time skeleton overlay appears on detected people.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `minifb` doesn't compile on target platform | Low | High | `minifb` supports Linux/X11/Wayland natively; fall back to `softbuffer` + `winit` if needed |
| Model inference is too slow for real-time | Med | Med | Use `yolov8n-pose` (nano) for speed; camera drops frames when consumer is slow (existing `try_send` behavior) |
| Camera format mismatch (not MJPEG) | Low | Med | V4l2Camera already validates MJPEG format on init and returns clear error |

## Open Questions

- None — the architecture is straightforward and all APIs are well-understood.

### Deferred Ideas

- GPU inference via CUDA/TensorRT feature flags
- FPS counter overlay
- CLI argument parsing for model path, camera device, resolution
- Recording/screenshot functionality
