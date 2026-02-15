mod draw;

use deli_camera::{Camera, CameraConfig, V4l2Camera};
use deli_infer::{Device, ModelSource, YoloPoseEstimator};
use draw::{draw_skeleton, rgb_to_argb};
use minifb::{Key, Window, WindowOptions};
use std::env;
use std::path::PathBuf;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const KEYPOINT_THRESHOLD: f32 = 0.3;

/// Convert deli_base Tensor<u8> to deli_math Tensor<f32> for pose estimator
fn tensor_u8_to_f32(t: &deli_base::Tensor<u8>) -> Result<deli_math::Tensor<f32>, deli_math::TensorError> {
    deli_math::Tensor::new(
        t.shape.clone(),
        t.data.iter().map(|&v| v as f32).collect(),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get model path from environment or use default
    let model_path: PathBuf = env::var("DELI_MODEL_PATH")
        .unwrap_or_else(|_| "models/yolov8n-pose.onnx".to_string())
        .into();

    println!("Camera Pose Experiment");
    println!("Model: {}", model_path.display());
    println!("Resolution: {}x{}", WIDTH, HEIGHT);
    println!("Controls: ESC to exit");
    println!();

    // Initialize camera
    println!("Opening camera...");
    let config = CameraConfig::default()
        .with_width(WIDTH as u32)
        .with_height(HEIGHT as u32);
    let mut camera = V4l2Camera::new(config)?;
    println!("Camera ready");

    // Initialize pose estimator
    println!("Loading pose model...");
    let mut estimator = YoloPoseEstimator::new(
        ModelSource::File(model_path),
        Device::Cpu,
    )?;
    println!("Model loaded");

    // Create display window
    let mut window = Window::new(
        "Camera Pose - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    // Limit to max 30 FPS
    window.set_target_fps(30);

    println!("Starting main loop...");

    // Main loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Capture frame
        let frame = camera.recv().await?;

        // Validate frame shape (safety check per plan)
        if frame.shape.len() != 3 || frame.shape[2] != 3 {
            eprintln!(
                "Warning: Expected [H, W, 3] frame shape, got {:?}",
                frame.shape
            );
            continue;
        }

        // Convert u8 → f32 for pose estimator
        let frame_f32 = tensor_u8_to_f32(&frame)?;

        // Run pose estimation
        let detections = estimator.estimate(&frame_f32)?;

        // Draw skeletons on frame
        let mut rgb_buf = frame.data; // Take ownership of buffer
        for detection in &detections {
            draw_skeleton(
                &mut rgb_buf,
                WIDTH,
                HEIGHT,
                detection,
                KEYPOINT_THRESHOLD,
            );
        }

        // Convert RGB → ARGB and display
        let argb = rgb_to_argb(&rgb_buf, WIDTH, HEIGHT);
        window.update_with_buffer(&argb, WIDTH, HEIGHT)?;
    }

    println!("Exiting...");
    Ok(())
}
