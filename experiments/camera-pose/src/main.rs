mod draw;

use deli_base::log;
use deli_video::{CameraConfig, V4l2Camera, VideoFrame};
use futures_util::StreamExt;
use deli_image::Image;
use deli_infer::backends::OnnxBackend;
use deli_infer::{Device, ModelSource, YoloPoseEstimator};
use draw::{draw_skeleton, rgb_to_argb};
use minifb::{Key, Window, WindowOptions};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const KEYPOINT_THRESHOLD: f32 = 0.001;

/// Decode a camera VideoFrame into an RGB tensor.
async fn frame_to_rgb(frame: VideoFrame) -> Result<deli_base::Tensor<u8>, Box<dyn std::error::Error>> {
    match frame {
        VideoFrame::Rgb(tensor) => Ok(tensor),
        VideoFrame::Jpeg(data) => match deli_image::decode_image(&data).await? {
            Image::U8(tensor) => Ok(tensor),
            _ => Err("Unexpected pixel format from JPEG decode".into()),
        },
    }
}

/// Convert Tensor<u8> to Tensor<f32> for pose estimator
fn tensor_u8_to_f32(
    t: &deli_base::Tensor<u8>,
) -> Result<deli_base::Tensor<f32>, deli_base::TensorError> {
    deli_base::Tensor::new(t.shape.clone(), t.data.iter().map(|&v| v as f32).collect())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get model path from environment or use default
    let model_path: PathBuf = env::var("DELI_MODEL_PATH")
        .unwrap_or_else(|_| "/home/desmond/models/yolo26n-pose.onnx".to_string())
        .into();

    deli_base::init_stdout_logger();

    log::info!("Camera Pose Experiment");
    log::info!("Model: {}", model_path.display());
    log::info!("Resolution: {}x{}", WIDTH, HEIGHT);
    log::info!("Controls: ESC to exit");

    // Initialize camera
    log::info!("Opening camera...");
    let config = CameraConfig::default()
        .with_width(WIDTH as u32)
        .with_height(HEIGHT as u32);
    let mut camera = V4l2Camera::new(config)?;
    log::info!("Camera ready");

    // Initialize pose estimator
    log::info!("Loading pose model...");
    let backend = OnnxBackend::new(Device::Cuda { device_id: 0 });
    //let backend = OnnxBackend::new(Device::Cpu);
    let mut estimator = YoloPoseEstimator::new(ModelSource::File(model_path), &backend)?;
    log::info!("Model loaded");

    // Create display window
    let mut window = Window::new(
        "Camera Pose - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    // Limit to max 30 FPS
    window.set_target_fps(30);

    log::info!("Starting main loop...");

    // Main loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Capture and decode frame
        let frame = frame_to_rgb(camera.next().await.unwrap()?).await?;

        // Validate frame shape (safety check per plan)
        if frame.shape.len() != 3 || frame.shape[2] != 3 {
            log::warn!("Expected [H, W, 3] frame shape, got {:?}", frame.shape);
            continue;
        }

        // Use actual frame dimensions (V4L2 may negotiate different resolution)
        let frame_h = frame.shape[0];
        let frame_w = frame.shape[1];

        // Convert u8 → f32 for pose estimator
        let frame_f32 = tensor_u8_to_f32(&frame)?;

        // Run pose estimation
        let t_infer = Instant::now();
        let detections = match estimator.estimate(&frame_f32) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Estimation error: {e}");
                continue;
            }
        };
        let infer_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        // Draw skeletons on frame
        let t_rest = Instant::now();
        let mut rgb_buf = frame.data; // Take ownership of buffer
        for detection in &detections {
            draw_skeleton(
                &mut rgb_buf,
                frame_w,
                frame_h,
                detection,
                KEYPOINT_THRESHOLD,
            );
        }

        // Convert RGB → ARGB and display
        let argb = rgb_to_argb(&rgb_buf, frame_w, frame_h);
        window.update_with_buffer(&argb, frame_w, frame_h)?;
        let rest_ms = t_rest.elapsed().as_secs_f64() * 1000.0;

        log::debug!("inference: {infer_ms:.1}ms | draw+display: {rest_ms:.1}ms | total: {:.1}ms | detections: {}", infer_ms + rest_ms, detections.len());
    }

    log::info!("Exiting...");
    Ok(())
}
