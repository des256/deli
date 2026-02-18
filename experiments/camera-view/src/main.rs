use deli_base::log;
use deli_video::{Camera, CameraConfig, V4l2Camera, VideoFrame};
use deli_image::DecodedImage;
use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

/// Convert HWC RGB buffer to packed ARGB u32 for minifb
fn rgb_to_argb(buf: &[u8], width: usize, height: usize) -> Vec<u32> {
    debug_assert!(
        buf.len() >= width * height * 3,
        "RGB buffer too small: expected {} bytes, got {}",
        width * height * 3,
        buf.len()
    );
    let mut argb = Vec::with_capacity(width * height);
    for i in 0..width * height {
        let idx = i * 3;
        let r = buf[idx] as u32;
        let g = buf[idx + 1] as u32;
        let b = buf[idx + 2] as u32;
        argb.push((r << 16) | (g << 8) | b);
    }
    argb
}

/// Decode a camera VideoFrame into an RGB tensor.
async fn frame_to_rgb(frame: VideoFrame) -> Result<deli_base::Tensor<u8>, Box<dyn std::error::Error>> {
    match frame {
        VideoFrame::Rgb(tensor) => Ok(tensor),
        VideoFrame::Jpeg(data) => match deli_image::decode_image(&data).await? {
            DecodedImage::U8(tensor) => Ok(tensor),
            _ => Err("Unexpected pixel format from JPEG decode".into()),
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    deli_base::init_stdout_logger();

    log::info!("Camera View");
    log::info!("Resolution: {}x{}", WIDTH, HEIGHT);
    log::info!("Controls: ESC to exit");

    // Initialize camera
    log::info!("Opening camera...");
    let config = CameraConfig::default()
        .with_width(WIDTH as u32)
        .with_height(HEIGHT as u32);
    let mut camera = V4l2Camera::new(config)?;
    log::info!("Camera ready");

    // Create display window
    let mut window = Window::new(
        "Camera View - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    window.set_target_fps(30);

    log::info!("Starting main loop...");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = frame_to_rgb(camera.recv().await?).await?;

        if frame.shape.len() != 3 || frame.shape[2] != 3 {
            log::warn!("Expected [H, W, 3] frame shape, got {:?}", frame.shape);
            continue;
        }

        let frame_h = frame.shape[0];
        let frame_w = frame.shape[1];

        let argb = rgb_to_argb(&frame.data, frame_w, frame_h);
        window.update_with_buffer(&argb, frame_w, frame_h)?;
    }

    log::info!("Exiting...");
    Ok(())
}
