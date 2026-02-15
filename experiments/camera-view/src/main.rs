use deli_camera::{Camera, CameraConfig, V4l2Camera};
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Camera View");
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

    // Create display window
    let mut window = Window::new(
        "Camera View - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    window.set_target_fps(30);

    println!("Starting main loop...");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = camera.recv().await?;

        if frame.shape.len() != 3 || frame.shape[2] != 3 {
            eprintln!(
                "Warning: Expected [H, W, 3] frame shape, got {:?}",
                frame.shape
            );
            continue;
        }

        let frame_h = frame.shape[0];
        let frame_w = frame.shape[1];

        let argb = rgb_to_argb(&frame.data, frame_w, frame_h);
        window.update_with_buffer(&argb, frame_w, frame_h)?;
    }

    println!("Exiting...");
    Ok(())
}
