use base::log;
use camera_viewer::Frame;
use com::Client;
use futures_util::StreamExt;
use minifb::{Key, Window, WindowOptions};

const DEFAULT_ADDR: &str = "127.0.0.1:9920";

/// Convert HWC RGB buffer to packed ARGB u32 for minifb
fn rgb_to_argb(buf: &[u8], width: usize, height: usize) -> Vec<u32> {
    let expected = width * height * 3;
    assert!(
        buf.len() >= expected,
        "RGB buffer too small: expected {} bytes, got {}",
        expected,
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
    base::init_stdout_logger();

    // Parse address from args or use default
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ADDR.to_string());

    log::info!("Camera Viewer");
    log::info!("Connecting to: {}", addr);

    // Connect to camera broadcaster
    let mut receiver = Client::<Frame>::connect(&addr).await?;
    log::info!("Connected to camera broadcaster");

    // Receive first frame to get dimensions, or use defaults
    log::info!("Waiting for first frame...");
    let first_frame = receiver.next().await.unwrap()?;
    let width = first_frame.width() as usize;
    let height = first_frame.height() as usize;
    log::info!("Received first frame: {}x{}", width, height);

    // Create display window
    let mut window = Window::new(
        "Camera Viewer - ESC to exit",
        width,
        height,
        WindowOptions::default(),
    )?;

    window.set_target_fps(30);

    log::info!("Starting display loop...");

    // Display first frame
    let argb = rgb_to_argb(first_frame.data(), width, height);
    window.update_with_buffer(&argb, width, height)?;

    // Main loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = receiver.next().await.unwrap()?;

        let fw = frame.width() as usize;
        let fh = frame.height() as usize;

        // Skip frames with unexpected dimensions
        if fw != width || fh != height {
            log::warn!(
                "Frame dimension mismatch: expected {}x{}, got {}x{}",
                width,
                height,
                fw,
                fh
            );
            continue;
        }

        // Convert RGB to ARGB and display
        let argb = rgb_to_argb(frame.data(), width, height);
        window.update_with_buffer(&argb, width, height)?;
    }

    log::info!("Exiting...");
    Ok(())
}
