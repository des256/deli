use {
    base::*,
    image::*,
    minifb::{Window, WindowOptions},
    video::*,
};

fn frame_to_u32(frame: &VideoFrame) -> Vec<u32> {
    let size = frame.color.size;
    let data = &frame.color.data;
    match frame.color.format {
        PixelFormat::Yuyv => yuyv_to_u32(size, data),
        PixelFormat::Srggb10p => srggb10p_to_u32(size, data),
        PixelFormat::Yu12 => yu12_to_u32(size, data),
        PixelFormat::Jpeg => jpeg_to_u32(data),
        PixelFormat::Rgb8 => rgb_to_u32(size, data),
        PixelFormat::Argb8 => argb_to_u32(size, data),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_stdout_logger();

    // Initialize camera
    let mut videoin = v4l2::create(None)?;
    let size = videoin.size();
    log_info!("resolution: {}x{}", size.x, size.y);
    //let format = videoin.format();
    //log_info!("format: {:?}", format);
    //let frame_rate = videoin.frame_rate();
    //log_info!("frame rate: {}", frame_rate);

    // Create display window
    let mut window = Window::new(
        "Camera View - ESC to exit",
        size.x,
        size.y,
        WindowOptions::default(),
    )?;

    while let Some(frame) = videoin.recv().await {
        let buf = frame_to_u32(&frame);
        window.update_with_buffer(&buf, frame.color.size.x, frame.color.size.y)?;
    }

    Ok(())
}
